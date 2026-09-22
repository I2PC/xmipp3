from typing import Optional, Tuple

import torch

from xmippPyModules.gmmAverageTools.weights import WeightFunction
from xmippPyModules.gmmAverageTools.results import EstimatorResult


class IRLSMEstimator:
    """Iteratively reweighted least-squares solver for robust estimation."""

    def __init__(
        self,
        weight_function: WeightFunction,
        max_iter: int,
        tol: float,
        damping_coef: float = 0.0,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        eps: float = 1.0e-8,
    ):
        self.weight_function = weight_function
        self.max_iter = max_iter
        self.tol = tol
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eps = eps

        self.n_its = None
        self.converged = False

    @torch.inference_mode()
    def _validate_prior(
        self, prior_mean: Optional[torch.Tensor], prior_variance: Optional[torch.Tensor]
    ) -> None:
        if (prior_mean is None) != (prior_variance is None):
            raise ValueError(
                "prior_mean and prior_variance must be provided together, "
                f"got {type(prior_mean) = }, {type(prior_variance) = }"
            )

    @torch.inference_mode()
    def _get_safe_variance(
        self,
        images: torch.Tensor,
        image_variance: Optional[torch.Tensor],
        image_std: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates per-pixel image variance and standard deviation, clamping them
        to protect against division by zero
        """
        if image_variance is None:
            image_variance = images.var(dim=0)
        if image_std is None:
            image_std = image_variance.sqrt()

        image_variance = torch.clamp_min(image_variance, self.eps)
        image_std = torch.clamp_min(image_std, self.eps)

        return image_variance, image_std

    @classmethod
    @torch.inference_mode()
    def calculate_update(
        cls,
        images: torch.Tensor,
        weights: torch.Tensor,
        *,
        ctf: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
        image_variance: Optional[torch.Tensor] = None,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:
        # New estimate calculation:
        # x_new = (s_1 / image_variance + prior_mean / prior_variance) /
        #         (s_2 / image_variance + 1 / prior_variance)
        if ctf is None:
            s_1 = torch.sum(weights * images, dim=0)
            s_2 = torch.sum(weights, dim=0)
        else:
            s_1 = torch.sum(weights * ctf * images, dim=0)
            s_2 = torch.sum(weights * ctf.square(), dim=0)

        if prior_mean is None or prior_variance is None:
            # s_2 will only be used in this calculation, can modify in-place
            return s_1 / (s_2.clamp_min_(eps))

        # Assume image variance and prior variance are safe to divide by,
        # since the ``fit`` method ensures it
        reciprocal_prior_variance = 1.0 / prior_variance
        numerator = s_1 / image_variance + prior_mean * reciprocal_prior_variance
        denominator = s_2 / image_variance + reciprocal_prior_variance

        return numerator / denominator.clamp_min_(eps)

    @torch.inference_mode()
    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        reference: torch.Tensor,
        ctf: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a single iteration of the Reweighted Least Squares update"""
        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp_(weights, min=self.min_weight, max=self.max_weight)

        if weights.ndim == 1:
            # Reshape weights to shape (batch, 1, ..., 1) to broadcast over image batch
            weights = weights.reshape(weights.shape[0], *((1,) * (images.ndim - 1)))

        update = IRLSMEstimator.calculate_update(
            images=images,
            weights=weights,
            ctf=ctf,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            image_variance=image_variance,
            eps=self.eps,
        )

        # Use update damping for calculating the new estimate
        eta = self.damping_coef
        new_estimate = eta * reference + (1.0 - eta) * update

        return new_estimate, weights

    @torch.inference_mode()
    def solve(
        self,
        images: torch.Tensor,
        *,
        image_variance: Optional[torch.Tensor] = None,
        image_std: Optional[torch.Tensor] = None,
        ctf: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
        max_iter_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the low-level Iteratively Reweighted Least Squares (IRLS) optimization.

        This method operates domain-agnostically on raw tensors (real or complex).

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing the images to
            be averaged using the robust IRLS procedure, batched along the first
            dimension. These can be real-space or Fourier-space images, provided
            ``self.weight_function`` handles complex values.
        image_variance : torch.Tensor, optional
            Variance of the input images. Can be a tensor matching ``image_shape``
            (per-pixel variance) or a scalar (global image variance).
            Defaults to variance of ``images`` along dimension 0.
        image_std : torch.Tensor, optional
            Standard deviation of the input images. Pre-calculated square root
            of ``image_variance`` to avoid redundant computation.
        ctf : torch.Tensor, optional
            Contrast Transfer Function of the input images matching or broadcastable
            to ``images.shape``. If None, images are assumed to be CTF-corrected.
        reference : torch.Tensor, optional
            Initial reference for robust averaging. Should match ``image_shape``.
            Defaults to the average of all images, i.e. ``images.mean(dim=0)``.
        prior_mean : torch.Tensor, optional
            Prior mean for the estimator. This will bias the produced estimation
            towards the prior mean, serving as a type of regularization (e.g. the
            prior mean might be a tensor of zeros, keeping the values of the
            reconstructed averages closer to zero).
            Its shape should match the shape of one image.
            Cannot be provided without also providing a value for ``prior_variance``.
            If not provided, no regularization will be applied.
        prior_variance : torch.Tensor, optional
            Prior variance for the estimator. This effectively controls the strength
            of the regularization imposed by the prior mean. A higher value of the
            prior variance means a *weaker* regularization.
            Cannot be provided without also providing a value for ``prior_mean``.
            If not provided, no regularization will be applied.
        max_iter_override : int, optional
            Maximum number of IRLS iterations to be performed by the estimator.
            Overrides the estimator's default ``max_iter`` for this run.

        Returns
        -------
        reference : torch.Tensor
            The estimated robust average in the same domain and shape as an input image.
        weights : torch.Tensor
            Element-wise particle weights from the final iteration, matching
            ``images.shape``. Returns a tensor of ones if ``max_iter == 0``.

        Raises
        ------
        ValueError
            If ``max_iter`` is negative.
        """
        max_iter = self.max_iter if max_iter_override is None else max_iter_override

        if max_iter < 0:
            raise ValueError(f"`max_iter` must be non-negative, got {max_iter}.")

        self._validate_prior(prior_mean, prior_variance)

        # Calculate default initial reference if not provided
        if reference is None:
            reference = images.mean(dim=0)

        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        weights = torch.ones(
            size=weight_shape, dtype=images.dtype, device=images.device
        )

        # Handle 0-iteration shortcut (unweighted baseline)
        if max_iter == 0:
            self.converged = True
            return reference, weights

        # Get safe variances and std
        image_variance, image_std = self._get_safe_variance(
            images, image_variance, image_std
        )
        if prior_variance is not None:
            if isinstance(prior_variance, torch.Tensor):
                prior_variance = torch.clamp_min(prior_variance, self.eps)
            else:
                prior_variance = max(prior_variance, self.eps)

        # Main iterations loop
        for _ in range(max_iter):
            next_reference, weights = self._fit_one_iteration(
                images,
                image_variance=image_variance,
                image_std=image_std,
                reference=reference,
                ctf=ctf,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )

            # Convergence check with norm-zero protection
            ref_norm = torch.linalg.norm(reference)
            diff_norm = torch.linalg.norm(next_reference - reference)
            relative_difference = diff_norm / (ref_norm + self.eps)

            # Update reference before possibly breaking out of the loop
            reference = next_reference

            if relative_difference < self.tol:
                self.converged = True
                break

        return reference, weights

    @torch.inference_mode()
    def fit(
        self,
        images: torch.Tensor,
        *,
        image_variance: Optional[torch.Tensor] = None,
        image_std: Optional[torch.Tensor] = None,
        ctf: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
        max_iter_override: Optional[int] = None,
    ) -> EstimatorResult:
        """
        Executes the IRLS optimization and returns a standardized EstimatorResult.

        Input images must be in real space to maintain the ``EstimatorResult``
        invariant that ``estimate`` is a real-space image.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing real-space images.
        image_variance : torch.Tensor, optional
            Variance of the input images. See ``IRLSMEstimator.solve()``.
        image_std : torch.Tensor, optional
            Standard deviation of the input images. See ``IRLSMEstimator.solve()``.
        ctf : torch.Tensor, optional
            Contrast Transfer Function. See ``IRLSMEstimator.solve()``.
        reference : torch.Tensor, optional
            Initial real-space reference. See ``IRLSMEstimator.solve()``.
        prior_mean : torch.Tensor, optional
            Prior mean tensor. See ``IRLSMEstimator.solve()``.
        prior_variance : torch.Tensor, optional
            Prior variance tensor. See ``IRLSMEstimator.solve()``.
        max_iter_override : int, optional
            Overrides default ``max_iter``. See ``IRLSMEstimator.solve()``.

        Returns
        -------
        EstimatorResult
            Dataclass containing:
            - ``estimate``: Real-space robust average tensor of shape ``image_shape``.
            - ``weights``: Aggregated weight scalar per image, formatted as shape
              ``(n_images, 1, ..., 1)``.
        """
        estimate, weights = self.solve(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )

        # Aggregate and reshape weights to (n_images, 1, 1) convention
        spatial_dims = tuple(range(1, images.ndim))  # first dim is batch
        agg_weights = weights.mean(dim=spatial_dims, keepdim=True)

        return EstimatorResult(estimate=estimate, weights=agg_weights)
