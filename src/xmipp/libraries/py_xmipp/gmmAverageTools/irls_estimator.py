from typing import Optional, Tuple

import torch

from xmippPyModules.gmmAverageTools.weights import WeightFunction


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

    def _validate_prior(
        self, prior_mean: Optional[torch.Tensor], prior_variance: Optional[torch.Tensor]
    ) -> None:
        if (prior_mean is None) != (prior_variance is None):
            raise ValueError(
                "prior_mean and prior_variance must be provided together, "
                f"got {type(prior_mean) = }, {type(prior_variance) = }"
            )

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
            # s_2 will only be used in this iteration, can modify in-place
            update = s_1 / (s_2.clamp_min_(self.eps))
        else:
            # Assume image variance and prior variance are safe to divide by,
            # since the ``fit`` method ensures it
            reciprocal_prior_variance = 1.0 / prior_variance
            numerator = s_1 / image_variance + prior_mean * reciprocal_prior_variance
            denominator = s_2 / image_variance + reciprocal_prior_variance

            update = numerator / denominator.clamp_min_(self.eps)

        # Use update damping for calculating the new estimate
        eta = self.damping_coef
        new_estimate = eta * reference + (1.0 - eta) * update

        return new_estimate, weights

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes the full Iteratively Reweighted Least Squares (IRLS) optimization.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing the images to
            be averaged using the robust IRLS procedure, batched along the first
            dimension of the ``images`` tensor.
        image_variance : torch.Tensor, optional
            Variance of the input images. It can be provided as:
            - A tensor of shape ``image_shape``, in which case it will be interpreted
            as the variance of each pixel in the images.
            - A scalar, indicating a global variance value for the whole image.
            If not provided, it defaults to the variance of the ``images`` tensor
            along its first dimension.
        image_std : torch.Tensor, optional
            Standard deviation of the input images. If given as input, it should be
            the element-wise square root of the ``image_variance`` input. It can be
            provided as an argument to avoid repeated computation of
            ``image_variance.sqrt()``.
        ctf : torch.Tensor, optional
            CTF of the input images. In principle it should be a tensor matching
            the shape of ``images``, although it can be any shape broadcastable
            to ``images.shape``. If not provided, the CTF will be ignored, which
            amounts to assuming that the input particles have been previously
            CTF-corrected.
        reference : torch.Tensor, optional
            Initial reference for the robust averaging (e.g. the average of
            all the images). Should match the shape of one image.
            If not provided, it will the default to the average of the
            input images (i.e. ``reference = images.mean(dim=0)``).
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
            If provided, this will override the object's ``max_iter`` attribute.

        Returns
        -------
        torch.Tensor
            The robust average produced by the estimator
        torch.Tensor or None
            The weights each particle received on the last iteration of the
            estimation process. The robust average output is the average
            of the input images weighted by these weights. Will only be None
            if the maximum number of iterations is set to zero.
        """
        self._validate_prior(prior_mean, prior_variance)

        # Get safe image variance and std
        image_variance, image_std = self._get_safe_variance(
            images, image_variance, image_std
        )

        # Clamp prior variance to protect against division by zero
        if prior_variance is not None:
            if isinstance(prior_variance, torch.Tensor):
                prior_variance = torch.clamp_min(prior_variance, self.eps)
            else:
                prior_variance = max(prior_variance, self.eps)

        # Calculate initial reference
        if reference is None:
            reference = images.mean(dim=0)

        weights = None
        self.converged = False
        max_iter = max_iter_override or self.max_iter

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

            # Convergence check
            relative_difference = torch.linalg.norm(
                next_reference - reference
            ) / torch.linalg.norm(reference)

            # Update reference before possibly breaking out of the loop
            reference = next_reference

            if relative_difference < self.tol:
                self.converged = True
                break

        return reference, weights
