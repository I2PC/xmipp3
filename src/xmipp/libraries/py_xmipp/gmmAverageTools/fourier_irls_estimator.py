from typing import Optional, Tuple, Literal

import torch

from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
from xmippPyModules.gmmAverageTools.results import EstimatorResult

WeightApproach = Literal["per-image", "per-coefficient"]


class JointIRLSFourier:
    """
    Fourier estimator using one IRLS solver on the Fourier representation of
    the images. It can operate on the modulus of the complex residual coefficient
    by coefficient (which gives one scalar weight for each Fourier coefficient),
    or on the norm of the full complex residual (which gives one scalar weight
    per image).
    """

    def __init__(
        self,
        irls_solver: IRLSMEstimator,
        eps: float = 1.0e-8,
        weight_approach: WeightApproach = "per-coefficient",
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.solver = irls_solver
        self.eps = eps
        self.weight_approach = weight_approach
        self.mask = mask

    @property
    def max_iter(self):
        return self.solver.max_iter

    def _get_safe_variance(
        self,
        fourier_images: torch.Tensor,
        image_variance: Optional[torch.Tensor] = None,
        image_std: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the variance of the modulus of the Fourier space images.
        Returns the variance and the standard deviation, min-clamped to protect
        against division by zero.
        """
        if image_variance is None:
            image_variance = fourier_images.abs().var(dim=0)
        if image_std is None:
            image_std = image_variance.sqrt()

        image_variance = torch.clamp_min(image_variance, self.eps)
        image_std = torch.clamp_min(image_std, self.eps)

        return image_variance, image_std

    def _get_masked_data(
        self,
        *,
        fourier_images: torch.Tensor,
        image_variance: Optional[torch.Tensor],
        image_std: Optional[torch.Tensor],
        ctf: Optional[torch.Tensor],
        reference: Optional[torch.Tensor],
        prior_mean: Optional[torch.Tensor],
        prior_variance: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if mask is None:
            return (
                fourier_images,
                image_variance,
                image_std,
                ctf,
                reference,
                prior_mean,
                prior_variance,
            )

        fourier_images = fourier_images[:, mask]

        if reference is not None:
            reference = reference[mask]

        if prior_mean is not None:
            prior_mean = prior_mean[mask]

        if prior_variance is not None and prior_variance.numel() > 1:
            prior_variance = prior_variance[mask]

        if ctf is not None:
            ctf = ctf[:, mask]

        if image_variance is not None:
            image_variance = image_variance[mask]

        if image_std is not None:
            image_std = image_std[mask]

        return (
            fourier_images,
            image_variance,
            image_std,
            ctf,
            reference,
            prior_mean,
            prior_variance,
        )

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
        fourier_transform_images: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the low-level IRLS optimization in Fourier space.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing images in real
            space (if ``fourier_transform_images=True``) or Fourier space.
        image_variance : torch.Tensor, optional
            Variance of the Fourier transform modulus of input images.
            Can be a tensor matching ``image_shape`` (per-coefficient variance) or a
            scalar (global image variance).
            Defaults to ``fourier_images.abs().var(dim=0)``.
        image_std : torch.Tensor, optional
            Standard deviation of the Fourier transform modulus. Pre-calculated square
            root of ``image_variance`` to avoid redundant computation.
        ctf : torch.Tensor, optional
            CTF of the input images matching or broadcastable to Fourier images.
            If None, images are assumed to be CTF-corrected.
        reference : torch.Tensor, optional
            Initial reference in the same domain as ``images``.
            Defaults to the average of the Fourier images.
        prior_mean : torch.Tensor, optional
            Prior mean in the same domain as ``images``.
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
        mask : torch.Tensor, optional
            Fourier-space boolean mask to apply to the images before the estimation.
            Only valid when ``weight_approach="per-image"``.

        Returns
        -------
        estimate : torch.Tensor
            Fourier-space estimated robust average tensor.
        weights : torch.Tensor
            Particle weights from final iteration.

        Raises
        ------
        ValueError
            If a ``mask`` is provided when ``weight_approach == "per-coefficient"``.

        Notes
        -----
        The distance function used by ``self.solver`` (which is of type
        ``IRLSMEstimator``) needs to operate correctly with complex tensors.
        """
        mask = self.mask if mask is None else mask

        if self.weight_approach == "per-coefficient" and mask is not None:
            raise ValueError(
                "Cannot provide a mask with per-coefficient Fourier estimators"
            )

        # Make sure all inputs are set to Fourier space
        fourier_images = images
        if fourier_transform_images:
            fourier_images = torch.fft.rfft2(images)
            if prior_mean is not None:
                prior_mean = torch.fft.rfft2(prior_mean)
            if reference is not None:
                reference = torch.fft.rfft2(reference)

        (
            fourier_images_masked,
            image_variance_masked,
            image_std_masked,
            ctf_masked,
            reference_masked,
            prior_mean_masked,
            prior_variance_masked,
        ) = self._get_masked_data(
            fourier_images=fourier_images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            mask=mask,
        )

        # Make sure image variance and std are initialized from the complex modulus
        image_variance_masked, image_std_masked = self._get_safe_variance(
            fourier_images_masked, image_variance_masked, image_std_masked
        )

        # Use the IRLS solver to perform the estimation
        estimate, weights = self.solver.solve(
            images=fourier_images_masked,
            image_variance=image_variance_masked,
            image_std=image_std_masked,
            ctf=ctf_masked,
            reference=reference_masked,
            prior_mean=prior_mean_masked,
            prior_variance=prior_variance_masked,
            max_iter_override=max_iter_override,
        )

        if mask is not None:
            # Aggregate and reshape weights to make averaging possible
            # NOTE: there are other possibilities that could be considered here for
            # local weights, although masking will mostly be used with global weights
            weight_spatial_dims = tuple(range(1, weights.ndim))
            target_weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
            weights = weights.mean(dim=weight_spatial_dims).view(target_weight_shape)

            # Re-calculate estimate with unmasked images
            estimate = IRLSMEstimator.calculate_update(
                images=fourier_images,
                weights=weights,
                ctf=ctf,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                image_variance=image_variance,
                eps=self.eps,
            )

        return estimate, weights

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
        fourier_transform_images: bool = True,
        mask: Optional[torch.Tensor] = None,
    ) -> EstimatorResult:
        """
        Executes Fourier-domain IRLS optimization and returns a real-space EstimatorResult.

        Parameters
        ----------
        images : torch.Tensor
            Input images tensor.
        image_variance : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        image_std : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        ctf : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        reference : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        prior_mean : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        prior_variance : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.
        max_iter_override : int, optional
            See ``JointIRLSFourier.solve()``.
        fourier_transform_images : bool, default=True
            See ``JointIRLSFourier.solve()``.
        mask : torch.Tensor, optional
            See ``JointIRLSFourier.solve()``.

        Returns
        -------
        EstimatorResult
            Dataclass containing:
            - ``estimate``: Real-space reconstructed estimate of shape ``image_shape``.
            - ``weights``: Aggregated weight tensor with shape ``(n_images, 1, ..., 1)``.
        """
        fourier_estimate, weights = self.solve(
            images=images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
            fourier_transform_images=fourier_transform_images,
            mask=mask,
        )

        # Aggregate and reshape weights to (n_images, 1, ..., 1) convention
        # The shape of weights might not match images due to masking
        weight_spatial_dims = tuple(range(1, weights.ndim))
        agg_weights = weights.mean(dim=weight_spatial_dims, keepdim=True)

        # Reshape to (n_images, 1, ..., 1) convention
        target_weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        agg_weights = agg_weights.view(target_weight_shape)

        return EstimatorResult(
            estimate=torch.fft.irfft2(fourier_estimate),
            weights=agg_weights,
        )
