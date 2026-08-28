from typing import Optional, Tuple

import torch

from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator


class JointIRLSFourier:
    """
    Fourier estimator using one IRLS solver on complex Fourier coefficients, meant to
    operate on the modulus of the complex residual.
    """

    def __init__(self, irls_solver: IRLSMEstimator, eps: float = 1.0e-8) -> None:
        self.solver = irls_solver
        self.eps = eps

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Executes the full Iteratively Reweighted Least Squares (IRLS) optimization
        in Fourier space.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing the images to
            be averaged using the robust IRLS procedure, batched along the first
            dimension of the ``images`` tensor.
            This tensor can either contain the real-space representation of the
            images, in which case ``fourier_transform_images`` should be set to
            ``True``; or the Fourier transform of the images, in which case
            ``fourier_transform_images`` should be set to ``False``.
        image_variance : torch.Tensor, optional
            Variance of the complex modulus of the Fourier transform of the
            input images. It can be provided as:
            - A tensor matching the shape of the Fourier transform of one image,
            in which case it will be interpreted as the variance of each frequency
            in the Fourier domain.
            - A scalar, indicating a global variance value for the whole
            (Fourier transform of the) image.
            If not provided, it defaults to the variance of the modulus of the
            Fourier-transformed images along the batch dimension.
        image_std : torch.Tensor, optional
            Standard deviation of the modulus of the Fourier transform of the
            input images. If given as input, it should be the element-wise square
            root of the ``image_variance`` input.
            It can be provided as an argument to avoid repeated computation of
            ``image_variance.sqrt()``.
        ctf : torch.Tensor, optional
            CTF of the input images. In principle it should be a tensor matching
            the shape of the Fourier-transformed images, although it can be any shape
            broadcastable to that. If not provided, the CTF will be ignored, which
            amounts to assuming that the input particles have been previously
            CTF-corrected .
        reference : torch.Tensor, optional
            Initial reference for the robust averaging (e.g. the average of
            all the images).
            Its shape should match the shape of one of the input images. Its domain
            should also match that of the input images: for example, if the input images
            are in real space (``fourier_transform_images=True``), then ``prior_mean``
            should also be in real space.
            If not provided, it will the default to the average of the Fourier
            representation of the input images.
        prior_mean : torch.Tensor, optional
            Prior mean for the estimator. This will bias the produced estimation
            towards the prior mean, serving as a type of regularization (e.g. the
            prior mean might be a tensor of zeros, keeping the values of the
            reconstructed averages closer to zero).
            Its shape should match the shape of one of the input images. Its domain
            should also match that of the input images: for example, if the input images
            are in real space (``fourier_transform_images=True``), then ``prior_mean``
            should also be in real space.
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

        Notes
        -----
        The distance function used by ``self.solver`` (which is of type
        ``IRLSMEstimator``) needs to operate correctly with complex tensors.
        """

        # Make sure all inputs are set to Fourier space
        fourier_images = images
        if fourier_transform_images:
            fourier_images = torch.fft.rfft2(images)
            if prior_mean is not None:
                prior_mean = torch.fft.rfft2(prior_mean)
            if reference is not None:
                reference = torch.fft.rfft2(reference)

        # Make sure image variance and std are initialized from the complex modulus
        image_variance, image_std = self._get_safe_variance(fourier_images)

        # Use the IRLS solver to perform the estimation
        return self.solver.fit(
            images=fourier_images,
            image_variance=image_variance,
            image_std=image_std,
            ctf=ctf,
            reference=reference,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=max_iter_override,
        )
