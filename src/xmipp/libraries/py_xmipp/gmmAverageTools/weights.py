from typing import Callable, Union

import torch

WeightFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
TAGARE_CONSTANT: float = 1.0e-5


def cosine_similarity(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Compute cosine-similarity weights between a batch of images and a reference.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch. The first dimension is assumed to be the batch
        dimension.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical-stability threshold used when normalizing vectors.

    Returns
    -------
    torch.Tensor
        One cosine-similarity weight per image, with shape ``(n_images,)``.
    """
    return torch.cosine_similarity(
        images.flatten(1),
        reference.flatten(),
        dim=1,
        eps=eps,
    )


def cross_correlation_weight(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Compute zero-mean cross-correlation weights between images and a reference.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch. The first dimension is assumed to be the batch
        dimension.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical-stability threshold used when normalizing vectors.

    Returns
    -------
    torch.Tensor
        One cross-correlation weight per image, with shape ``(n_images,)``.
    """
    image_dims = tuple(range(1, images.ndim))

    centered_images = images - images.mean(dim=image_dims, keepdim=True)
    centered_reference = reference - reference.mean()

    return cosine_similarity(
        centered_images,
        centered_reference,
        std=std,
        eps=eps,
    )


@torch.no_grad()
def tagare_weight(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    beta: float = 1.0e-6,
    centered_correlation: bool = False,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute Tagare weights between a batch of images and a reference.

    The weight combines a correlation term with an exponential penalty based
    on the component of each image orthogonal to the reference.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch. The first dimension is assumed to be the batch
        dimension.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    beta : float, optional
        Scaling factor applied to the squared orthogonal residual.
    centered_correlation : bool, optional
        If True, use zero-mean cross-correlation for the correlation term.
        Otherwise, use cosine similarity.
    eps : float, optional
        Numerical-stability threshold used when normalizing vectors.

    Returns
    -------
    torch.Tensor
        One Tagare weight per image, with shape ``(n_images,)``.
    """
    image_dims = tuple(range(1, images.ndim))

    cosine_sim = cosine_similarity(images, reference, std, eps).abs_()

    if centered_correlation:
        correlation_term = cross_correlation_weight(images, reference, std, eps).abs_()
    else:
        correlation_term = cosine_sim

    image_norm_sq = torch.linalg.vector_norm(images, dim=image_dims).square_()

    orth_norm_sq = image_norm_sq * (1.0 - cosine_sim.square()).clamp_min_(0.0)

    return orth_norm_sq.mul_(-beta).exp_().mul_(correlation_term)


@torch.no_grad()
def tagare_weight_precomputed(
    images_flat: torch.Tensor,
    image_norms: torch.Tensor,
    image_norm_sq: torch.Tensor,
    reference: torch.Tensor,
    beta: float,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """
    Compute Tagare weights using precomputed image norms.

    This specialized version is intended for repeated comparisons between a
    fixed image batch and changing references.

    Parameters
    ----------
    images_flat : torch.Tensor
        Flattened image batch with shape ``(n_images, n_pixels)``.
    image_norms : torch.Tensor
        Euclidean norm of each flattened image, with shape ``(n_images,)``.
    image_norm_sq : torch.Tensor
        Squared Euclidean norm of each image, with shape ``(n_images,)``.
    reference : torch.Tensor
        Current reference image.
    beta : float
        Scaling factor applied to the squared orthogonal residual.
    eps : float, optional
        Numerical-stability threshold used when normalizing vectors.

    Returns
    -------
    torch.Tensor
        One Tagare weight per image, with shape ``(n_images,)``.
    """
    reference_flat = reference.reshape(-1)
    reference_norm = torch.linalg.vector_norm(reference_flat)

    denominator = image_norms.clamp_min(eps) * reference_norm.clamp_min(eps)

    cosine_sim = torch.mv(images_flat, reference_flat)
    cosine_sim = cosine_sim.div_(denominator).abs_()

    orth_norm_sq = image_norm_sq * (1.0 - cosine_sim.square()).clamp_min_(0.0)

    return orth_norm_sq.mul_(-beta).exp_().mul_(cosine_sim)


def calculate_beta_auto(
    imgs: torch.Tensor,
    mult: float = 1.0,
) -> float:
    """
    Automatically determine the Tagare exponential scaling parameter.

    Parameters
    ----------
    imgs : torch.Tensor
        Input image batch of shape ``(n, h, w)``.
    mult : float, optional
        Multiplier applied to the automatically calculated value.

    Returns
    -------
    float
        Automatically calculated beta value.
    """
    return mult * TAGARE_CONSTANT / imgs.var(dim=(1, 2)).mean().item()


def smooth_redescending_weights_modulus(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float],
    delta,
):
    """
    Computes smooth redescending M-estimator weights using a Gaussian-like influence metric.
    Applies ``.abs()`` to the differences between the images and the reference in
    order to work properly with complex residuals.

    Parameters
    ----------
    images : torch.Tensor
        Images tensor of shape (n, h, w).
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : torch.Tensor | float
        Standard deviation scale factor.
    delta : float
        Tuning parameter governing the rejection scale threshold of outlier features.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n, h, w) containing the smooth redescending weights.
    """
    # In-place abs not supported for complex tensors
    abs_residuals = torch.abs((images - reference) / std)
    variance_scale = delta**2

    # w(r) = exp(- r^2 / delta**2 )
    square_residuals = abs_residuals.square_()
    return square_residuals.neg_().div_(variance_scale).exp_()
