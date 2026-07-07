from typing import Callable, Union, Optional

import torch

DistanceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TAGARE_CONSTANT: float = 1.0e-5


def cosine_similarity(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Calculates spatial cosine similarity between each image and the given reference.

    Parameters
    ----------
    images : torch.Tensor
        Input images batch. The first dimension is assumed to be the batch dimension.
    reference : torch.Tensor
        Reference template tensor of shape (h, w).
    std : Union[torch.Tensor, float], optional
        Standard deviation tracking parameter, by default 1.0.
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical stability factor avoiding zero-magnitude vectors during calculation steps,
        by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) holding the cosine similarity between each image and the
        reference.
    """
    return torch.cosine_similarity(
        images.flatten(1), reference.flatten(), dim=1, eps=eps
    )


def cross_correlation(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """
    Evaluates zero-mean structural cross-correlation between each image and
    the reference.

    Parameters
    ----------
    images : torch.Tensor
        Input images batch. The first dimension is assumed to be the batch dimension.
    reference : torch.Tensor
        Reference template tensor. Its shape must match each of the images
        (e.g. ``reference.shape == images[0].shape``).
    std : Union[torch.Tensor, float], optional
        Standard deviation parameter tracking scale boundaries, by default 1.0.
        Unused; kept for interface consistency.
    centered_correlation: bool, optional
        If True, the images and the reference are centered to zero-mean
        before calculating the similarity term. Default is False.
    eps : float, optional
        Numerical stability threshold constant passed down to internal calculations, by default 1.0e-8.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) holding the computed cross-correlation between each image
        and the reference.
    """
    image_dims = tuple(range(1, images.ndim))
    centered_images = images - images.mean(dim=image_dims, keepdim=True)
    centered_reference = reference - reference.mean()

    return cosine_similarity(centered_images, centered_reference, std, eps)


@torch.no_grad()
def tagare_distance(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,  # Maintained for API consistency
    beta: float = 1.0e-6,
    centered_correlation: bool = False,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Computes a robust distance metric based on Tagare's similarity function,
    incorporating cosine similarity and orthogonal residual components.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images of shape (n, h, w). Should work with higher dimensional images
        as long as there is only one batch dimension, and it is the first one.
    reference : torch.Tensor
        Reference image tensor.
    std : Union[torch.Tensor, float], optional
        Currently unused in calculation; maintained for API compatibility, by default 1.0.
    beta : float, optional
        Scaling factor for the orthogonal penalty, by default 1.0e-6.
    centered_correlation: bool, optional
        If True, the cosine similarity term is replaced by cross correlation. The
        difference is whether the images and the reference are centered to zero-mean
        when calculating the similarity term. Default is False.
    eps : float, optional
        Epsilon for numerical stability, by default 1.0e-6.
    inv_type : str, optional
        Method to invert similarity to distance, by default "neg".

    Returns
    -------
    torch.Tensor
        Tensor of shape (n,) holding the calculated distance per image.
    """
    image_dims = tuple(range(1, images.ndim))

    cosine_sim = cosine_similarity(images, reference, std, eps).abs_()

    # First term: cosine similarity (if centered_correlation=False)
    # or cross-correlation (if centered_correlation=True)
    if centered_correlation:
        correlation_term = cross_correlation(images, reference, std, eps).abs_()
    else:
        # Avoid recomputing the cosine similarity
        correlation_term = cosine_sim

    # Second term: norm of the orthogonal component
    image_norm_sq = torch.linalg.vector_norm(images, dim=image_dims).square_()
    orth_norm_sq = image_norm_sq * (1.0 - cosine_sim.square()).clamp_min_(0.0)

    weights = orth_norm_sq.mul_(-beta).exp_().mul_(correlation_term)

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def calculate_beta_auto(imgs: torch.Tensor, mult: float = 1.0) -> float:
    """
    Automatically scales the Tagare exponential scaling factor based on the average
    variance across the provided input images dataset.

    Parameters
    ----------
    images : torch.Tensor
        Input batch dataset images tensor of shape (n, h, w).
    mult : float, optional
        Scalar multiplier adjustment value modifying the baseline parameter scaling, by default 1.0.

    Returns
    -------
    float
        The calculated floating-point automatic beta scaling parameter.
    """
    return mult * TAGARE_CONSTANT / imgs.var(dim=(1, 2)).mean().item()


def invert_similarity(
    similarity: torch.Tensor,
    inv_type: Optional[str] = "neg",
    eps: float = 1e-6,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Inverts a similarity metric into a distance/dissimilarity metric using the
    specified strategy.

    Parameters
    ----------
    similarity : torch.Tensor
        Similarity scores tensor.
    inv_type : str | None, optional
        Inversion type strategy:
        - "neg" / "negative": returns -similarity
        - "reciprocal": returns 1 / similarity
        - "negative_exponential" / "neg_exp" / "negexp": returns exp(-similarity)
        - "none" / None: returns similarity unmodified
        By default "neg".
    eps : float, optional
        Lower bounding clamp value for 'reciprocal' to avoid zero division, by default 1.0e-6.
    inplace : bool, optional
        If True, performs operations in-place on the tensor to save memory, by default True.

    Returns
    -------
    torch.Tensor
        The inverted similarity values acting as a distance/dissimilarity metric.

    Raises
    -------
    ValueError
        If an unrecognized `inv_type` string is provided.
    """
    if inv_type is None or inv_type.lower() == "none":
        return similarity

    inv_type = inv_type.lower()
    if inv_type in ["neg", "negative"]:
        return similarity.neg_() if inplace else similarity.neg()

    if inv_type == "reciprocal":
        return (
            similarity.clamp_(min=eps).reciprocal_()
            if inplace
            else torch.reciprocal(torch.clamp(similarity, min=eps))
        )

    if inv_type in ["negative_exponential", "neg_exp", "negexp"]:
        return similarity.neg_().exp_() if inplace else torch.exp(-similarity)

    raise ValueError(f"Unknown similarity inversion strategy: {inv_type}")
