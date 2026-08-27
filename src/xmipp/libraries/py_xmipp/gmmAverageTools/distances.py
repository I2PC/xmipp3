from typing import Callable, Optional, Union

import torch

from xmippPyModules.gmmAverageTools.weights import (
    cosine_similarity,
    cross_correlation_weight,
    tagare_weight,
    tagare_weight_precomputed,
)


DistanceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def cosine_distance(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Compute a dissimilarity derived from cosine-similarity weights.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical-stability threshold.
    inv_type : str, optional
        Strategy used to convert similarity into dissimilarity.

    Returns
    -------
    torch.Tensor
        One dissimilarity value per image.
    """
    weights = cosine_similarity(images, reference, std, eps)
    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def cross_correlation_distance(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    eps: float = 1.0e-8,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Compute a dissimilarity derived from cross-correlation weights.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    eps : float, optional
        Numerical-stability threshold.
    inv_type : str, optional
        Strategy used to convert similarity into dissimilarity.

    Returns
    -------
    torch.Tensor
        One dissimilarity value per image.
    """
    weights = cross_correlation_weight(images, reference, std, eps)
    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def tagare_distance(
    images: torch.Tensor,
    reference: torch.Tensor,
    std: Union[torch.Tensor, float] = 1.0,
    beta: float = 1.0e-6,
    centered_correlation: bool = False,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Compute a dissimilarity derived from Tagare weights.

    Parameters
    ----------
    images : torch.Tensor
        Input image batch.
    reference : torch.Tensor
        Reference image tensor.
    std : torch.Tensor or float, optional
        Unused; kept for interface consistency.
    beta : float, optional
        Scaling factor applied to the squared orthogonal residual.
    centered_correlation : bool, optional
        Whether to use zero-mean cross-correlation in the Tagare weight.
    eps : float, optional
        Numerical-stability threshold.
    inv_type : str, optional
        Strategy used to convert similarity into dissimilarity.

    Returns
    -------
    torch.Tensor
        One dissimilarity value per image.
    """
    weights = tagare_weight(
        images,
        reference,
        std=std,
        beta=beta,
        centered_correlation=centered_correlation,
        eps=eps,
    )

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def tagare_distance_precomputed(
    images_flat: torch.Tensor,
    image_norms: torch.Tensor,
    image_norm_sq: torch.Tensor,
    reference: torch.Tensor,
    beta: float,
    eps: float = 1.0e-6,
    inv_type: str = "neg",
) -> torch.Tensor:
    """
    Compute a Tagare dissimilarity using precomputed image quantities.

    Parameters
    ----------
    images_flat : torch.Tensor
        Flattened image batch with shape ``(n_images, n_pixels)``.
    image_norms : torch.Tensor
        Euclidean norm of each flattened image.
    image_norm_sq : torch.Tensor
        Squared Euclidean norm of each image.
    reference : torch.Tensor
        Current reference image.
    beta : float
        Scaling factor applied to the squared orthogonal residual.
    eps : float, optional
        Numerical-stability threshold.
    inv_type : str, optional
        Strategy used to convert similarity into dissimilarity.

    Returns
    -------
    torch.Tensor
        One Tagare dissimilarity value per image.
    """
    weights = tagare_weight_precomputed(
        images_flat,
        image_norms,
        image_norm_sq,
        reference,
        beta,
        eps,
    )

    return invert_similarity(weights, inv_type=inv_type, eps=eps)


def invert_similarity(
    similarity: torch.Tensor,
    inv_type: Optional[str] = "neg",
    eps: float = 1.0e-6,
    inplace: bool = True,
) -> torch.Tensor:
    """
    Convert a similarity into a distance-like or dissimilarity quantity.

    Parameters
    ----------
    similarity : torch.Tensor
        Similarity scores.
    inv_type : str or None, optional
        Inversion strategy. Supported values are ``"neg"``,
        ``"reciprocal"``, ``"negative_exponential"`` and ``"none"``.
    eps : float, optional
        Minimum value used by reciprocal inversion.
    inplace : bool, optional
        Whether to perform the inversion in-place when possible.

    Returns
    -------
    torch.Tensor
        Inverted similarity values.

    Raises
    ------
    ValueError
        If ``inv_type`` is not recognized.
    """
    if inv_type is None or inv_type.lower() == "none":
        return similarity

    inv_type = inv_type.lower()

    if inv_type in ("neg", "negative"):
        return similarity.neg_() if inplace else similarity.neg()

    if inv_type == "reciprocal":
        if inplace:
            return similarity.clamp_(min=eps).reciprocal_()
        return similarity.clamp(min=eps).reciprocal()

    if inv_type in (
        "negative_exponential",
        "neg_exp",
        "negexp",
    ):
        return (
            similarity.neg_().exp_()
            if inplace
            else torch.exp(-similarity)
        )

    raise ValueError(
        f"Unknown similarity inversion strategy: {inv_type}"
    )
