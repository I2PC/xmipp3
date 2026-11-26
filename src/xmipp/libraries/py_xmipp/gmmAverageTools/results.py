from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class EstimatorResult:
    """
    Dataclass to store the output of an image estimator.

    Attributes
    ----------
    estimate : torch.Tensor
        The estimated class mean in real space (regardless of whether the estimator
        operated in Fourier or real space).
        Its shape matches the spatial dimensions of the input images, excluding the
        batch dimension (i.e., shape is ``images.shape[1:]``).
    weights : torch.Tensor
        Tensor of shape ``(n, 1, ..., 1)`` where ``n`` is the number of input images.
        Has trailing singleton dimensions matching ``estimate.ndim`` to allow
        broadcasting. Contains the weight assigned to each image during estimation,
        or an aggregated global score when local per-pixel weights are used.
    gmm_distances : torch.Tensor, optional
        Unprocessed distances from the last GMM iteration. Set to None for non-GMM estimators.
        Default is None.
    """

    estimate: torch.Tensor
    weights: torch.Tensor
    gmm_distances: torch.Tensor | None = None
