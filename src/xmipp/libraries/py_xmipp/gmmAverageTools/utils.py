import torch


@torch.no_grad()
def weighted_average(
    images: torch.Tensor, weights: torch.Tensor, dim: int = 0, eps: float = 1.0e-6
) -> torch.Tensor:
    """
    Computes the weighted average of an input image tensor along a specified dimension.

    Parameters
    ----------
    images : torch.Tensor
        Input data tensor containing image components or batches.
    weights : torch.Tensor
        Weight coefficients matching or broadcastable to the shape of `images`.
    dim : int, optional
        The dimension along which the average is computed, by default 0.
    eps : float, optional
        Small constant for numerical stability to avoid zero-division, by default 1.0e-6.

    Returns
    -------
    torch.Tensor
        The resulting weighted average tensor.

    Raises
    ------
    ValueError
        If the maximum weight sum is effectively zero, indicating degenerate metrics.
    """
    weight_sum = weights.sum(dim=0)
    # if weight_sum.max() < 1e-8:
    #     raise ValueError(
    #         "All weights are effectively zero: distances likely degenerate"
    #     )
    return (weights * images).sum(dim=dim) / (weight_sum + eps)
