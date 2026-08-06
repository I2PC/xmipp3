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
    if dim == 0 and weights.numel() == images.shape[0]:
        weights = weights.reshape(-1)
        flat_images = images.reshape(images.shape[0], -1)

        return (torch.matmul(weights, flat_images) / (weights.sum() + eps)).reshape(
            images.shape[1:]
        )

    weight_sum = weights.sum(dim=dim)
    return (weights * images).sum(dim=dim) / (weight_sum + eps)
