from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GMMDiagnostics:
    """
    Dataclass to store diagnostic metrics and parameters associated with fitting a
    Gaussian Mixture Model (GMM).

    Attributes
    ----------
    distances : torch.Tensor
        Tensor containing the distances evaluated for each sample under the GMM fit.
        These are the original, unprocessed distances (for example, they are not
        standardized).
    standardized_distances : bool
        Flag indicating whether the distances have been standardized (i.e., converted
        to z-scores by subtracting mean and dividing by standard deviation).
    means : tuple[float, float]
        The estimated means for the Gaussian mixture components.
    variances : tuple[float, float]
        The estimated variances for the Gaussian mixture components.
    component_weights : tuple[float, float]
        The mixture weights (prior probabilities) associated with each component,
        summing to 1.
    responsibilities : torch.Tensor, optional
        Tensor of shape ``(n, 2)`` containing the posterior probabilities (responsibilities)
        of each sample belonging to each GMM component.
        Default is None.
    """

    distances: torch.Tensor
    standardized_distances: bool
    means: tuple[float, float]
    variances: tuple[float, float]
    component_weights: tuple[float, float]
    responsibilities: torch.Tensor | None = None

    def get_fit_info_dict(self) -> dict[str, float]:
        return {
            "mean1": self.means[0],
            "mean2": self.means[1],
            "variance1": self.variances[0],
            "variance2": self.variances[1],
            "weight_component1": self.component_weights[0],
            "weight_component2": self.component_weights[1],
        }


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
    gmm_diagnostics : GMMDiagnostics, optional
        Diagnostics object for assessing the GMM fit. Set to None for non-GMM estimators.
        Default is None.
    """

    estimate: torch.Tensor
    weights: torch.Tensor
    gmm_diagnostics: GMMDiagnostics | None = None
