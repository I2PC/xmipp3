from typing import Optional, Tuple

# import numpy as np
import torch

# from sklearn.mixture import GaussianMixture

from xmippPyModules.gmmAverageTools.distances import DistanceFunction
from xmippPyModules.gmmAverageTools.utils import weighted_average
from xmippPyModules.gmmAverageTools.torch_gaussian_mixture import TorchGaussianMixture


class RecursiveGMMEstimator:
    """Recursive robust averaging estimator based on GMM responsibilities."""

    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        standardize_distances: bool = True,
        random_state: Optional[int] = None,
    ):
        self.model = TorchGaussianMixture(
            n_components=2, random_state=random_state, warm_start=True
        )
        self.distance_function = distance_function
        self.max_iter = max_iter
        self.tol = tol
        self.standardize_distances = standardize_distances

        self.n_its = None
        self.converged = False

    def _new_model(self) -> TorchGaussianMixture:
        """
        Creates a new GaussianMixture model. Useful to reset the object's state.
        """
        model = TorchGaussianMixture(
            n_components=2,
            random_state=self.model.random_state,
            warm_start=True,
        )

        return model

    def _initialize_model_params(self, distances: torch.Tensor) -> None:
        """
        Initializes self.model's component weight and mean parameters:
        - Good (lower distance) class: weight 0.8, mean equal to the 0.2 quantile of distances.
        - Bad (higher distance) class: weight 0.2, mean equal to the 0.8 quantile of distances.
        """
        component_weights = torch.tensor(
            [0.8, 0.2],
            dtype=distances.dtype,
            device=distances.device,
        )

        component_means = torch.quantile(
            distances.reshape(-1),
            1.0 - component_weights,
        )

        self.model.means_init = component_means.reshape(2, 1)
        self.model.weights_init = component_weights

    def _standardize(
        self, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        If self.standardize_distances is True, returns the standardized distances to
        a mean of zero and standard deviation of one.
        """
        if not self.standardize_distances:
            return distances, 0.0, 1.0

        std = distances.std().clamp_min(1.0e-8)
        mean = distances.mean()

        return (distances - mean) / std, mean.item(), std.item()

    def _responsibility_weights(
        self,
        model: TorchGaussianMixture,
        distances: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Calculates the weights assigned to each of the images according the their
        distance to the reference and the fitted GMM.
        The weight of an image is defined as the (posterior) probability of the image
        belonging to the good component of the GMM, given its distance to the reference.
        """
        good_component = torch.argmin(model.means_.mean(dim=1))
        responsibilities = model.predict_proba(distances)[:, good_component]

        # .view(-1, 1, 1) allows the weights to broadcast over image batches
        # NOTE: this would need to be modified to generalize to other dimensional images
        return responsibilities.to(dtype=dtype, device=device).view(-1, 1, 1)

    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        reference: torch.Tensor,
        initialize_params: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Performs one iteration of the GMM estimation procedure:
        1. Calculate distances from each image to the reference.
        2. Fit GMM to the distance distribution
        3. Calculate image weights as probabilities given by the GMM.
        4. Update reference as the new weighted average.
        """
        distances = self.distance_function(images, reference)
        std_distances, _, _ = self._standardize(distances)

        # Prepare distances for the TorchGaussianMixture model
        if std_distances.ndim == 1:
            std_distances = std_distances[:, None]
        
        if initialize_params:
            self._initialize_model_params(std_distances)
        
        # Fit GMM to the distance distribution
        self.model.fit(std_distances)

        # Get weights and update reference
        weights = self._responsibility_weights(
            self.model, std_distances, dtype=images.dtype, device=images.device
        )
        next_reference = weighted_average(images, weights)
        rel_change = torch.linalg.norm(next_reference - reference) / (
            torch.linalg.norm(reference) + 1.0e-8
        )

        return distances, weights, next_reference, bool(rel_change < self.tol)

    @torch.inference_mode()
    def fit(
        self,
        images: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        initialize_params: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Coordinates the whole estimation process
        """
        # Reset the GMM to avoid carrying over state from previous fit() calls
        self.model = self._new_model()

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )

        self.converged = False
        for i in range(self.max_iter):
            distances, weights, next_reference, converged = self._fit_one_iteration(
                images, reference, initialize_params=initialize_params and i == 0
            )

            # Update reference
            reference = next_reference

            # Check convergence
            if converged:
                self.converged = True
                break

        return reference, weights, distances
