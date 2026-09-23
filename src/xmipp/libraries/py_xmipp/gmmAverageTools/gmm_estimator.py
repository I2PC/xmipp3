from typing import Optional, Tuple

import torch

from xmippPyModules.gmmAverageTools.distances import DistanceFunction
from xmippPyModules.gmmAverageTools.utils import weighted_average
from xmippPyModules.gmmAverageTools.torch_gaussian_mixture import TorchGaussianMixture
from xmippPyModules.gmmAverageTools.results import EstimatorResult, GMMDiagnostics


class RecursiveGMMEstimator:
    """Recursive robust averaging estimator based on GMM responsibilities."""

    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        standardize_distances: bool = True,
        random_state: Optional[int] = None,
        gmm_max_iter: int = 20,
        gmm_tol: float = 1.0e-4,
        check_degenerate_model: bool = True,
        min_component_separation: float = 0.05,
        min_good_component_weight: float = 0.30,
    ):
        self.model = TorchGaussianMixture(
            n_components=2,
            max_iter=gmm_max_iter,
            tol=gmm_tol,
            random_state=random_state,
            warm_start=True,
        )

        self.distance_function = distance_function
        self.max_iter = max_iter
        self.tol = tol
        self.standardize_distances = standardize_distances

        self.check_degenerate_model = check_degenerate_model
        self.min_component_separation = min_component_separation
        self.min_good_component_weight = min_good_component_weight

        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol

        self.n_its = None
        self.converged = False

    def _new_model(self) -> TorchGaussianMixture:
        """
        Creates a new GaussianMixture model. Useful to reset the object's state.
        """
        model = TorchGaussianMixture(
            n_components=2,
            max_iter=self.gmm_max_iter,
            tol=self.gmm_tol,
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

    def _get_good_component_idx(self):
        """
        Returns the index of the component of the GMM model with a lower mean.
        """
        return torch.argmin(self.model.means_.mean(dim=1))

    def _get_model_means(self) -> tuple[float, float]:
        """
        Returns the GMM model's two scalar means
        """
        return self.model.means_[0, 0].item(), self.model.means_[1, 0].item()

    def _get_model_variances(self) -> tuple[float, float]:
        """
        Returns the GMM model's two scalar variances
        """
        return self.model.covariances_[0].item(), self.model.covariances_[1].item()

    def _get_model_component_weights(self) -> tuple[float, float]:
        """
        Returns the GMM model's weight for each of its components. The order
        they are returned in matches the internal model's order, which does
        not necessarily mean the 'good' component is the first one. To identify
        the good coomponent use ``self._get_good_component_idx()`
        """
        return self.model.weights_[0].item(), self.model.weights_[1].item()

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
        good_component = self._get_good_component_idx()
        responsibilities = model.predict_proba(distances)[:, good_component]

        # .view(-1, 1, 1) allows the weights to broadcast over image batches
        # NOTE: this would need to be modified to generalize to other dimensional images
        return responsibilities.to(dtype=dtype, device=device).view(-1, 1, 1)

    def _check_degeneracy(
        self,
        model: TorchGaussianMixture,
        min_component_separation: float,
        min_good_component_weight: float,
    ) -> bool:
        """
        Checks whether the two components of a GMM model are degenerate, i.e,
        their means are too close to represent two distinct groups OR the component
        corresponding to 'good' images has too little weight.

        Parameters
        ----------
        model : TorchGaussianMixture
            One-dimensional GMM model with two components already fit to some data
        min_component_separation : float
            Threshold used to check for degeneracy. The model will be considered
            degenerate if
            ``abs(mean_2 - mean_1) / sqrt(variance_1 + variance_2) < min_component_separation``.
        min_component_weight : float
            Minimum weight for the 'good' GMM component. If k is the index of the
            component with a lower mean, the model will be considered degenerate if
            ``model.weights_[k] < min_component_weight``.

        Returns
        -------
        bool
            True if the model is degenerate, False otherwise
        """
        mean1, mean2 = self._get_model_means()
        variance1, variance2 = self._get_model_variances()

        distance_between_means_sq = (mean2 - mean1) ** 2
        normalized_separation_sq = distance_between_means_sq / (variance1 + variance2)
        degenerate_separation = bool(
            normalized_separation_sq < min_component_separation**2
        )

        good_component_weight = self._get_model_component_weights()[
            self._get_good_component_idx()
        ]
        degenerate_weight = good_component_weight < min_good_component_weight

        return degenerate_separation or degenerate_weight

    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        reference: torch.Tensor,
        initialize_params: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
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
    ) -> EstimatorResult:
        """
        Coordinates the whole GMM robust estimation process:
        1. Calculate initial reference (if not provided)
        2. Calculate distances from each image to the reference
        3. Fit a 2-component GMM to the distance distribution
        4. Use GMM model to assign weights to each image
        5. Calculate new reference as the weighted average of the images.
        6. If the change in the reference is small enough or the number
        of iterations exceeds ``self.max_iter``, stop. Otherwise go back
        to step 2, using the newly calculated reference.

        Parameters
        ----------
        images : torch.Tensor
            Tensor of shape ``(n_images, *image_shape)`` containing the images to
            be averaged using the robust IRLS procedure, batched along the first
            dimension of the ``images`` tensor.
        reference : Optional[torch.Tensor], optional
            Initial reference for the robust averaging (e.g. the average of
            all the images). Should match the shape of one image.
            If not provided, it will the default to the average of the
            input images (i.e. ``reference = images.mean(dim=0)``).
        initialize_params : bool, optional
            If True, the GMM model's means will be initialized on the first iteration
            to predetermined values (using the initial distance distribution's 0.2
            and 0.8 quantiles), and the GMM component weights will be initialized to
            0.8 and 0.2, respectively. Default is False.

        Returns
        -------
        torch.Tensor
            The robust average produced by the estimator on its final iteration.
        torch.Tensor or None
            The weights each particle received on the last iteration of the
            estimation process. The robust average output is the average
            of the input images weighted by these weights. Will only be None
            if the maximum number of iterations is set to zero.
            Will only be None if the maximum number of iterations is set to zero.
        torch.Tensor or None
            The distance from each particle to the reference that each particle
            got on the estimator's last iteration. These are the distances that
            the GMM was fit to in order to calculate the final image weights.
            They are not the distances from each image to the output estimate,
            but to the previous reference, which was used as input to the last
            iteration.
            Will only be None if the maximum number of iterations is set to zero.
        """
        # Reset the GMM to avoid carrying over state from previous fit() calls
        self.model = self._new_model()

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )
        weights = None
        distances = None

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

        # Avoid overwriting weights so that the responsibilities are available for diagnostics
        final_weights = weights
        decided_degenerate = None
        if self.check_degenerate_model and weights is not None:
            if self._check_degeneracy(
                self.model,
                min_component_separation=self.min_component_separation,
                min_good_component_weight=self.min_good_component_weight,
            ):
                final_weights = torch.ones_like(weights)
                reference = images.mean(dim=0)
                decided_degenerate = True
            else:
                decided_degenerate = False

        diagnostics = GMMDiagnostics(
            distances=distances,
            standardized_distances=self.standardize_distances,
            means=self._get_model_means(),
            variances=self._get_model_variances(),
            component_weights=self._get_model_component_weights(),
            responsibilities=weights,
            decided_degenerate=decided_degenerate,
        )
        result = EstimatorResult(
            estimate=reference,
            weights=final_weights,
            gmm_diagnostics=diagnostics,
        )

        return result
