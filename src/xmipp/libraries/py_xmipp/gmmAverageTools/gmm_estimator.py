from collections.abc import Callable

import torch
from xmippPyModules.gmmAverageTools.results import EstimatorResult
from xmippPyModules.gmmAverageTools.torch_gaussian_mixture import TorchGaussianMixture
from xmippPyModules.gmmAverageTools.utils import weighted_average

MIN_ELEMENTS_FOR_GMM = 50
DEFAULT_BAD_WEIGHT = 0.05
DEFAULT_BAD_QUANTILE = 0.95

DistanceFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

class _UnusableGMM(Exception):
    """Fall back to uniform weights after an unusable GMM iteration."""

    def __init__(self, reason: str, distances: torch.Tensor) -> None:
        super().__init__(reason)
        self.distances = distances


class RecursiveGMMEstimator:
    """Recursive robust averaging estimator based on GMM responsibilities."""

    def __init__(
        self,
        distance_function: DistanceFunction,
        max_iter: int = 1,
        tol: float = 1.0e-4,
        initial_bad_mean_quantile: float | None = None,
        initial_bad_weight: float | None = None,
        random_state: int | None = None,
        gmm_max_iter: int = 20,
        gmm_tol: float = 1.0e-4,
        check_degenerate_model: bool = True,
        min_component_separation: float = 0.05,
        min_good_component_weight: float = 0.30,
    ) -> None:
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

        if initial_bad_mean_quantile is not None and not (
            0.5 < initial_bad_mean_quantile < 1.0
        ):
            raise ValueError(
                "initial_bad_mean_quantile must be strictly between 0.5 and 1.0"
            )
        if initial_bad_weight is not None and not (0.0 < initial_bad_weight < 1.0):
            raise ValueError("initial_bad_weight must be strictly between 0.0 and 1.0")
        self.initial_bad_mean_quantile = initial_bad_mean_quantile
        self.initial_bad_weight = initial_bad_weight

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
        bad_weight = (
            DEFAULT_BAD_WEIGHT
            if self.initial_bad_weight is None
            else self.initial_bad_weight
        )

        component_weights = torch.tensor(
            [1.0 - bad_weight, bad_weight],
            dtype=distances.dtype,
            device=distances.device,
        )

        bad_quantile = (
            DEFAULT_BAD_QUANTILE
            if self.initial_bad_mean_quantile is None
            else self.initial_bad_mean_quantile
        )
        quantiles_for_means = torch.tensor(
            [0.5, bad_quantile], dtype=distances.dtype, device=distances.device
        )
        component_means = torch.quantile(distances.reshape(-1), quantiles_for_means)

        self.model.means_init = component_means.reshape(2, 1)
        self.model.weights_init = component_weights

    def _preprocess_distances(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Validate and standardize the distances used for GMM fitting.
        """
        if distances.numel() < MIN_ELEMENTS_FOR_GMM:
            raise _UnusableGMM("too_few_distances", distances)

        spread = distances.std(unbiased=False)
        dist_range = distances.max() - distances.min()
        if not torch.isfinite(spread) or dist_range < 1.0e-8:
            raise _UnusableGMM("constant_distances", distances)

        std = distances.std().clamp_min(1.0e-8)
        return (distances - distances.mean()) / std

    def _get_good_component_idx(self) -> torch.Tensor:
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
        self, model: TorchGaussianMixture, distances: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return raw responsibilities and peak-anchored weights for the good component."""
        good_component = self._get_good_component_idx()
        raw = model.predict_proba(distances)[:, good_component]
        order = torch.argsort(distances.reshape(-1))
        sorted_raw = raw[order]

        peak_idx = int(sorted_raw.argmax())
        sorted_peak = sorted_raw.clone()
        sorted_peak[:peak_idx] = sorted_raw[peak_idx]
        sorted_peak[peak_idx:] = sorted_raw[peak_idx:].cummin(dim=0).values

        peak = torch.empty_like(raw)
        peak[order] = sorted_peak

        return raw, peak

    def _degeneracy_checks(self) -> tuple[bool, bool]:
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
        degenerate_separation : bool
            True if the model fails the separation check, False otherwise
        degenerate_weight : bool
            True if the model fails the minimum weight check, False otherwise
        """
        mean1, mean2 = self._get_model_means()
        variance1, variance2 = self._get_model_variances()

        distance_between_means_sq = (mean2 - mean1) ** 2
        normalized_separation_sq = distance_between_means_sq / (variance1 + variance2)
        degenerate_separation = bool(
            normalized_separation_sq < self.min_component_separation**2
        )

        good_component_weight = self._get_model_component_weights()[
            self._get_good_component_idx()
        ]
        degenerate_weight = good_component_weight < self.min_good_component_weight

        return degenerate_separation, degenerate_weight

    def _fit_one_iteration(
        self,
        images: torch.Tensor,
        reference: torch.Tensor,
        initialize_params: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Performs one iteration of the GMM estimation procedure:
        1. Calculate distances from each image to the reference.
        2. Fit GMM to the distance distribution
        3. Calculate image weights as probabilities given by the GMM.
        4. Update reference as the new weighted average.
        """
        distances = self.distance_function(images, reference)
        preprocessed_distances = self._preprocess_distances(distances)

        # Prepare distances for the TorchGaussianMixture model
        if preprocessed_distances.ndim == 1:
            preprocessed_distances = preprocessed_distances[:, None]

        if initialize_params:
            self._initialize_model_params(preprocessed_distances)

        self.model.fit(preprocessed_distances)

        parameters = (self.model.means_, self.model.covariances_, self.model.weights_)
        finite_parameters = all(torch.isfinite(p).all() for p in parameters)
        valid_covariances = torch.all(self.model.covariances_ > 0)
        valid_weights = torch.all(self.model.weights_ > 0)
        if not (finite_parameters and valid_covariances and valid_weights):
            raise _UnusableGMM("invalid_gmm", distances)

        raw_responsibilities, weights = self._responsibility_weights(
            self.model, preprocessed_distances
        )

        if not all(
            torch.isfinite(w).all() and bool(((w >= 0) & (w <= 1)).all())
            for w in (raw_responsibilities, weights)
        ):
            raise RuntimeError("Got invalid responsibilities from GMM")

        if self.check_degenerate_model and any(self._degeneracy_checks()):
            raise _UnusableGMM("degenerate_components", distances)

        # Reject a collapsed correction before it can corrupt the next reference.
        if weights.mean() < 1.0e-4 or (
            weights.sum() < 0.05 * raw_responsibilities.sum()
        ):
            raise _UnusableGMM("collapsed_weights", distances)

        # Reshape weights to allow broadcasting over images
        weight_shape = (images.shape[0],) + (1,) * (images.ndim - 1)
        weights = weights.view(weight_shape)

        # The weight sum has been checked; do not add a fixed epsilon to it.
        next_reference = weighted_average(images, weights, eps=0.0)
        if not torch.isfinite(next_reference).all():
            raise _UnusableGMM("invalid_reference", distances)
        rel_change = torch.linalg.norm(next_reference - reference) / (
            torch.linalg.norm(reference) + 1.0e-8
        )

        return distances, weights, next_reference, bool(rel_change < self.tol)

    @torch.inference_mode()
    def fit(
        self,
        images: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> EstimatorResult:
        """Fit a robust average by iteratively reweighting images with a GMM.

        Parameters
        ----------
        images : torch.Tensor
            Images of shape ``(n_images, *image_shape)``.
        reference : torch.Tensor, optional
            Initial reference with the shape of one image. Defaults to the image mean.

        Returns
        -------
        EstimatorResult
            The estimate, final weights, and last iteration's unprocessed distances.
            With zero iterations, returns the initial reference, unit weights,
            and zero distances.
        """
        # Reset the GMM to avoid carrying over state from previous fit() calls
        self.model = self._new_model()

        # Get initial reference
        reference = (
            images.mean(dim=0) if reference is None else reference.to(images.device)
        )

        # Convention: if no iterations are performed, weights are all one and distances are all zero
        weights = torch.ones(
            size=(images.shape[0],) + (1,) * (images.ndim - 1),
            dtype=images.dtype,
            device=images.device,
        )
        distances = torch.zeros_like(weights)
        self.converged = False
        self.n_its = 0

        for i in range(self.max_iter):
            self.n_its = i + 1

            try:
                distances, weights, reference, converged = self._fit_one_iteration(
                    images,
                    reference,
                    initialize_params=(i == 0),
                )
            except _UnusableGMM as exc:
                distances = exc.distances
                weights = torch.ones_like(weights)
                reference = images.mean(dim=0)
                break

            if converged:
                self.converged = True
                break

        return EstimatorResult(
            estimate=reference,
            weights=weights.view(-1, 1, 1),
            gmm_distances=distances,
        )
