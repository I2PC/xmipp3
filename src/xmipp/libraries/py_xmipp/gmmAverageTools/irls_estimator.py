from typing import Optional, Tuple

import torch

from xmippPyModules.gmmAverageTools.weights import WeightFunction


class IRLSMEstimator:
    """Iteratively reweighted least-squares solver for robust estimation."""

    def __init__(
        self,
        weight_function: WeightFunction,
        max_iter: int,
        tol: float,
        damping_coef: float = 0.0,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        eps: float = 1.0e-8,
    ):
        self.weight_function = weight_function
        self.max_iter = max_iter
        self.tol = tol
        self.damping_coef = damping_coef
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eps = eps

        self.n_its = None
        self.converged = False

    def _validate_prior(
        self, prior_mean: Optional[torch.Tensor], prior_variance: Optional[torch.Tensor]
    ) -> None:
        if (prior_mean is None) != (prior_variance is None):
            raise ValueError(
                "prior_mean and prior_variance must be provided together, "
                f"got {type(prior_mean) = }, {type(prior_variance) = }"
            )

    @torch.inference_mode()
    def fit_one_iteration(
        self,
        images: torch.Tensor,
        image_variance: torch.Tensor,
        image_std: torch.Tensor,
        reference: torch.Tensor,
        ctf: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs a single iteration of the Reweighted Least Squares update"""
        weights = self.weight_function(images, reference, image_std)

        # Weight capping
        if self.min_weight is not None or self.max_weight is not None:
            weights = torch.clamp_(weights, min=self.min_weight, max=self.max_weight)

        if weights.ndim == 1:
            # Reshape weights to shape (batch, 1, ..., 1) to broadcast over image batch
            weights = weights.reshape(weights.shape[0], *((1, ) * (images.ndim - 1)))

        # New estimate calculation:
        # x_new = (s_1 / image_variance + prior_mean / prior_variance) /
        #         (s_2 / image_variance + 1 / prior_variance)
        if ctf is None:
            s_1 = torch.sum(weights * images, dim=0)
            s_2 = torch.sum(weights, dim=0)
        else:
            s_1 = torch.sum(weights * ctf * images, dim=0)
            s_2 = torch.sum(weights * ctf.square(), dim=0)

        if prior_mean is None or prior_variance is None:
            # s_2 will only be used in this iteration, can modify in-place
            update = s_1 / (s_2.clamp_min_(self.eps))
        else:
            # Image and prior variance might be used outside this iteration, do not
            # modify them in-place
            safe_image_variance = image_variance.clamp_min(self.eps)
            safe_prior_variance = prior_variance.clamp_min(self.eps)

            numerator = s_1 / safe_image_variance + prior_mean / safe_prior_variance
            denominator = s_2 / safe_image_variance + safe_prior_variance.reciprocal_()

            update = numerator / denominator.clamp_min_(self.eps)

        # Use update damping for calculating the new estimate
        eta = self.damping_coef
        new_estimate = eta * reference + (1.0 - eta) * update

        return new_estimate, weights

    @torch.inference_mode()
    def fit(
        self,
        images: torch.Tensor,
        *,
        image_variance: Optional[torch.Tensor] = None,
        image_std: Optional[torch.Tensor] = None,
        ctf: Optional[torch.Tensor] = None,
        reference: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_variance: Optional[torch.Tensor] = None,
        max_iter_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Executes the full Iteratively Reweighted Least Squares (IRLS) optimization."""
        self._validate_prior(prior_mean, prior_variance)

        # Build default parameters
        if image_variance is None:
            image_variance = images.var(dim=0)
        if image_std is None:
            image_std = image_variance.sqrt()
        if reference is None:
            reference = images.mean(dim=0)
        
        weights = None
        self.converged = False
        max_iter = max_iter_override or self.max_iter

        # Main iterations loop
        for _ in range(max_iter):
            next_reference, weights = self.fit_one_iteration(images,
                image_variance=image_variance,
                image_std=image_std,
                reference=reference,
                ctf=ctf,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )

            # Convergence check
            relative_difference = torch.linalg.norm(next_reference - reference) / torch.linalg.norm(reference)
            # Update reference before possibly breaking out of the loop
            reference = next_reference

            if relative_difference < self.tol:
                self.converged = True
                break
        
        return reference, weights
