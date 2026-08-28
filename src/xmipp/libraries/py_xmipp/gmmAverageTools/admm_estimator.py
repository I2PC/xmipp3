from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass

import torch
import numpy as np

from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
from xmippPyModules.gmmAverageTools.fourier_irls_estimator import JointIRLSFourier


class ImageData(NamedTuple):
    images: torch.Tensor
    variance: torch.Tensor
    std: torch.Tensor


class ADMMData(NamedTuple):
    real: ImageData
    fourier: ImageData
    ctf: Optional[torch.Tensor] = None


@dataclass
class ADMMState:
    reference_real: torch.Tensor
    reference_fourier: torch.Tensor
    dual_vars: torch.Tensor
    mu: float


class ADMMIterationResult(NamedTuple):
    next_real: torch.Tensor
    next_fourier: torch.Tensor
    weights_real: torch.Tensor
    weights_fourier: torch.Tensor
    next_real_transformed: torch.Tensor


class ADMMEstimator:
    """ADMM estimator coupling real-space and Fourier-space IRLS updates."""

    def __init__(
        self,
        irls_real: IRLSMEstimator,
        irls_fourier: JointIRLSFourier,
        max_iter: int,
        initial_mu: float,
        fourier_multiplier: float,
        atol: float = 1.0e-4,
        rtol: float = 1.0e-4,
        eps: float = 1.0e-8,
    ) -> None:
        self.irls_real = irls_real
        self.irls_fourier = irls_fourier
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.initial_mu = initial_mu
        self.fourier_multiplier = fourier_multiplier
        self.eps = eps

        self.converged = False
        self.n_its = None

    def _real_update(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        real_irls_max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solves the real-space subproblem with IRLS"""
        prior_mean = torch.fft.irfft2(
            state.reference_fourier + state.dual_vars / state.mu, norm="ortho"
        )
        return self.irls_real.fit(
            images=data.real.images,
            image_variance=data.real.variance,
            image_std=data.real.std,
            reference=state.reference_real,
            prior_mean=prior_mean,
            prior_variance=1.0 / state.mu,
            max_iter_override=real_irls_max_iter,
        )

    def _fourier_update(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        next_real_transformed: torch.Tensor,
        fourier_irls_max_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solves the Fourier-space subproblem with the joint IRLS approach"""
        prior_mean = next_real_transformed - state.dual_vars / state.mu
        prior_variance = self.fourier_multiplier * (1.0 / state.mu)

        return self.irls_fourier.fit(
            images=data.fourier.images,
            image_variance=data.fourier.variance,
            image_std=data.fourier.std,
            ctf=data.ctf,
            reference=state.reference_fourier,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            max_iter_override=fourier_irls_max_iter,
            fourier_transform_images=False,
        )

    def _fit_one_iteration(
        self,
        *,
        data: ADMMData,
        state: ADMMState,
        real_irls_max_iter: Optional[int] = None,
        fourier_irls_max_iter: Optional[int] = None,
    ) -> ADMMIterationResult:
        # Real update
        next_real, real_update_weights = self._real_update(
            data=data,
            state=state,
            real_irls_max_iter=real_irls_max_iter,
        )

        # Fourier space update
        next_real_transformed = torch.fft.rfft2(next_real, norm="ortho")
        next_fourier, fourier_update_weights = self._fourier_update(
            data=data,
            state=state,
            next_real_transformed=next_real_transformed,
            fourier_irls_max_iter=fourier_irls_max_iter,
        )

        return ADMMIterationResult(
            next_real=next_real,
            next_fourier=next_fourier,
            weights_real=real_update_weights,
            weights_fourier=fourier_update_weights,
            next_real_transformed=next_real_transformed,
        )

    def fit(
        self,
        images: torch.Tensor,
        *,
        images_fourier: Optional[torch.Tensor] = None,
        initial_reference_real: Optional[torch.Tensor] = None,
        initial_reference_fourier: Optional[torch.Tensor] = None,
        image_variance_real: Optional[torch.Tensor] = None,
        image_std_real: Optional[torch.Tensor] = None,
        image_variance_fourier: Optional[torch.Tensor] = None,
        image_std_fourier: Optional[torch.Tensor] = None,
        ctf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        data = self._prepare_data(
            images=images,
            images_fourier=images_fourier,
            image_variance_real=image_variance_real,
            image_std_real=image_std_real,
            image_variance_fourier=image_variance_fourier,
            image_std_fourier=image_std_fourier,
            ctf=ctf,
        )

        state = self._initialize_state(
            data=data,
            initial_reference_real=initial_reference_real,
            initial_reference_fourier=initial_reference_fourier,
        )

        self.converged = False

        for i in range(self.max_iter):
            results = self._fit_one_iteration(
                data=data,
                state=state,
                real_irls_max_iter=min(5 + i, self.irls_real.max_iter),
                fourier_irls_max_iter=min(5 + i, self.irls_fourier.max_iter),
            )

            # Update dual variables
            primal_residual = results.next_fourier - results.next_real_transformed
            state.dual_vars += state.mu * primal_residual

            # Convergence check only every five iterations to save time
            if i % 5 != 4:
                state.reference_real = results.next_real
                state.reference_fourier = results.next_fourier
                continue

            # Calculate residual norms for convergence check before updating references
            primal_norm, dual_norm, eps_primal, eps_dual = self._residuals(
                state=state,
                results=results,
                primal_residual=primal_residual,
            )

            # Update references
            state.reference_real = results.next_real
            state.reference_fourier = results.next_fourier

            # ADMM convergence check
            if primal_norm < eps_primal and dual_norm < eps_dual:
                self.converged = True
                break

            # Penalty parameter update
            state.mu = self._mu_update(state.mu, primal_norm, dual_norm)

        # Calculate final estimate as the mean of real and fourier references
        fourier_ref_to_real = torch.fft.irfft2(state.reference_fourier, norm="ortho")
        estimate = (state.reference_real + fourier_ref_to_real) / 2

        return estimate, results.weights_real, results.weights_fourier

    def _mu_update(self, mu: float, primal_norm: float, dual_norm: float) -> float:
        if primal_norm > 10 * dual_norm:
            return 2.0 * mu
        if dual_norm > 10 * primal_norm:
            return 0.5 * mu
        return mu

    def _residuals(
        self,
        *,
        state: ADMMState,
        results: ADMMIterationResult,
        primal_residual: torch.Tensor,
    ):
        next_real = results.next_real
        next_real_transformed = results.next_real_transformed
        next_fourier = results.next_fourier

        p = next_fourier.numel()  # number of restrictions
        n = next_real.numel()  # number of primal variables
        primal_norm = torch.linalg.norm(primal_residual).item()
        dual_norm = (
            state.mu * torch.linalg.norm(next_fourier - state.reference_fourier).item()
        )

        eps_primal = np.sqrt(p) * self.atol + self.rtol * max(
            torch.linalg.norm(next_real_transformed).item(),
            torch.linalg.norm(next_fourier).item(),
        )
        eps_dual = (
            np.sqrt(n) * self.atol
            + self.rtol * torch.linalg.norm(state.dual_vars).item()
        )

        return primal_norm, dual_norm, eps_primal, eps_dual

    def _prepare_data(
        self,
        images: torch.Tensor,
        images_fourier: Optional[torch.Tensor] = None,
        image_variance_real: Optional[torch.Tensor] = None,
        image_std_real: Optional[torch.Tensor] = None,
        image_variance_fourier: Optional[torch.Tensor] = None,
        image_std_fourier: Optional[torch.Tensor] = None,
        ctf: Optional[torch.Tensor] = None,
    ) -> ADMMData:
        """Prepare real- and Fourier-space image data for ADMM estimation."""
        if images_fourier is None:
            images_fourier = torch.fft.rfft2(images, norm="ortho")

        image_variance_real, image_std_real = self.irls_real._get_safe_variance(
            images,
            image_variance_real,
            image_std_real,
        )

        image_variance_fourier, image_std_fourier = (
            self.irls_fourier._get_safe_variance(
                images_fourier,
                image_variance_fourier,
                image_std_fourier,
            )
        )

        return ADMMData(
            real=ImageData(
                images=images,
                variance=image_variance_real,
                std=image_std_real,
            ),
            fourier=ImageData(
                images=images_fourier,
                variance=image_variance_fourier,
                std=image_std_fourier,
            ),
            ctf=ctf,
        )

    def _initialize_state(
        self,
        data: ADMMData,
        initial_reference_real: Optional[torch.Tensor] = None,
        initial_reference_fourier: Optional[torch.Tensor] = None,
    ) -> ADMMState:
        """Initialize references, dual variables, and penalty parameter."""
        if initial_reference_real is None:
            initial_reference_real = data.real.images.mean(dim=0)

        if initial_reference_fourier is None:
            initial_reference_fourier = data.fourier.images.mean(dim=0)

        return ADMMState(
            reference_real=initial_reference_real,
            reference_fourier=initial_reference_fourier,
            dual_vars=torch.zeros_like(initial_reference_fourier),
            mu=self.initial_mu,
        )
