from typing import Optional
import math

import torch


class TorchGaussianMixture:
    """
    Minimal torch-native Gaussian Mixture Model.

    Designed as a lightweight replacement for sklearn.mixture.GaussianMixture
    in the specific 1D-distance use case:

        distances -> shape (n,) or (n, 1)
        n_components -> usually 2

    Supports:
    - fit(x)
    - predict_proba(x)
    - means_
    - weights_
    - covariances_
    - means_init
    - weights_init
    - random_state

    Notes
    -----
    This is intentionally minimal:
    - diagonal covariance only;
    - no full sklearn API compatibility;
    - no batching over independent GMMs;
    - no sophisticated initialization beyond quantiles/random fallback.
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1.0e-4,
        reg_covar: float = 1.0e-6,
        random_state: Optional[int] = None,
        warm_start: bool = False,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.warm_start = warm_start

        self.means_init: Optional[torch.Tensor] = None
        self.weights_init: Optional[torch.Tensor] = None

        self.weights_: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.covariances_: Optional[torch.Tensor] = None

        self.converged_: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: Optional[torch.Tensor] = None

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x)

        if x.ndim == 1:
            x = x[:, None]

        if x.ndim != 2:
            raise ValueError(f"Expected x with shape (n,) or (n, d), got {tuple(x.shape)}.")

        if x.shape[0] < self.n_components:
            raise ValueError(
                f"Need at least n_components={self.n_components} samples, "
                f"got {x.shape[0]}."
            )

        return x

    def _make_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.random_state is None:
            return None

        generator = torch.Generator(device=device)
        generator.manual_seed(self.random_state)
        return generator

    def _initialize(self, x: torch.Tensor) -> None:
        n, d = x.shape
        device = x.device
        dtype = x.dtype

        ## Mean initialization
        # If explicit initialization has been given, use it
        if self.means_init is not None:
            means = torch.as_tensor(self.means_init, dtype=dtype, device=device)
            means = means.reshape(self.n_components, d)
        else:
            generator = self._make_generator(device)
            if generator is not None:
                # Randomly sample 'k' unique data points from x
                indices = torch.randperm(n, generator=generator, device=device)[:self.n_components]
                means = x[indices].clone()
            # If random state is None, fall back to deterministic initialization
            else:
                # Simple deterministic quantile initialization for 1D.
                # For d > 1, this still works column-wise but is intentionally basic.
                qs = torch.linspace(
                    0.2,
                    0.8,
                    self.n_components,
                    dtype=dtype,
                    device=device,
                )
                means = torch.quantile(x, qs, dim=0)

        if self.weights_init is not None:
            weights = torch.as_tensor(self.weights_init, dtype=dtype, device=device)
            weights = weights.reshape(self.n_components)
            weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)
        else:
            weights = torch.full(
                (self.n_components,),
                1.0 / self.n_components,
                dtype=dtype,
                device=device,
            )

        # Shared empirical variance as a safe initial covariance.
        var = x.var(dim=0, unbiased=False).clamp_min(self.reg_covar)
        covariances = var.expand(self.n_components, d).clone()

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _estimate_log_gaussian_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns log N(x | mean_k, diag(cov_k)) with shape (n, k).
        """
        if self.means_ is None or self.covariances_ is None:
            raise RuntimeError("Model parameters are not initialized.")

        x_expanded = x[:, None, :]                  # (n, 1, d)
        means = self.means_[None, :, :]             # (1, k, d)
        covariances = self.covariances_[None, :, :] # (1, k, d)

        log_2pi = math.log(2.0 * math.pi)

        log_prob = -0.5 * (
            ((x_expanded - means) ** 2 / covariances).sum(dim=-1)
            + torch.log(covariances).sum(dim=-1)
            + x.shape[1] * log_2pi
        )

        return log_prob

    def _e_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.weights_ is None:
            raise RuntimeError("Model parameters are not initialized.")

        log_gaussian = self._estimate_log_gaussian_prob(x)
        log_weights = torch.log(self.weights_.clamp_min(torch.finfo(x.dtype).eps))
        weighted_log_prob = log_gaussian + log_weights[None, :]

        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1)
        log_resp = weighted_log_prob - log_prob_norm[:, None]
        responsibilities = torch.exp(log_resp)

        mean_log_likelihood = log_prob_norm.mean()
        return mean_log_likelihood, responsibilities

    def _m_step(self, x: torch.Tensor, responsibilities: torch.Tensor) -> None:
        dtype = x.dtype
        eps = torch.finfo(dtype).eps

        nk = responsibilities.sum(dim=0).clamp_min(eps)  # (k,)

        weights = nk / x.shape[0]
        means = responsibilities.T @ x / nk[:, None]

        diff = x[:, None, :] - means[None, :, :]
        covariances = (responsibilities[:, :, None] * diff**2).sum(dim=0) / nk[:, None]
        covariances = covariances.clamp_min(self.reg_covar)

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    @torch.inference_mode()
    def fit(self, x: torch.Tensor) -> "TorchGaussianMixture":
        x = self._prepare_input(x)

        if (
            not self.warm_start
            or self.weights_ is None
            or self.means_ is None
            or self.covariances_ is None
        ):
            self._initialize(x)

        previous_lower_bound = None
        self.converged_ = False

        for i in range(self.max_iter):
            lower_bound, responsibilities = self._e_step(x)
            self._m_step(x, responsibilities)

            self.n_iter_ = i + 1
            self.lower_bound_ = lower_bound

            if previous_lower_bound is not None:
                change = torch.abs(lower_bound - previous_lower_bound)
                if change < self.tol:
                    self.converged_ = True
                    break

            previous_lower_bound = lower_bound

        return self

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        _, responsibilities = self._e_step(x)
        return responsibilities