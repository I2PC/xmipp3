"""Noise-corrected cosine weighting with the repository's Torch GMM solver."""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator
from xmippPyModules.gmmAverageTools.noise_corrected_cosine import NoiseCorrectedCosine
from xmippPyModules.gmmAverageTools.results import EstimatorResult
from xmippPyModules.gmmAverageTools.utils import weighted_average


class NoiseCorrectedCosineEstimator:
    """Fit per-image cosine weights or peak-corrected Torch GMM weights.

    Parameters
    ----------
    weighting : {'gmm', 'cosine'}
        GMM fits unclipped corrected distances by default. Cosine weights are
        ``clip(cosine, 0, 1)**weight_power``. Both average original input images.
    max_iter, tol : int, float
        Outer iteration limit and relative-reference-change tolerance.
    mask : torch.Tensor, optional
        Binary support used by the metric, not a filter on the output average.
    metric_params : dict, optional
        Keyword arguments for NoiseCorrectedCosine.
    gmm_params : dict, optional
        RecursiveGMMEstimator parameters other than its distance function,
        iteration limit, tolerance, and reference-coefficient callback.
    weight_power : float
        Positive exponent for direct cosine weighting only.
    """

    def __init__(self, *, weighting="gmm", max_iter=10, tol=1e-4,
                 mask=None, metric_params=None, gmm_params=None, weight_power=1.0):
        if weighting not in ("gmm", "cosine"):
            raise ValueError("weighting must be 'gmm' or 'cosine'")
        if (not isinstance(max_iter, int) or max_iter < 0 or not math.isfinite(tol)
                or tol < 0 or not math.isfinite(weight_power) or weight_power <= 0):
            raise ValueError("Invalid max_iter, tolerance, or weight_power")
        self.weighting = weighting
        self.max_iter = max_iter
        self.tol = tol
        self.mask = mask
        self.metric_params = dict(metric_params or {})
        self.gmm_params = dict(gmm_params or {})
        reserved = {"distance_function", "max_iter", "tol", "reference_weights_callback"}
        if reserved.intersection(self.gmm_params):
            raise ValueError("Set outer iteration options on the estimator, not gmm_params")
        self.weight_power = weight_power

    @torch.inference_mode()
    def fit(self, images: torch.Tensor, reference: Optional[torch.Tensor] = None,
            *, reference_weights: Optional[torch.Tensor] = None) -> EstimatorResult:
        """Fit a fresh cache/model for the supplied batch.

        No reference initializes the mean with uniform coefficients. A supplied
        linear batch reference requires its normalized ``reference_weights``.
        Without coefficients it is treated as independent, using metric_params'
        ``reference_noise_variance`` (default zero). Nonlinear batch references
        such as a median are not covered by this covariance model.
        """
        params = dict(self.metric_params)
        params.setdefault("mask", self.mask)
        self.metric = NoiseCorrectedCosine(images, **params)
        if reference is None:
            reference = images.mean(dim=0)
            reference_weights = images.new_full((len(images),), 1.0 / len(images))
        else:
            if reference.is_complex():
                raise ValueError("The reference must be a real spatial image")
            reference = reference.to(device=images.device, dtype=images.dtype)
        if reference.shape != images.shape[1:] or not bool(torch.isfinite(reference).all()):
            raise ValueError("reference must be finite and match an image")
        self.metric.set_reference_weights(reference_weights)
        self.fallback_reason = None
        if self.weighting == "gmm":
            self.solver = RecursiveGMMEstimator(
                distance_function=self.metric, max_iter=self.max_iter, tol=self.tol,
                reference_weights_callback=self.metric.set_reference_weights,
                **self.gmm_params)
            result = self.solver.fit(images, reference, reference_weights=reference_weights)
            self.n_its = self.solver.n_its
            self.converged = self.solver.converged
            self.fallback_reason = result.gmm_diagnostics.fallback_reason
            self.last_distances = result.gmm_diagnostics.distances
            return result

        self.n_its = 0
        self.converged = self.max_iter == 0
        self.last_distances = images.new_zeros(len(images))
        weights = images.new_ones((len(images), 1, 1))
        for iteration in range(self.max_iter):
            self.n_its = iteration + 1
            self.last_distances = self.metric.distances(reference)
            weights = (1 - self.last_distances).clamp(0, 1).pow(self.weight_power).view(-1, 1, 1)
            if bool(weights.sum() <= 1e-8):
                warnings.warn("All cosine weights vanished; returning the ordinary mean.",
                              RuntimeWarning, stacklevel=2)
                self.fallback_reason = "collapsed_weights"
                weights = torch.ones_like(weights)
                reference = images.mean(dim=0)
                self.metric.set_reference_weights(weights.reshape(-1) / weights.sum())
                break
            update = weighted_average(images, weights, eps=0.0)
            change = torch.linalg.vector_norm(update - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-8)
            self.metric.set_reference_weights(weights.reshape(-1) / weights.sum())
            reference = update
            if bool(change < self.tol):
                self.converged = True
                break
        return EstimatorResult(estimate=reference, weights=weights)
