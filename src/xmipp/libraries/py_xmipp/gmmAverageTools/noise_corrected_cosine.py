"""Torch-native signal-cosine estimation under additive white input noise.

Image-side quantities are cached once. Reference-noise propagation is exact for
fixed linear coefficients and approximate for coefficients learned from noise.
The corrected ratio is not an unbiased cosine estimator.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional, Union

import torch


@torch.inference_mode()
def estimate_noise_variance(
    images: torch.Tensor, support: torch.Tensor, batch_size: int = 64,
) -> torch.Tensor:
    """Estimate pixel noise variance from 2x2 checkerboard MAD.

    Independent Gaussian pixel noise contributes variance ``4*sigma**2`` to
    ``x00-x10-x01+x11``. Smooth signal approximately cancels; fine-scale signal
    can inflate the estimate. Only blocks wholly inside support are used.
    Images must be float32/float64, before this metric's preprocessing.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    valid = (support[:-1, :-1] & support[1:, :-1]
             & support[:-1, 1:] & support[1:, 1:])
    if not bool(valid.any()):
        raise ValueError("Automatic noise estimation needs a valid 2x2 mask block")
    variances = images.new_empty(images.shape[0])
    for start in range(0, len(images), batch_size):
        block = images[start:start + batch_size]
        differences = (block[:, :-1, :-1] - block[:, 1:, :-1]
                       - block[:, :-1, 1:] + block[:, 1:, 1:])[:, valid]
        median = torch.quantile(differences, 0.5, dim=1, keepdim=True)
        mad = torch.quantile((differences - median).abs(), 0.5, dim=1)
        variances[start:start + len(block)] = (mad / (2 * 0.6744897501960817)).square()
    return variances


class NoiseCorrectedCosine:
    """A cached distance callable for an unchanged real image batch.

    Parameters
    ----------
    images : torch.Tensor
        Finite float32/float64 images with shape (n, h, w). Device and dtype are
        preserved; the batch must not be changed while this cache is used.
    mask : torch.Tensor, optional
        Binary metric support, default whole image. Means use this support.
    noise_variance : 'auto', float or torch.Tensor
        Per-pixel noise variance in the supplied image units, before this
        metric's preprocessing. Auto uses checkerboard MAD, not labels/SNR.
    pool_noise : bool
        Pool automatic variances using their median. Only suitable for a common
        noise variance (image-wise standardization can invalidate this).
    min_signal_fraction : float
        Resolve signal energies only above this fraction of expected noise
        energy. Unresolved inputs have cosine zero, hence distance one.
    clip : bool
        Clip cosine estimates to [-1, 1]. Default False avoids a GMM point mass
        at zero distance. Unclipped distances may be negative or exceed two.
    reference_noise_variance : float
        Pixel variance of an independent reference. Ignored for linear batch
        references with supplied coefficients; zero means a clean reference.
    batch_size : int
        Batch size for initial feature construction and noise estimation.
    filter_sigma : float
        Gaussian smoothing width in pixels for score calculation only. Zero
        disables smoothing. This is not the noise standard deviation.
    min_frequency, max_frequency : float, optional
        Radial band limits in cycles/pixel, independent of Xmipp's existing
        Nyquist-normalized mask cutoffs. Default: no band restriction.
    """

    @torch.inference_mode()
    def __init__(
        self, images: torch.Tensor, *, mask: Optional[torch.Tensor] = None,
        noise_variance: Union[str, float, torch.Tensor] = "auto",
        pool_noise: bool = False, min_signal_fraction: float = 0.05,
        clip: bool = False, reference_noise_variance: float = 0.0,
        batch_size: int = 64,
        filter_sigma: float = 0.0, min_frequency: float = 0.0,
        max_frequency: Optional[float] = None,
    ):
        if (images.ndim != 3 or len(images) == 0
                or images.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(images).all())):
            raise ValueError("Expected finite float32/float64 images of shape (n, h, w)")
        if (batch_size < 1 or not math.isfinite(min_signal_fraction)
                or min_signal_fraction < 0):
            raise ValueError("Invalid batch_size or min_signal_fraction")
        self.images = images
        self.support = (torch.ones(images.shape[1:], dtype=torch.bool, device=images.device)
                        if mask is None else torch.as_tensor(mask, device=images.device))
        if (self.support.shape != images.shape[1:]
                or not bool(((self.support == 0) | (self.support == 1)).all())):
            raise ValueError("mask must be binary and match an image")
        self.support = self.support.bool()
        count = int(self.support.sum())
        if count < 2:
            raise ValueError("At least two pixels must be inside the mask")
        self.trace = images.new_tensor(count - 1)
        self._feature_scale = None
        self.noise_profile = None
        self.spectral_weights = None
        if (not math.isfinite(filter_sigma) or filter_sigma < 0
                or not math.isfinite(min_frequency) or min_frequency < 0
                or (max_frequency is not None and
                    (not math.isfinite(max_frequency) or max_frequency <= min_frequency))):
            raise ValueError("Invalid Gaussian width or radial frequency limits")
        if filter_sigma > 0 or min_frequency > 0 or max_frequency is not None:
            self._configure_filter(count, filter_sigma, min_frequency, max_frequency)
        self.min_signal_fraction = min_signal_fraction
        self.clip = clip
        self._warned_unresolved = False

        if isinstance(noise_variance, str):
            if noise_variance != "auto":
                raise ValueError("noise_variance must be 'auto', a scalar, or shape (n,)")
            variances = estimate_noise_variance(images, self.support, batch_size)
            if pool_noise:
                variances.fill_(torch.quantile(variances, 0.5))
        else:
            variances = torch.as_tensor(noise_variance, dtype=images.dtype, device=images.device)
            variances = torch.broadcast_to(variances, (len(images),)).clone()
        if not bool(torch.isfinite(variances).all() & (variances >= 0).all()):
            raise ValueError("noise_variance must be finite and nonnegative")
        self.noise_variance = variances
        first = self._features(images[:batch_size])
        self.features = first.new_empty((len(images), first.shape[1]))
        self.features[:len(first)] = first
        for start in range(len(first), len(images), batch_size):
            self.features[start:start + batch_size] = self._features(images[start:start + batch_size])
        self.image_energy = self.features.real.square().sum(dim=1)
        if self.features.is_complex():
            self.image_energy += self.features.imag.square().sum(dim=1)
        self.noise_energy = self.trace * variances
        self.signal_energy = self.image_energy - self.noise_energy
        self.independent_reference_variance = images.new_tensor(reference_noise_variance)
        if not bool(torch.isfinite(self.independent_reference_variance)
                    & (self.independent_reference_variance >= 0)):
            raise ValueError("reference_noise_variance must be finite and nonnegative")
        self.set_reference_weights(None)

    def _configure_filter(self, count: int, sigma: float, minimum: float,
                          maximum: Optional[float]) -> None:
        h, w = self.images.shape[1:]
        options = {"device": self.images.device, "dtype": self.images.dtype}
        radius = torch.hypot(torch.fft.fftfreq(h, **options)[:, None],
                             torch.fft.rfftfreq(w, **options)[None, :])
        keep = (radius > 0) & (radius >= minimum)
        if maximum is not None:
            keep &= radius <= maximum
        # Squared Gaussian transfer; rfft interior columns represent two full
        # Fourier coefficients. DC/Nyquist columns already contain their pairs.
        omega = keep * torch.exp(-4 * math.pi**2 * sigma**2 * radius.square())
        multiplicity = self.images.new_full((w // 2 + 1,), 2.0)
        multiplicity[0] = 1.0
        if w % 2 == 0:
            multiplicity[-1] = 1.0
        self.spectral_weights = omega * multiplicity[None, :]
        mask_fft = torch.fft.rfft2(self.support.to(self.images.dtype), norm="ortho")
        # C = diag(m) - m m^T/K is an orthogonal projector. Thus the diagonal
        # of F C C^T F* is K/(h*w) - |F m|^2/K for unit white input noise.
        # Filtering is diagonal in Fourier space: only this diagonal is needed
        # for the exact trace, even though masking correlates frequencies.
        self.noise_profile = (count / (h*w) - mask_fft.abs().square() / count).clamp_min(0)
        self.trace = (self.spectral_weights * self.noise_profile).sum()
        self._frequency_keep = self.spectral_weights > 0
        if not bool(self._frequency_keep.any()):
            raise ValueError("The filter retains no Fourier coefficients")
        self._feature_scale = self.spectral_weights[self._frequency_keep].sqrt()

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        values = images[:, self.support]
        values = values - values.mean(dim=1, keepdim=True)
        if self._feature_scale is None:
            return values
        centered = images.new_zeros(images.shape)
        centered[:, self.support] = values
        spectrum = torch.fft.rfft2(centered, dim=(-2, -1), norm="ortho")
        return spectrum[:, self._frequency_keep] * self._feature_scale

    @torch.inference_mode()
    def set_reference_weights(self, coefficients: Optional[torch.Tensor]) -> None:
        """Track normalized coefficients of a batch reference, or independence.

        None designates an independent reference whose variance was supplied at
        construction. Coefficients must describe the reference actually scored.
        """
        if coefficients is None:
            self.reference_weights = None
            self.reference_variance = self.independent_reference_variance
            self.cross_noise = self.trace.new_zeros(())
            return
        a = torch.as_tensor(coefficients, dtype=self.images.dtype, device=self.images.device)
        if (a.shape != (len(self.images),) or not bool(torch.isfinite(a).all())
                or bool((a < 0).any()) or not bool(torch.isclose(a.sum(), a.new_tensor(1.0)))):
            raise ValueError("Reference coefficients must be normalized nonnegative shape (n,)")
        self.reference_weights = a.clone()
        self.reference_variance = (a.square() * self.noise_variance).sum()
        self.cross_noise = self.trace * a * self.noise_variance

    @torch.inference_mode()
    def distances(self, reference: torch.Tensor) -> torch.Tensor:
        """Calculate one signed corrected distance (1 - cosine) per image."""
        if (reference.shape != self.images.shape[1:] or reference.is_complex()
                or reference.device != self.images.device
                or reference.dtype != self.images.dtype
                or not bool(torch.isfinite(reference).all())):
            raise ValueError("reference must match image shape, device, and real dtype")
        ref = self._features(reference.unsqueeze(0))[0]
        cross = (self.features @ ref.conj()).real - self.cross_noise
        ref_noise_energy = self.trace * self.reference_variance
        ref_signal = torch.vdot(ref, ref).real - ref_noise_energy
        usable = ((self.signal_energy > 0) & (ref_signal > 0)
                  & (self.signal_energy > self.min_signal_fraction * self.noise_energy)
                  & (ref_signal > self.min_signal_fraction * ref_noise_energy))
        denominator = (self.signal_energy.clamp_min(0) * ref_signal.clamp_min(0)).sqrt()
        scores = torch.where(usable, cross / denominator.clamp_min(torch.finfo(reference.dtype).tiny),
                             torch.zeros_like(cross))
        self.last_unresolved = ~usable
        self.last_clipped = scores.abs() > 1
        self.last_scores = scores.clamp(-1, 1) if self.clip else scores
        if bool(self.last_unresolved.any()) and not self._warned_unresolved:
            warnings.warn("Noise correction left signal energies unresolved; those distances are 1. "
                          "Inspect noise estimates or enable filtering.", RuntimeWarning, stacklevel=2)
            self._warned_unresolved = True
        return 1.0 - self.last_scores

    def __call__(self, images: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Implement the repository distance interface for this cached batch."""
        if images is not self.images:
            raise ValueError("This distance cache belongs to a different image batch")
        return self.distances(reference)
