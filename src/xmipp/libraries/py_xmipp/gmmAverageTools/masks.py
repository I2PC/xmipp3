from typing import Tuple

import numpy as np


def create_circular_mask(image_shape: Tuple[int, int], radius: float) -> np.ndarray:
    """Create a circular mask centered in the image.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        Shape of the image as (height, width).
    radius : float
        Radius of the circle in pixels.

    Returns
    -------
    np.ndarray
        Boolean array of shape (height, width) where True values fall
        within or on the circle boundary.
    """
    h, w = image_shape
    center = (w // 2, h // 2)
    Y, X = np.ogrid[:h, :w]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    return dist <= radius**2


### ===================
### Fourier space masks
### ===================


def _cutoff_to_pixels(
    cutoff: float,
    image_shape: tuple[int, int],
    unit: str = "pixels",
    pixel_size: float | None = None,
) -> float:
    """Convert a frequency cutoff to pixels (radius in frequency space).

    Parameters
    ----------
    cutoff : float
        Cutoff value expressed in ``unit``.
    image_shape : tuple[int, int]
        Spatial shape of the image as (height, width).
    unit : {"pixels", "normalized", "cycles_per_unit", "distance"}, optional
        Unit in which ``cutoff`` is expressed.

        - ``"pixels"`` — radius in frequency-space pixels; no conversion
          needed (default).
        - ``"normalized"`` — fraction of the Nyquist radius, in
          ``[0, 1]``. The Nyquist radius is taken as ``min(h, w) / 2``.
        - ``"cycles_per_unit"`` — spatial frequency in cycles per the
          same unit as ``pixel_size`` (e.g. cycles/mm). Requires
          ``pixel_size``.
        - ``"distance"`` — real-space resolution in the same unit as
          ``pixel_size`` (e.g. Å, nm, mm). The cutoff is converted to
          the equivalent spatial frequency ``1 / cutoff`` before
          mapping to pixels. Requires ``pixel_size``.

    pixel_size : float or None, optional
        Physical size of one pixel in real-space units (e.g. Å/pixel,
        mm/pixel). Required when ``unit`` is ``"cycles_per_unit"`` or
        ``"distance"``; ignored otherwise.

    Returns
    -------
    float
        Equivalent cutoff radius in frequency-space pixels.

    Raises
    ------
    ValueError
        If ``unit`` is unrecognised, if ``pixel_size`` is not provided
        when required, or if ``cutoff <= 0`` when ``unit="distance"``.
    """
    h, w = image_shape

    if unit == "pixels":
        return cutoff

    if unit == "normalized":
        nyquist_px = min(h, w) / 2.0
        return cutoff * nyquist_px

    if unit in ("cycles_per_unit", "distance"):
        if pixel_size is None:
            raise ValueError(f"pixel_size must be provided when unit='{unit}'.")
        if unit == "distance":
            if cutoff <= 0:
                raise ValueError(
                    f"cutoff must be positive when unit='distance', got {cutoff}."
                )
            # Convert real-space resolution to spatial frequency (cycles/unit)
            cutoff = 1.0 / cutoff

        # cutoff is now in cycles/unit; map to frequency-space pixels
        f_nyquist = 1.0 / (2.0 * pixel_size)  # cycles/unit at Nyquist
        nyquist_px = min(h, w) / 2.0  # Nyquist radius in pixels
        return (cutoff / f_nyquist) * nyquist_px

    raise ValueError(
        f"Unknown unit '{unit}'. Expected 'pixels', 'normalized', "
        "'cycles_per_unit', or 'distance'."
    )


def create_lowpass_rfft_mask(
    image_shape: tuple[int, int],
    cutoff: float,
    unit: str = "pixels",
    pixel_size: float | None = None,
) -> np.ndarray:
    """Create a low-pass filter mask for use with ``np.fft.rfft2`` output.

    Passes frequencies whose distance from DC is less than or equal to
    ``cutoff`` and suppresses everything above it. The mask accounts for
    the non-redundant (right-half) frequency layout produced by
    ``rfft2``, whose output shape is ``(h, w // 2 + 1)``.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Spatial shape of the *original* image as (height, width), i.e.
        the shape **before** calling ``rfft2``.
    cutoff : float
        Cut-off frequency (or resolution). Frequencies closer to DC
        than this value are kept; all others are zeroed out. Interpreted
        according to ``unit``.
    unit : {"pixels", "normalized", "cycles_per_unit", "distance"}, optional
        Unit for ``cutoff`` (default ``"pixels"``). See
        :func:`_cutoff_to_pixels` for details.
    pixel_size : float or None, optional
        Physical size of one pixel in real-space units. Required when
        ``unit`` is ``"cycles_per_unit"`` or ``"distance"``; ignored
        otherwise.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(h, w // 2 + 1)`` aligned with the
        ``rfft2`` output.

    Examples
    --------
    Keep the lowest 20 % of the frequency radius:

    >>> mask = create_lowpass_rfft_mask(img.shape, cutoff=0.2, unit="normalized")

    Keep frequencies corresponding to features larger than 3 Å
    (pixel size 1.5 Å/pixel):

    >>> mask = create_lowpass_rfft_mask(
    ...     img.shape, cutoff=3.0, unit="distance", pixel_size=1.5
    ... )
    """
    h, w = image_shape
    rfft_w = w // 2 + 1
    cutoff_px = _cutoff_to_pixels(cutoff, image_shape, unit, pixel_size)

    row_freq = np.fft.ifftshift(np.arange(h) - h // 2)
    col_freq = np.arange(rfft_w)

    R2 = row_freq[:, np.newaxis] ** 2 + col_freq[np.newaxis, :] ** 2
    return R2 <= cutoff_px**2
