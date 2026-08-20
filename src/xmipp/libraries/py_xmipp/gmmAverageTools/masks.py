from typing import Tuple

import numpy as np


def create_circular_mask(image_shape: Tuple[int, int], radius: float) -> np.ndarray:
    """
    Create a circular mask centered in the image.

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