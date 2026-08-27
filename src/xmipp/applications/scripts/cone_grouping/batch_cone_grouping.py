#!/usr/bin/env python3

from __future__ import annotations

import math
import argparse
from pathlib import Path

import numpy as np
import starfile
import pandas as pd


# The functions ``sample_projection_directions``, ``spherical_to_cartesian``,
# and ``group_projection_directions`` were originally written 
# by Oier Lauzirika Zarrabeitia in https://github.com/oierlauzi/factorem
# They have been vendored for this script
# and modified by Andrés Contreras to add docstrings, and adapt 
# ``group_projection_directions`` to closest-reference grouping.
def sample_projection_directions(n: int) -> np.ndarray:
    """
    Generate approximately uniform directions over a hemisphere.

    Directions are generated using a Fibonacci (golden-angle) lattice.
    Since antipodal directions represent the same projection direction,
    only the hemisphere with z >= 0 is sampled.

    Parameters
    ----------
    n : int
        Number of directions to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2), where each row contains the azimuthal
        angle phi and polar angle theta, in radians.
    """
    out = np.empty((n, 2))

    K = math.pi * (3 - math.sqrt(5))
    i = np.arange(n)
    out[:, 0] = K * i

    z = np.linspace(0.0, 1.0, n)
    np.arccos(z, out=out[:, 1])

    return out


def spherical_to_cartesian(rot: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """
    Convert spherical angles to Cartesian unit vectors.

    The sign convention for ``rot`` and ``tilt`` is chosen to be
    consistent with ``euler_zyz_to_matrix``.

    Parameters
    ----------
    rot : np.ndarray
        Azimuthal angles, in radians.
    tilt : np.ndarray
        Polar angles, in radians. ``rot`` and ``tilt`` must be
        broadcast-compatible.

    Returns
    -------
    np.ndarray
        Cartesian unit vectors with shape ``broadcast_shape + (3,)``,
        where the last dimension contains the x, y, and z components.
    """
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape)
    dtype = np.result_type(rot, tilt, np.float32)
    out = np.empty(batch_shape + (3,), dtype=dtype)

    sin_tilt = np.sin(tilt)

    out[..., 0] = np.cos(rot) * sin_tilt
    out[..., 1] = np.sin(rot) * sin_tilt
    out[..., 2] = np.cos(tilt)

    return out


def group_projection_directions(
    directions: np.ndarray,
    references: np.ndarray,
    consider_mirrors: bool = True,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Assign each direction to its closest reference direction.

    Closeness is measured by angular distance, equivalently by maximizing
    the dot product between unit vectors. If ``consider_mirrors`` is True,
    antipodal directions are treated as equivalent.

    Parameters
    ----------
    directions : np.ndarray
        Unit direction vectors with shape (n_directions, 3).
    references : np.ndarray
        Unit reference vectors with shape (n_references, 3).
    consider_mirrors : bool, default=True
        If True, directions v and -v are considered equivalent.
    batch_size : int, default=1024
        Number of directions processed at once.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_directions,) containing the index of
        the closest reference for each direction.
    """
    n_references = len(references)
    n_directions = len(directions)

    index_dtype = np.min_scalar_type(n_references - 1)
    result = np.empty(n_directions, dtype=index_dtype)

    start = 0
    while start < n_directions:
        end = min(n_directions, start + batch_size)
        direction_batch = directions[start:end]

        cos = direction_batch @ references.T
        if consider_mirrors:
            cos = abs(cos)

        result[start:end] = np.argmax(cos, axis=1)
        start = end

    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-xmd",
        required=True,
        type=Path,
        help="Path to the .xmd file containing the particle metadata",
    )
    parser.add_argument(
        "--out-star",
        required=True,
        type=Path,
        help=(
            "Path to the output .star file, which will contain the metadata "
            "in the input .xmd, plus the grouping indices"
        ),
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=100,
        help="Number of classes to group projection directions in",
    )
    parser.add_argument(
        "--grouping-batch-size",
        type=int,
        default=1024,
        help="Batch size used to process the viewing directions when grouping",
    )
    parser.add_argument(
        "--out-group-column",
        type=str,
        default="cone_group",
        help=(
            "Name for the column in the output metadata file that "
            "contains the index for each particle's group"
        ),
    )

    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    data = pd.DataFrame(starfile.read(args.input_xmd))

    # Xmipp stores Euler angles in degrees
    rot = np.deg2rad(data.angleRot.to_numpy())
    tilt = np.deg2rad(data.angleTilt.to_numpy())
    psi = np.deg2rad(data.anglePsi.to_numpy())

    directions = spherical_to_cartesian(rot, tilt)
    del rot, tilt

    references_spherical = sample_projection_directions(args.n_groups)
    references = spherical_to_cartesian(
        references_spherical[:, 0], references_spherical[:, 1]
    )
    del references_spherical

    group_indices = group_projection_directions(
        directions=directions,
        references=references,
        consider_mirrors=True,
        batch_size=args.grouping_batch_size,
    )
    del directions, references

    data[args.out_group_column] = group_indices

    starfile.write(data=data, filename=args.out_star)


if __name__ == "__main__":
    main()
