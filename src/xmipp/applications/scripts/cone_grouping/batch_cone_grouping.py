#!/usr/bin/env python3

from __future__ import annotations

import math
import argparse
from pathlib import Path

import numpy as np
import starfile
import pandas as pd

MDL_ANGLE_PSI = "anglePsi"
MDL_FLIP = "flip"


# The functions ``euler_zyz_to_matrix``, ``sample_projection_directions``,
# and ``group_projection_directions`` were originally written
# by Oier Lauzirika Zarrabeitia in https://github.com/oierlauzi/factorem
# They have been vendored for this script
# and modified by Andrés Contreras to add docstrings, and adapt
# ``group_projection_directions`` to closest-reference grouping.


def euler_zyz_to_matrix(
    rot: np.ndarray, tilt: np.ndarray, psi: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert ZYZ Euler angles to rotation matrices. Input angles are given in
    radians and may have any broadcast-compatible shapes.

    Parameters
    ----------
    rot : np.ndarray
        First rotation angle about the z axis, in radians.
    tilt : np.ndarray
        Rotation angle about the y axis, in radians.
    psi : np.ndarray
        Final rotation angle about the z axis, in radians.
    out : np.ndarray, optional
        Array in which to store the resulting matrices. It must have
        shape ``broadcast_shape + (3, 3)`` and the same dtype as ``rot``.

    Returns
    -------
    np.ndarray
        Rotation matrices with shape ``broadcast_shape + (3, 3)``.

    Notes
    -----
    With the convention used by this implementation, the projection direction is
    given by the third row of the rotation matrix. This can be extracted by
    doing ``euler_zyz_to_matrix(...)[..., 2, :]``.
    """
    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = rot.dtype

    if out is None:
        out = np.empty(result_shape, dtype=dtype)
    elif out.shape != result_shape or out.dtype != dtype:
        raise RuntimeError("Invalid output array was provided")

    ai = rot
    aj = tilt
    ak = psi

    # Obtain sin and cos of the angles
    ci = np.cos(ai)
    si = np.sin(ai)
    cj = np.cos(aj)
    sj = np.sin(aj)
    ck = np.cos(ak)
    sk = np.sin(ak)

    # Obtain the combinations
    cc = cj * ci
    cs = cj * si
    sc = sj * ci
    ss = sj * si

    # Build the matrix
    out[..., 0, 0] = ck * cc - sk * si
    out[..., 0, 1] = ck * cs + sk * ci
    out[..., 0, 2] = -ck * sj
    out[..., 1, 0] = -sk * cc - ck * si
    out[..., 1, 1] = -sk * cs + ck * ci
    out[..., 1, 2] = sk * sj
    out[..., 2, 0] = sc
    out[..., 2, 1] = ss
    out[..., 2, 2] = cj

    return out


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


def _nearest_orthogonal(matrix: np.ndarray):
    u, _, vh = np.linalg.svd(matrix)
    return u @ vh


def compute_in_plane_alignment_matrix(
    reference_matrix_3d: np.ndarray, rotation_matrices_3d: np.ndarray
) -> np.ndarray:
    delta_matrices_3d = reference_matrix_3d @ rotation_matrices_3d.swapaxes(-2, -1)

    return _nearest_orthogonal(delta_matrices_3d[..., :2, :2])


def matrix_to_xmipp_psi_radians_flip(
    matrix_batch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a batch of 2D orthogonal matrices to XMIPP parameters.

    Parameters
    ----------
    matrix_batch : np.ndarray
        Batch of 2D orthogonal transformation matrices with shape (..., 2, 2).

    Returns
    -------
    psi : np.ndarray
        In-plane rotation angles in radians with shape (...).
    flip : np.ndarray
        Boolean array indicating whether a reflection/flip was applied with shape (...).
    """
    flip = np.linalg.det(matrix_batch) < 0.0
    sign = np.where(flip, -1.0, 1.0)

    cosines = sign * matrix_batch[..., 0, 0]
    sines = sign * matrix_batch[..., 0, 1]

    psi = np.arctan2(sines, cosines)

    return psi, flip


def group_in_cones_and_align(
    rot: np.ndarray,
    tilt: np.ndarray,
    psi: np.ndarray,
    n_groups: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group particle orientations by projection direction and compute their
    in-plane alignments to the corresponding group references.

    The in-plane alignment includes a rotation angle ``psi`` and a ``flip`` boolean
    flag that indicates whether the particle should be flipped around the Y axis.
    This follows the convention for ``xmipp_transform_geometry``.

    Parameters
    ----------
    rot, tilt, psi : np.ndarray
        Euler angles for the particle alignment, following Xmipp's intrinsic
        ZYZ convention:
        $$ E_i = R_Z(- \\psi) R_Y(\\text{tilt}) R_Z(- \\text{rot}), $$
        where $E_i$ is the matrix that transform from global coordinates to the
        reference system of particle $i$: $x_i = E_i x$.
        These are expected to be arrays of shape ``(n,)``.
    n_groups : int
        Number of groups to group the particles in.
    batch_size : int
        Batch size used to group the particles together.

    Returns
    -------
    group_indices : np.ndarray
        Array of shape ``(n,)`` containing, for each particle, the index of the
        referenced it was grouped with.
    psi_alignment_deg : np.ndarray
        Array of shape ``(n,)`` containing, for each particle, the in-plane rotation
        angle needed to align it with its group's reference; in degrees. Follows
        Xmipp's conventions for ``xmipp_transform_geometry``.
    flip : np.ndarray
        Array of shape ``(n,)`` containing, for each particle, a boolean flag that
        indicates whether the particle needs to be flipped (around the Y axis) to
        align it with its group's reference.
    """
    references_spherical = sample_projection_directions(n_groups)

    rot_ref = references_spherical[:, 0]
    tilt_ref = references_spherical[:, 1]
    psi_ref = np.asarray(0.0)

    # Extract 3D Euler matrices for references and given angles (necessary for alignment)
    reference_matrix = euler_zyz_to_matrix(rot=rot_ref, tilt=tilt_ref, psi=psi_ref)
    particle_matrix = euler_zyz_to_matrix(rot=rot, tilt=tilt, psi=psi)

    # Viewing direction is the third row of the Euler matrix (necessary for grouping)
    reference_directions = reference_matrix[..., 2, :]
    particle_directions = particle_matrix[..., 2, :]

    group_indices = group_projection_directions(
        directions=particle_directions,
        references=reference_directions,
        consider_mirrors=True,  # only sampled the z>=0 hemisphere for references, so must use consider_mirrors=True
        batch_size=batch_size,
    )

    alignment_2d = compute_in_plane_alignment_matrix(
        reference_matrix_3d=reference_matrix[group_indices],
        rotation_matrices_3d=particle_matrix,
    )

    psi_alignment, flip = matrix_to_xmipp_psi_radians_flip(alignment_2d)

    # xmipp expects angles in degrees, not radians
    psi_alignment_deg = np.rad2deg(psi_alignment)

    return group_indices, psi_alignment_deg, flip


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

    group_indices, alignment_psi_deg, flip = group_in_cones_and_align(
        rot, tilt, psi, n_groups=args.n_groups, batch_size=args.grouping_batch_size
    )
    del rot, tilt, psi

    data[args.out_group_column] = group_indices.astype(np.int64)
    data[MDL_ANGLE_PSI] = alignment_psi_deg
    data[MDL_FLIP] = flip.astype(np.int8)

    starfile.write(data=data, filename=args.out_star)


if __name__ == "__main__":
    main()
