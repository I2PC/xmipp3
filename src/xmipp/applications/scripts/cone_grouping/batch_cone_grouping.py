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


def generate_reference_orientations(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate approximately uniform reference orientations over a hemisphere.

    Reference viewing directions are sampled with
    :func:`sample_projection_directions`. The sampled azimuthal and polar
    angles are interpreted respectively as Xmipp ``rot`` and ``tilt`` angles,
    while ``psi`` is set to zero. Thus, each reference has a fixed in-plane
    orientation that acts as the target orientation when particles assigned
    to that reference are subsequently aligned.

    All angles used internally are in radians.

    Parameters
    ----------
    n : int
        Number of reference orientations to generate.

    Returns
    -------
    matrices : np.ndarray
        Reference orientation matrices with shape ``(n, 3, 3)``, following
        the Xmipp orientation convention used by
        :func:`euler_zyz_to_matrix`.
    directions : np.ndarray
        Corresponding unit viewing directions with shape ``(n, 3)``. Each
        direction is the third row of its orientation matrix.
    """
    spherical = sample_projection_directions(n)

    rot = spherical[:, 0]
    tilt = spherical[:, 1]
    psi = np.asarray(0.0)

    matrices = euler_zyz_to_matrix(rot, tilt, psi)
    directions = matrices[..., 2, :]

    return matrices, directions


def _nearest_orthogonal(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    return u @ vh


def compute_in_plane_alignment_matrix(
    reference_matrix_3d: np.ndarray,
    rotation_matrices_3d: np.ndarray,
) -> np.ndarray:
    """
    Compute the in-plane transformations aligning orientations to references.

    Orientation matrices follow the convention that they transform global
    coordinates into the coordinate system of the corresponding projection.
    Therefore, for a particle orientation ``E`` and reference orientation
    ``E_ref``, the relative coordinate transformation is

    ``E_ref @ E.T``.

    If both orientations have exactly the same viewing direction, the upper
    left 2x2 block of this matrix is the corresponding in-plane orthogonal
    transformation. For particles grouped only approximately with a reference,
    that block need not be exactly orthogonal, so it is projected onto the
    nearest orthogonal matrix.

    The resulting 2D transformation may have determinant -1 when the particle
    and reference viewing directions are antipodal.

    Parameters
    ----------
    reference_matrix_3d : np.ndarray
        Reference orientation matrix or batch of matrices with shape
        ``(..., 3, 3)``.
    rotation_matrices_3d : np.ndarray
        Particle orientation matrix or broadcast-compatible batch of matrices
        with shape ``(..., 3, 3)``.

    Returns
    -------
    np.ndarray
        In-plane orthogonal alignment matrices with shape ``(..., 2, 2)``.
    """
    delta_matrices_3d = reference_matrix_3d @ rotation_matrices_3d.swapaxes(-2, -1)

    return _nearest_orthogonal(delta_matrices_3d[..., :2, :2])


def align_to_references(
    particle_matrices: np.ndarray,
    reference_matrices: np.ndarray,
    group_indices: np.ndarray,
    symmetries: np.ndarray,
    symmetry_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the in-plane alignment of particles to their assigned references.

    For each particle ``i``, ``group_indices[i]`` identifies the reference
    orientation to which it was assigned, while ``symmetry_indices[i]``
    identifies the symmetry operation selected during grouping.

    Symmetry operations act on Xmipp orientation matrices by right
    multiplication,

    ``E_equiv = E @ S``,

    so the corresponding viewing direction transforms as ``d_equiv = d @ S``.
    The symmetry-equivalent particle orientation is first constructed and then
    aligned in-plane to the selected reference.

    A symmetry array containing a single element is assumed to represent the
    trivial C1 group and therefore to contain the identity matrix; in that
    case the matrix multiplication is skipped.

    All returned angles are in radians.

    Parameters
    ----------
    particle_matrices : np.ndarray
        Particle orientation matrices with shape ``(n_particles, 3, 3)``.
    reference_matrices : np.ndarray
        Reference orientation matrices with shape ``(n_references, 3, 3)``.
    group_indices : np.ndarray
        Integer array of shape ``(n_particles,)`` containing the selected
        reference index for each particle.
    symmetries : np.ndarray
        Symmetry rotation matrices with shape ``(n_symmetries, 3, 3)``.
        The identity must be included.
    symmetry_indices : np.ndarray
        Integer array of shape ``(n_particles,)`` containing the symmetry
        operation selected for each particle during grouping.

    Returns
    -------
    psi : np.ndarray
        Xmipp in-plane alignment angles, in radians, with shape
        ``(n_particles,)``.
    flip : np.ndarray
        Boolean array of shape ``(n_particles,)`` indicating whether the
        corresponding in-plane transformation includes a reflection.
    """
    if len(symmetries) > 1:
        particle_matrices = particle_matrices @ symmetries[symmetry_indices]

    alignment_2d = compute_in_plane_alignment_matrix(
        reference_matrix_3d=reference_matrices[group_indices],
        rotation_matrices_3d=particle_matrices,
    )

    return matrix_to_xmipp_psi_radians_flip(alignment_2d)


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


def group_projection_directions(
    directions: np.ndarray,
    references: np.ndarray,
    symmetries: np.ndarray,
    consider_mirrors: bool = True,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each direction to its closest reference direction, taking underlying
    group symmetries into account.

    Because the viewing directions ``d`` and ``d @ s`` are equivalent whenever
    ``s`` is a symmetry of the volume, the distances between each direction ``d``
    and reference ``r`` are calculated as the minimum of distances between
    ``d @ s1`` and ``r @ s2``, where ``s1, s2`` are symmetries.

    Closeness is measured by angular distance, equivalently by maximizing
    the dot product between unit vectors. If ``consider_mirrors`` is True,
    antipodal directions are treated as equivalent.

    Parameters
    ----------
    directions : np.ndarray
        Unit direction vectors with shape (n_directions, 3).
    references : np.ndarray
        Unit reference vectors with shape (n_references, 3).
    symmetries : np.ndarray
        Symmetry matrices for the underlying volume, shape (n_symmetries, 3, 3).
        This must include the identity matrix as one of its elements.
        For asymmetric volumes, this can be the trivial symmetry group 'C1',
        which can be generated with ``trivial_symmetry_group()``.
    consider_mirrors : bool, optional
        If True, directions v and -v are considered equivalent. Default is True.
    batch_size : int, optional
        Number of directions processed at once. Default is 1024.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_directions,) containing the index of
        the closest reference for each direction.
    np.ndarray
        Integer array of shape (n_directions,) containing the index of the
        chosen symmetry operation for each direction.
    """
    # Compute r[i] @ s[j].T for each reference and symmetry, because comparing r @ s.T
    # with a direction d is equivalent to comparing d @ s with r
    #
    # references:     (n_references, 3)
    # symmetries:     (n_symmetries, 3, 3)
    # references_sym: (n_references, n_symmetries, 3)
    references_sym = np.tensordot(references, symmetries, axes=([1], [2]))

    n_directions = len(directions)
    n_references = len(references)
    n_symmetries = len(symmetries)

    group_indices = np.empty(n_directions, dtype=np.min_scalar_type(n_references - 1))
    symmetry_indices = np.empty(
        n_directions, dtype=np.min_scalar_type(n_symmetries - 1)
    )

    start = 0
    while start < n_directions:
        end = min(n_directions, start + batch_size)
        direction_batch = directions[start:end]

        # direction_batch: (batch, 3)
        # references_sym:  (n_references, n_symmetries, 3)
        # cos:             (batch, n_references, n_symmetries)
        cos = np.tensordot(direction_batch, references_sym, axes=([1], [2]))

        if consider_mirrors:
            np.abs(cos, out=cos)

        # Flatten the reference and symmetry dimensions for argmax
        cos_flat = cos.reshape(end - start, -1)  # (batch, n_references * n_symmetries)
        flat_indices = np.argmax(cos_flat, axis=1)  # (batch, )
        j, k = np.unravel_index(flat_indices, shape=(n_references, n_symmetries))
        # j: (batch, ), indexing n_references
        # k: (batch, ), indexing n_symmetries

        group_indices[start:end] = j
        symmetry_indices[start:end] = k

        start = end

    return group_indices, symmetry_indices


def direct_rotation_around_z(angle_rad: float) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=np.float64)

    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    matrix[0, 0] = cos
    matrix[1, 0] = sin

    matrix[0, 1] = -sin
    matrix[1, 1] = cos

    matrix[2, 2] = 1

    return matrix


def direct_rotation_around_x(angle_rad: float) -> np.ndarray:
    matrix = np.zeros((3, 3))

    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)

    matrix[1, 1] = cos
    matrix[2, 1] = sin

    matrix[1, 2] = -sin
    matrix[2, 2] = cos

    matrix[0, 0] = 1

    return matrix


def generate_cn_matrices(n: int) -> np.ndarray:
    generator_angle = 2 * np.pi / n

    return np.array([direct_rotation_around_z(i * generator_angle) for i in range(n)])


def generate_dn_matrices(n: int) -> np.ndarray:
    cn = generate_cn_matrices(n)  # shape (n, 3, 3)

    rotx = direct_rotation_around_x(np.pi)  # shape (3, 3)
    twofold_rotations = rotx @ cn

    return np.concatenate([cn, twofold_rotations])


def trivial_symmetry_group() -> np.ndarray:
    return np.eye(3)[None, ...]


def parse_symmetry_group_string(sym: str) -> tuple[str, int]:
    sym_type = sym[0]
    order = int(sym[1:])

    return sym_type.lower(), order


def generate_symmetries(sym: str | None) -> np.ndarray:
    if sym is None:
        return trivial_symmetry_group()

    sym_type, order = parse_symmetry_group_string(sym)

    if order <= 0:
        raise ValueError("Group order must be a positive integer")

    if sym_type == "c":
        if order == 1:
            return trivial_symmetry_group()
        return generate_cn_matrices(order)

    if sym_type == "d":
        if order == 1:
            raise ValueError("The group D1 is not supported")

        return generate_dn_matrices(order)

    raise ValueError(f"Unsupported symmetry group symbol: {sym}")


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
    parser.add_argument(
        "--symmetry-group",
        type=str,
        default="c1",
        help="Symbol for the symmetry group of the volume. Currently only cn and dn are supported",
    )

    return parser


def main():
    args = build_argument_parser().parse_args()

    # Generate symmetries for the object
    symmetries = generate_symmetries(sym=args.symmetry_group)

    # Build the references
    reference_matrices, reference_directions = generate_reference_orientations(
        n=args.n_groups
    )
    data = pd.DataFrame(starfile.read(args.input_xmd))

    # Xmipp stores Euler angles in degrees
    rot = np.deg2rad(data.angleRot.to_numpy())
    tilt = np.deg2rad(data.angleTilt.to_numpy())
    psi = np.deg2rad(data.anglePsi.to_numpy())

    particle_matrices = euler_zyz_to_matrix(rot, tilt, psi)
    particle_directions = particle_matrices[..., 2, :]
    del rot, tilt, psi

    # Grouping
    group_indices, symmetry_indices = group_projection_directions(
        directions=particle_directions,
        references=reference_directions,
        symmetries=symmetries,
        consider_mirrors=True,
        batch_size=args.grouping_batch_size,
    )
    del particle_directions, reference_directions

    # Alignment
    alignment_psi, flip = align_to_references(
        particle_matrices=particle_matrices,
        reference_matrices=reference_matrices,
        group_indices=group_indices,
        symmetries=symmetries,
        symmetry_indices=symmetry_indices,
    )
    del symmetries
    del particle_matrices, reference_matrices

    data[args.out_group_column] = (
        group_indices.astype(np.int64) + 1
    )  # 1-based indices are preferred for class ids
    data[MDL_ANGLE_PSI] = np.rad2deg(alignment_psi)
    data[MDL_FLIP] = flip.astype(np.int8)

    starfile.write(data=data, filename=args.out_star)


if __name__ == "__main__":
    main()
