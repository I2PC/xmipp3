#!/usr/bin/env python3

from __future__ import annotations

import math
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import starfile
import pandas as pd

MDL_ANGLE_PSI = "anglePsi"
MDL_FLIP = "flip"
SHIFT_X_COL = "shiftX"
SHIFT_Y_COL = "shiftY"


@dataclass
class EulerAngles:
    """
    Euler angles in radians, stored in equally shaped arrays.

    Attributes
    ----------
    rot : np.ndarray
        Rotation about the initial Z axis.
    tilt : np.ndarray
        Rotation about the intermediate Y axis.
    psi : np.ndarray
        In-plane rotation about the final Z axis.
    """

    rot: np.ndarray
    tilt: np.ndarray
    psi: np.ndarray

    def __getitem__(self, index) -> EulerAngles:
        """Apply the same selection to all three angle arrays."""
        return EulerAngles(
            rot=self.rot[index],
            tilt=self.tilt[index],
            psi=self.psi[index],
        )


@dataclass
class Assignments:
    """
    Particle-to-reference assignments and their matching information.

    All attributes have shape ``(m,)``, where ``m`` is the number of
    assignments. Entry ``a`` describes one particle-reference pair.
    Each pair appears at most once, using its best symmetry match.

    Nearest-reference grouping produces one assignment per particle.
    Threshold grouping can produce several; with complete coverage,
    ``m >= n_particles``.

    Attributes
    ----------
    particle_indices : np.ndarray
        Zero-based positional indices into the input particle arrays.
        These are not metadata identifiers such as ``itemId``.
    group_indices : np.ndarray
        Zero-based indices into the reference arrays.
    symmetry_indices : np.ndarray
        Zero-based indices into the symmetry array. For each assignment,
        the selected operation acts on the particle as ``E @ S``.
    angular_distances : np.ndarray
        Minimum angular distances in radians, accounting for symmetry
        and, when enabled, antipodal equivalence.
    fallback : np.ndarray
        Boolean flags identifying assignments made to the nearest
        reference because none satisfied the threshold. Always false
        in nearest-reference mode.
    """

    particle_indices: np.ndarray
    group_indices: np.ndarray
    symmetry_indices: np.ndarray
    angular_distances: np.ndarray
    fallback: np.ndarray


# The functions ``euler_zyz_to_matrix``, ``sample_projection_directions``,
# and ``group_projection_directions`` were originally written
# by Oier Lauzirika Zarrabeitia in https://github.com/oierlauzi/factorem
# They have been vendored for this script
# and modified by Andrés Contreras to add docstrings, and adapt
# ``group_projection_directions`` to closest-reference grouping.


def euler_zyz_to_matrix(
    angles: EulerAngles, out: np.ndarray | None = None
) -> np.ndarray:
    """
    Convert ZYZ Euler angles to rotation matrices. Input angles are given in
    radians and may have any broadcast-compatible shapes.

    The returned matrix ``E`` follows the Xmipp orientation convention used
    throughout this module: it maps coordinates from the global reference frame
    to the particle/projection reference frame. The viewing direction is

    ``E[..., 2, :] = (sin(tilt) cos(rot),
                    sin(tilt) sin(rot),
                    cos(tilt))``.

    Equivalent orientations under a volume symmetry ``S`` are represented by
    right multiplication, ``E @ S``.

    Parameters
    ----------
    angles : EulerAngles
        Euler angles in radians.
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
    rot = angles.rot
    tilt = angles.tilt
    psi = angles.psi

    # Create the output
    batch_shape = np.broadcast_shapes(rot.shape, tilt.shape, psi.shape)
    result_shape = batch_shape + (3, 3)
    dtype = angles.rot.dtype

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


def generate_reference_orientations(
    n: int,
) -> tuple[EulerAngles, np.ndarray]:
    """Generate reference orientations over the hemisphere with z >= 0.

    Viewing directions follow the Fibonacci sampling used by
    ``sample_projection_directions``. Each reference has zero psi,
    fixing the target in-plane orientation for subsequent alignment.

    Parameters
    ----------
    n : int
        Positive number of reference orientations.

    Returns
    -------
    angles : EulerAngles
        Reference Euler angles in radians, each with shape ``(n,)``.
    matrices : np.ndarray
        Xmipp orientation matrices with shape ``(n, 3, 3)``.
        Viewing directions are available as ``matrices[:, 2, :]``.
    """
    if n <= 0:
        raise ValueError("The number of references must be positive")

    spherical = sample_projection_directions(n)
    angles = EulerAngles(
        rot=spherical[:, 0],
        tilt=spherical[:, 1],
        psi=np.zeros(n),
    )
    return angles, euler_zyz_to_matrix(angles)


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
    particle_matrices_3d: np.ndarray,
    particle_shifts_2d: np.ndarray,
    reference_matrices_3d: np.ndarray,
    group_indices: np.ndarray,
    symmetries: np.ndarray,
    symmetry_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    particle_matrices_3d : np.ndarray
        Particle orientation matrices with shape ``(n_particles, 3, 3)``.
    particle_shifts_2d : np.ndarray
        Shifts encoding the translation part of the alignment matrix, shape
        ``(n_particles, 2)`` where ``particle_shifts_2d[:, 0]`` corresponds
        to shifts in X and ``particle_shifts_2d[:, 1]`` corresponds to shifts
        in Y.
    reference_matrices_3d : np.ndarray
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
    aligned_shifts : np.ndarray
        Translation component of the new 2D affine alignment, with shape
        ``(n_particles, 2)``. If ``Q`` is the new in-plane alignment and
        ``s`` is the original centering shift, the returned shift is
        ``Q @ s``, so that the final transform is ``Q @ (x + s)``.
    """
    if len(symmetries) > 1:
        particle_matrices_3d = particle_matrices_3d @ symmetries[symmetry_indices]

    alignment_2d = compute_in_plane_alignment_matrix(
        reference_matrix_3d=reference_matrices_3d[group_indices],
        rotation_matrices_3d=particle_matrices_3d,
    )

    aligned_shifts = (alignment_2d @ particle_shifts_2d[..., None])[..., 0]
    psi, flip = matrix_to_xmipp_psi_radians_flip(alignment_2d)

    return psi, flip, aligned_shifts


def align_assignments(
    particle_matrices_3d: np.ndarray,
    particle_shifts_2d: np.ndarray,
    reference_matrices_3d: np.ndarray,
    assignments: Assignments,
    symmetries: np.ndarray,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align each assignment from its original particle, in bounded batches.

    Returns psi in radians, boolean flips, and transformed shifts, all in
    assignment order. Repeated particles are aligned independently to each
    selected reference; an existing assignment's alignment is never reused.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    n_assignments = len(assignments.particle_indices)
    psi = np.empty(n_assignments, dtype=float)
    flip = np.empty(n_assignments, dtype=bool)
    shifts = np.empty((n_assignments, 2), dtype=float)

    for start in range(0, n_assignments, batch_size):
        end = min(start + batch_size, n_assignments)
        particles = assignments.particle_indices[start:end]
        psi[start:end], flip[start:end], shifts[start:end] = align_to_references(
            particle_matrices_3d=particle_matrices_3d[particles],
            particle_shifts_2d=particle_shifts_2d[particles],
            reference_matrices_3d=reference_matrices_3d,
            group_indices=assignments.group_indices[start:end],
            symmetries=symmetries,
            symmetry_indices=assignments.symmetry_indices[start:end],
        )

    return psi, flip, shifts


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


def _symmetry_adjusted_references(
    references: np.ndarray,
    symmetries: np.ndarray,
) -> np.ndarray:
    """Return ``r @ S.T`` with shape ``(n_references, n_symmetries, 3)``.

    Comparing a direction ``d`` with ``r @ S.T`` is equivalent to
    comparing the symmetry-equivalent direction ``d @ S`` with ``r``.
    """
    return np.tensordot(references, symmetries, axes=([1], [2]))


def _best_reference_matches(
    directions: np.ndarray,
    references_sym: np.ndarray,
    consider_mirrors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the best symmetry match for each direction-reference pair.

    Parameters
    ----------
    directions : np.ndarray
        Unit directions for one batch, shape ``(batch, 3)``.
    references_sym : np.ndarray
        Symmetry-adjusted references, shape ``(n_references, n_symmetries, 3)``.
    consider_mirrors : bool
        Whether antipodal directions are treated as equivalent.

    Returns
    -------
    best_cosines : np.ndarray
        Maximum matching cosines, shape ``(batch, n_references)``.
    symmetry_indices : np.ndarray
        Selected symmetry indices with the same shape. Exact ties
        select the first symmetry in the supplied order.
    """
    # cosines is shape (batch, n_references, n_symmetries)
    cosines = np.tensordot(directions, references_sym, axes=([1], [2]))

    if not np.isfinite(cosines).all():
        raise ValueError("Matching cosines must be finite")

    if consider_mirrors:
        np.abs(cosines, out=cosines)

    symmetry_indices = np.argmax(cosines, axis=2)
    best_cosines = np.take_along_axis(cosines, symmetry_indices[..., None], axis=2)[
        ..., 0
    ]

    return best_cosines, symmetry_indices


def group_projection_directions(
    directions: np.ndarray,
    references: np.ndarray,
    symmetries: np.ndarray,
    consider_mirrors: bool = True,
    batch_size: int = 1024,
    *,
    mode: str = "nearest",
    angular_radius: float | None = None,
    unassigned_policy: str = "error",
) -> Assignments:
    """
    Assign directions to references under volume symmetry.

    Parameters
    ----------
    directions : np.ndarray
        Unit particle directions, shape ``(n_particles, 3)``.
    references : np.ndarray
        Unit reference directions, shape ``(n_references, 3)``.
    symmetries : np.ndarray
        Volume symmetry matrices, shape ``(n_symmetries, 3, 3)``.
        The identity must be included.
    consider_mirrors : bool, optional
        Treat antipodal directions as equivalent.
    batch_size : int, optional
        Positive number of particles processed per batch.
    mode : {"nearest", "threshold"}, optional
        Select the nearest reference or all references within the radius.
    angular_radius : float or None, optional
        Inclusive angular radius in radians, required only in threshold
        mode. Range: [0, pi/2] with antipodal equivalence, otherwise [0, pi].
    unassigned_policy : {"error", "nearest"}, optional
        In threshold mode, fail if any particle is uncovered or assign
        uncovered particles to their nearest reference. Nondefault values
        are rejected in nearest mode.

    Returns
    -------
    Assignments
        Flat assignments ordered by particle and then reference index.
        Each pair uses its best symmetry; exact ties select the first
        symmetry. Nearest-reference ties select the first reference.
        Indices are zero-based and angular distances are in radians.

    Raises
    ------
    ValueError
        For invalid options, nonfinite matching cosines, or uncovered
        particles when ``unassigned_policy="error"``. Coverage failures
        report one-based input row numbers and nearest-reference distances.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(references) == 0 or len(symmetries) == 0:
        raise ValueError("At least one reference and one symmetry are required")
    if len(directions) == 0:
        raise ValueError("At least one particle is required")
    if unassigned_policy not in ("error", "nearest"):
        raise ValueError("unassigned_policy must be 'error' or 'nearest'")
    if mode == "nearest":
        if angular_radius is not None or unassigned_policy != "error":
            raise ValueError("Radius and fallback options require threshold mode")
    else:
        max_radius = np.pi / 2 if consider_mirrors else np.pi
        if (
            angular_radius is None
            or not np.isfinite(angular_radius)
            or not 0 <= angular_radius <= max_radius
        ):
            raise ValueError(
                f"angular_radius must be finite and between 0 and {max_radius} radians"
            )

    if mode == "nearest":
        return _group_to_nearest(
            directions, references, symmetries, consider_mirrors, batch_size
        )

    if mode == "threshold":
        return _group_with_threshold(
            directions,
            references,
            symmetries,
            consider_mirrors,
            batch_size,
            angular_radius=angular_radius,
            unassigned_policy=unassigned_policy,
        )

    raise ValueError("mode must be 'nearest' or 'threshold'")


def _group_to_nearest(
    directions: np.ndarray,
    references: np.ndarray,
    symmetries: np.ndarray,
    consider_mirrors: bool = True,
    batch_size: int = 1024,
) -> Assignments:
    """
    Assign each direction to its closest reference direction, taking underlying
    group symmetries into account.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(references) == 0 or len(symmetries) == 0:
        raise ValueError("At least one reference and one symmetry are required")

    # shape (n_references, n_symmetries, 3)
    references_sym = _symmetry_adjusted_references(references, symmetries)

    n_particles = len(directions)

    group_indices = np.empty(n_particles, dtype=np.intp)
    symmetry_indices = np.empty(n_particles, dtype=np.intp)
    nearest_distances = np.empty(n_particles, dtype=float)

    for start in range(0, n_particles, batch_size):
        end = min(n_particles, start + batch_size)
        best_cosines, best_symmetries = _best_reference_matches(
            directions=directions[start:end],
            references_sym=references_sym,
            consider_mirrors=consider_mirrors,
        )
        rows = np.arange(end - start)
        selected_groups = np.argmax(best_cosines, axis=1)

        group_indices[start:end] = selected_groups
        symmetry_indices[start:end] = best_symmetries[rows, selected_groups]
        nearest_distances[start:end] = np.arccos(
            np.clip(best_cosines[rows, selected_groups], -1.0, 1.0)
        )

    return Assignments(
        particle_indices=np.arange(n_particles),
        group_indices=group_indices,
        symmetry_indices=symmetry_indices,
        angular_distances=nearest_distances,
        fallback=np.zeros(n_particles, dtype=bool),
    )


def _group_with_threshold(
    directions: np.ndarray,
    references: np.ndarray,
    symmetries: np.ndarray,
    consider_mirrors: bool = True,
    batch_size: int = 1024,
    *,
    angular_radius: float,
    unassigned_policy: str = "error",
) -> Assignments:
    """
    Group each direction with every reference that is closer than an angular
    radius (in radians), accounting for symmetry.
    """
    # shape (n_references, n_symmetries, 3)
    references_sym = _symmetry_adjusted_references(references, symmetries)

    n_particles = len(directions)
    chunks: list[Assignments] = []
    nearest_distances = np.empty(n_particles, dtype=float)
    uncovered = np.empty(n_particles, dtype=bool)

    for start in range(0, n_particles, batch_size):
        end = min(n_particles, start + batch_size)
        best_cosines, best_symmetries = _best_reference_matches(
            directions=directions[start:end],
            references_sym=references_sym,
            consider_mirrors=consider_mirrors,
        )
        rows = np.arange(end - start)
        angles = np.arccos(np.clip(best_cosines, -1.0, 1.0))

        # Save nearest distances for reporting later in case of uncovered particles
        nearest_groups = np.argmax(best_cosines, axis=1)
        nearest_distances[start:end] = angles[rows, nearest_groups]

        membership = angles <= angular_radius
        missing = ~membership.any(axis=1)
        uncovered[start:end] = missing
        if unassigned_policy == "nearest":
            membership[rows[missing], nearest_groups[missing]] = True

        particle_rows, groups = np.nonzero(membership)
        distances = angles[particle_rows, groups]
        fallback = missing[particle_rows]

        chunks.append(
            Assignments(
                particle_indices=particle_rows + start,
                group_indices=groups,
                symmetry_indices=best_symmetries[particle_rows, groups],
                angular_distances=distances,
                fallback=fallback,
            )
        )

    if uncovered.any() and unassigned_policy == "error":
        missing_indices = np.flatnonzero(uncovered)
        # Round upwards so copying the reported radius does not under-cover.
        covering_radius = np.ceil(np.rad2deg(nearest_distances.max()))

        raise ValueError(
            f"{len(missing_indices)}/{n_particles} particles "
            f"({100 * len(missing_indices) / n_particles:.2f}%) have no reference "
            f"within the specified radius of {np.rad2deg(angular_radius):.4f} deg. "
            f"A radius of at least {covering_radius} covers this dataset "
            "with these references."
        )

    return Assignments(
        particle_indices=np.concatenate([c.particle_indices for c in chunks]),
        group_indices=np.concatenate([c.group_indices for c in chunks]),
        symmetry_indices=np.concatenate([c.symmetry_indices for c in chunks]),
        angular_distances=np.concatenate([c.angular_distances for c in chunks]),
        fallback=np.concatenate([c.fallback for c in chunks]),
    )


def deduplicate_with_symmetry(
    directions: np.ndarray,
    symmetries: np.ndarray,
    angular_distance_threshold: float,
    batch_size: int,
    consider_mirrors: bool,
) -> np.ndarray:
    """
    Eliminate from the set of directions those that are closer than a certain
    distance threshold, taking symmetries into account.

    Directions are processed in their original order. If two references are
    equivalent, the first one is retained.

    Parameters
    ----------
    directions : np.ndarray
        Array of shape ``(n, 3)`` whose elements are vectors on the unit sphere.
    symmetries : np.ndarray
        Array of shape ``(m, 3, 3)`` whose elements are rotation matrices corresponding
        to the symmetries of the volume.
    angular_distance_threshold : float
        Threshold in radians under which two references are considered to be equivalent.
    batch_size : int
        Batch size for processing the references when computing pairwise distances.
        The full distance matrix has shape ``(m, n, n)`` when accounting for symmetries,
        so batching can be used to reduce the memory cost of this function.
    consider_mirrors : bool
        If True, antipodal (or sufficiently close to antipodal) viewing directions
        are considered equivalent.

    Returns
    -------
    np.ndarray
        Boolean mask of shape ``(n,)`` indicating which directions should be kept.
    """
    # Convert angular distance to a cosine value
    cosine_threshold = np.cos(angular_distance_threshold)
    n_references = len(directions)

    available = np.ones(n_references, dtype=bool)

    symmetric_references = _symmetry_adjusted_references(directions, symmetries)

    for i in range(n_references):
        # An eliminated reference must not eliminate other references
        if not available[i]:
            continue

        reference_sym = symmetric_references[i]  # (m, 3)

        # Only compare with later references to avoid self-comparison
        # and ensure first reference wins
        for start in range(i + 1, n_references, batch_size):
            end = min(start + batch_size, n_references)

            candidates = directions[start:end]

            # reference_sym: (m, 3)
            # candidates:    (batch, 3)
            # cosines:       (m, batch)
            cosines = reference_sym @ candidates.T

            if consider_mirrors:
                np.abs(cosines, out=cosines)

            # Minimum angular distance over all equivalent orientations corresponds to
            # maximum cosine similarity
            max_cosines = np.max(cosines, axis=0)

            eliminate = max_cosines > cosine_threshold

            # Bitwise AND to ensure eliminated references stay eliminated
            available[start:end] &= ~eliminate

    return available


def min_distance_between_directions(
    directions: np.ndarray, consider_mirrors: bool
) -> float:
    """
    Returns the minimum angular distance between each pair of directions having
    distinct indices on the ``directions`` array.

    Parameters
    ----------
    directions : np.ndarray
        Unit direction vectors with shape ``(n, 3)``.
    consider_mirrors : bool, optional
        If True, antipodal directions are considered equivalent, so the
        angular distance is computed from the absolute dot product.
        Default is False.

    Returns
    -------
    float
        Minimum angular distance, in radians, between directions at distinct
        positions in the input array. Returns ``np.inf`` for a single
        direction.
    """
    n_directions = len(directions)
    if n_directions == 1:
        return np.inf

    cosines = directions @ directions.T

    if consider_mirrors:
        np.abs(cosines, out=cosines)

    np.fill_diagonal(cosines, -np.inf)  # avoid distance from a direction to itself

    max_cosine = np.max(cosines)
    max_cosine = np.clip(max_cosine, -1.0, 1.0)

    return np.arccos(max_cosine)


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


def build_assignment_metadata(
    data: pd.DataFrame,
    assignments: Assignments,
    alignment_psi: np.ndarray,
    flip: np.ndarray,
    aligned_shifts: np.ndarray,
    group_column: str,
) -> pd.DataFrame:
    """
    Build one metadata row per assignment.

    Alignment arrays must follow the same order as ``assignments``.
    ``assignments.particle_indices`` selects positional rows from
    ``data``, whose order remains the original input particle order.
    Input rows are copied before modification.
    Output group identifiers are one-based.
    """
    diagnostic_columns = {"angular_distance_deg", "assignment_fallback"}
    collisions = diagnostic_columns.intersection(data.columns)
    if collisions or group_column in diagnostic_columns:
        raise ValueError(
            "Assignment output column names conflict with input metadata or "
            f"the group column: {sorted(collisions | ({group_column} & diagnostic_columns))}"
        )

    result = data.iloc[assignments.particle_indices].copy().reset_index(drop=True)
    result["angular_distance_deg"] = np.rad2deg(assignments.angular_distances)
    result["assignment_fallback"] = assignments.fallback.astype(np.int8)

    result[group_column] = assignments.group_indices.astype(np.int64) + 1
    result[MDL_ANGLE_PSI] = np.rad2deg(alignment_psi)
    result[MDL_FLIP] = flip.astype(np.int8)
    result[SHIFT_X_COL] = aligned_shifts[:, 0]
    result[SHIFT_Y_COL] = aligned_shifts[:, 1]

    return result


def build_reference_metadata(
    angles: EulerAngles,
    assignments: Assignments,
) -> pd.DataFrame:
    """Build reference metadata, including zero counts for empty groups.

    Output angles are in degrees and group identifiers are one-based.
    Counts describe memberships, so their sum equals the number of
    assignments rather than necessarily the number of input particles.
    """
    n_references = len(angles.rot)

    return pd.DataFrame(
        {
            "cone_group": np.arange(1, n_references + 1),
            "angleRot": np.rad2deg(angles.rot),
            "angleTilt": np.rad2deg(angles.tilt),
            "anglePsi": np.rad2deg(angles.psi),
            "n_particles": np.bincount(
                assignments.group_indices, minlength=n_references
            ),
        }
    )


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
        "--out-group-column",
        type=str,
        default="cone_group",
        help=(
            "Name for the column in the output metadata file that "
            "contains the index for each particle's group"
        ),
    )
    parser.add_argument(
        "--out-reference-md",
        type=Path,
        help=("Path to the output .star file with info about the generated references"),
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
        "--symmetry-group",
        type=str,
        default="c1",
        help="Symbol for the symmetry group of the volume. Currently only cn and dn are supported",
    )
    parser.add_argument(
        "--deduplicate-references",
        action="store_true",
        default=False,
        help=(
            "Eliminate redundant reference viewing directions, taking the object's "
            "symmetry into account. Two directions will be considered redundant when "
            "their angular distance when considering symmetries falls below half the "
            "minimum angular distance between all pairs of original references when "
            "not considering symmetries.\n"
            "Enabling this option means that the final number of groups may be lower "
            "than the specified number of groups (selected through --n-groups)."
        ),
    )

    parser.add_argument(
        "--grouping-mode",
        choices=("nearest", "threshold"),
        default="nearest",
        help="Assign to the nearest reference or to all references within a radius",
    )
    parser.add_argument(
        "--angular-radius-deg",
        type=float,
        help="Inclusive angular radius in degrees, required in threshold mode (0 to 90)",
    )
    parser.add_argument(
        "--unassigned-policy",
        choices=("error", "nearest"),
        default="error",
        help="Threshold mode: fail on uncovered particles or use a marked nearest fallback",
    )

    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if args.n_groups <= 0:
        parser.error("--n-groups must be positive")
    if args.grouping_batch_size <= 0:
        parser.error("--grouping-batch-size must be positive")

    if args.grouping_mode == "nearest" and (
        args.angular_radius_deg is not None or args.unassigned_policy != "error"
    ):
        parser.error("--angular-radius-deg and nearest fallback require threshold mode")
    if args.grouping_mode == "threshold":
        radius = args.angular_radius_deg
        if radius is None:
            parser.error("--angular-radius-deg is required in threshold mode")
        if not math.isfinite(radius) or not 0 <= radius <= 90:
            parser.error("--angular-radius-deg must be finite and between 0 and 90")

    symmetries = generate_symmetries(sym=args.symmetry_group)

    reference_angles, reference_matrices = generate_reference_orientations(
        n=args.n_groups
    )
    reference_directions = reference_matrices[:, 2, :]

    if args.deduplicate_references:
        threshold = 0.5 * min_distance_between_directions(
            reference_directions, consider_mirrors=True
        )
        keep_mask = deduplicate_with_symmetry(
            directions=reference_directions,
            symmetries=symmetries,
            angular_distance_threshold=threshold,
            batch_size=args.grouping_batch_size,
            consider_mirrors=True,
        )
        reference_angles = reference_angles[keep_mask]
        reference_matrices = reference_matrices[keep_mask]
        reference_directions = reference_matrices[:, 2, :]

    # Read particle alignment angles - Xmipp stores Euler angles in degrees
    data = pd.DataFrame(starfile.read(args.input_xmd))
    particle_angles = EulerAngles(
        rot=np.deg2rad(data["angleRot"].to_numpy()),
        tilt=np.deg2rad(data["angleTilt"].to_numpy()),
        psi=np.deg2rad(data["anglePsi"].to_numpy()),
    )
    shifts = data[[SHIFT_X_COL, SHIFT_Y_COL]].to_numpy(dtype=float)
    particle_matrices = euler_zyz_to_matrix(particle_angles)
    particle_directions = particle_matrices[..., 2, :]

    assignments = group_projection_directions(
        directions=particle_directions,
        references=reference_directions,
        symmetries=symmetries,
        consider_mirrors=True,
        batch_size=args.grouping_batch_size,
        mode=args.grouping_mode,
        angular_radius=(
            None
            if args.angular_radius_deg is None
            else np.deg2rad(args.angular_radius_deg)
        ),
        unassigned_policy=args.unassigned_policy,
    )
    del particle_directions, reference_directions

    alignment_psi, flip, aligned_shifts = align_assignments(
        particle_matrices_3d=particle_matrices,
        particle_shifts_2d=shifts,
        reference_matrices_3d=reference_matrices,
        assignments=assignments,
        symmetries=symmetries,
        batch_size=args.grouping_batch_size,
    )
    del symmetries
    del particle_matrices, reference_matrices

    output_data = build_assignment_metadata(
        data=data,
        assignments=assignments,
        alignment_psi=alignment_psi,
        flip=flip,
        aligned_shifts=aligned_shifts,
        group_column=args.out_group_column,
    )
    starfile.write(data=output_data, filename=args.out_star)

    fallback_count = int(assignments.fallback.sum())
    if fallback_count:
        print(
            f"WARNING: nearest-reference fallback used for {fallback_count}/{len(data)} "
            "particles; see assignment_fallback in the output.",
            file=sys.stderr,
        )

    if args.out_reference_md:
        reference_md = build_reference_metadata(
            angles=reference_angles,
            assignments=assignments,
        )
        starfile.write(data=reference_md, filename=args.out_reference_md)


if __name__ == "__main__":
    main()
