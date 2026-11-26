#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import starfile
from xmippPyModules.coneGroupingTools.cone_grouping import (
    Assignments,
    EulerAngles,
    align_assignments,
    deduplicate_with_symmetry,
    euler_zyz_to_matrix,
    generate_reference_orientations,
    group_projection_directions,
    min_distance_between_directions,
)
from xmippPyModules.coneGroupingTools.symmetry import (
    SUPPORTED_SYMMETRIES,
    generate_symmetries,
)

MDL_ANGLE_PSI = "anglePsi"
MDL_FLIP = "flip"
SHIFT_X_COL = "shiftX"
SHIFT_Y_COL = "shiftY"


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
        help=f"Symmetry group of the volume. Supported groups: {SUPPORTED_SYMMETRIES}.",
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

    try:
        symmetries = generate_symmetries(sym=args.symmetry_group)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

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
