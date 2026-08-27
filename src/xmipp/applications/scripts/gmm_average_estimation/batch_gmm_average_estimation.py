#!/usr/bin/env python3

import argparse
from pathlib import Path
import warnings
from typing import Tuple, Union, Optional

import mrcfile
import numpy as np
import starfile
import torch
import pandas as pd

from xmippPyModules.gmmAverageTools.data import (
    read_images,
    MDL_REF_COLUMN,
    MDL_ITEM_ID_COLUMN,
)
from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator
from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
from xmippPyModules.gmmAverageTools.weights import (
    calculate_beta_auto,
    tagare_weight_precomputed,
)
from xmippPyModules.gmmAverageTools.distances import tagare_distance_precomputed
from xmippPyModules.gmmAverageTools.masks import create_circular_mask
from xmippPyModules.gmmAverageTools.utils import weighted_average

ESTIMATOR_WEIGHT_COLUMNS = {
    "gmm": ["wRobust", "wRobustGmm"],
    "irls": ["wRobust"],
}

# Estimator parameters
# NOTE: to be changed for configurable arguments in the future
EXTERNAL_GMM_MAX_ITER = 15
INTERNAL_GMM_MAX_ITER = 20
GMM_STANDARDIZE_DISTANCES = True
ESTIMATOR_RANDOM_STATE = 42
ESTIMATOR_TOL = 1.0e-4
IRLS_MAX_ITER = 50
IRLS_DAMPING_COEF = 0.0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-xmd",
        required=True,
        type=Path,
        help="Path to the .xmd file containing the path to the image stack",
    )
    parser.add_argument(
        "--base-xmd",
        type=Path,
        help=(
            "Path to the base .xmd file the weights should be added to. "
            "The original file will not be modified, but a new one will be "
            "created with the same information plus the weights. "
            "If not specified, the input .xmd file will be used."
        ),
    )
    parser.add_argument(
        "--out-star",
        type=Path,
        help="Path to the location of the new .star file",
    )
    parser.add_argument(
        "--out-corrected-avg",
        type=Path,
        help="Path to the output .mrc file for the corrected class average",
    )
    parser.add_argument(
        "--out-original-avg",
        type=Path,
        help="Path to the output .mrc file for the original class average",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Compute device for PyTorch",
    )
    parser.add_argument(
        "--group-by-column",
        type=str,
        help=(
            f"Column by which the images will be grouped. The default "
            f"value is '{MDL_REF_COLUMN}', which correspond to the xmipp "
            f"column for class id"
        ),
        default=MDL_REF_COLUMN,
    )
    parser.add_argument(
        "--estimator-type",
        type=str,
        choices=["gmm", "irls"],
        help=(
            "Type of estimator to use for the robust estimation. "
            "'irls' corresponds to an Iteratively Reweighted Least Squares (IRLS) "
            "procedure, whilst 'gmm' corresponds to a GMM-reweighted version of IRLS"
        ),
    )

    return parser


def validate_item_ids(data: pd.DataFrame, name: str) -> None:
    """Validate that a metadata table contains unique particle item identifiers."""
    if MDL_ITEM_ID_COLUMN not in data.columns:
        raise ValueError(
            f"{name} metadata does not contain " f"'{MDL_ITEM_ID_COLUMN}'."
        )

    if data[MDL_ITEM_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"{name} metadata contains duplicated " f"'{MDL_ITEM_ID_COLUMN}' values."
        )


def initialize_estimator(
    unmasked_images: torch.Tensor, masked_images: torch.Tensor, estimator_type: str
):
    # Calculate the automatic scaling parameter for the distance or weight functions.
    auto_beta = calculate_beta_auto(imgs=unmasked_images, mult=1.0)

    # Calculate norms and flatten images for precomputed versions of
    # distance or weight functions
    masked_images_flat = masked_images.flatten(1)
    image_norm_sq = masked_images_flat.square().sum(dim=1)
    image_norms = image_norm_sq.sqrt()

    # GMM type estimator: initialize distance function (currently only supporting
    # Tagare distance) and other estimator params
    if estimator_type == "gmm":
        print("Initializing GMM estimator...")

        def distance_function(
            _unused_images: torch.Tensor,
            reference: torch.Tensor,
        ) -> torch.Tensor:
            return tagare_distance_precomputed(
                images_flat=masked_images_flat,
                image_norms=image_norms,
                image_norm_sq=image_norm_sq,
                reference=reference,
                beta=auto_beta,
            )

        estimator = RecursiveGMMEstimator(
            distance_function=distance_function,
            max_iter=EXTERNAL_GMM_MAX_ITER,
            tol=ESTIMATOR_TOL,
            standardize_distances=GMM_STANDARDIZE_DISTANCES,
            random_state=ESTIMATOR_RANDOM_STATE,
            gmm_max_iter=INTERNAL_GMM_MAX_ITER,
        )
    # IRLS type estimator: initialize weight function (currently only supporting
    # Tagare weights) and other estimator params
    elif estimator_type == "irls":
        print("Initializing IRLS M-estimator...")

        def weight_function(
            _unused_images: torch.Tensor,
            reference: torch.Tensor,
            _unused_std: torch.Tensor,
        ):
            return tagare_weight_precomputed(
                images_flat=masked_images_flat,
                image_norms=image_norms,
                image_norm_sq=image_norm_sq,
                reference=reference,
                beta=auto_beta,
            )

        estimator = IRLSMEstimator(
            weight_function=weight_function,
            max_iter=IRLS_MAX_ITER,
            tol=ESTIMATOR_TOL,
            damping_coef=IRLS_DAMPING_COEF,
        )
    else:
        raise ValueError(f"Unrecognized estimator type: {estimator_type}")

    return estimator


def process_class(
    data: pd.DataFrame,
    group_by_column: str,
    group_by_value: int,
    device: Union[torch.device, str] = "cpu",
    write_metadata: Optional[pd.DataFrame] = None,
    estimator_type: str = "gmm",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate robust and conventional averages for one particle class.

    Parameters
    ----------
    data : pandas.DataFrame
        Metadata describing the preprocessed particles.
    group_by_column: str
        Name of the column of ``data`` that should be used to group the particles.
        E.g. they could be grouped by class id.
    group_by_value : int
        Identifier of the class to process.
    device : torch.device or str, optional
        Device used for the estimation.
    write_metadata : pandas.DataFrame, optional
        Metadata table in which the calculated particle weights are stored.
        Particles are matched using their item identifiers.
    estimator_type, optional
        Type of estimator used to make the robust averaging. Can be 'gmm' or 'irls'.
        Default is 'gmm'.

    Returns
    -------
    numpy.ndarray
        Robust weighted class average.
    numpy.ndarray
        Conventional unweighted class average.
    """
    class_data = data[data[group_by_column] == group_by_value]
    images = read_images(data=class_data, device=device)

    # Mask the images and precompute the quantities reused by the distance
    # function during every recursive-estimator iteration.
    mask_np = create_circular_mask(
        image_shape=tuple(images.shape[1:]),
        radius=images.shape[1] // 2,
    )
    mask_tensor = torch.from_numpy(mask_np).to(
        device=images.device,
        dtype=images.dtype,
    )
    masked_images = images * mask_tensor

    estimator = initialize_estimator(
        unmasked_images=images,
        masked_images=masked_images,
        estimator_type=estimator_type,
    )
    reference = masked_images.mean(dim=0)

    # Fit the estimator, storing main weights as ``weights``
    if estimator_type == "gmm":
        print("Running the GMM estimator...")
        _, weights, original_distances = estimator.fit(
            images=masked_images, reference=reference
        )

        robust_weights_np = -original_distances.detach().cpu().numpy().reshape(-1)
        gmm_weights_np = weights.detach().cpu().numpy().reshape(-1)

    elif estimator_type == "irls":
        print("Running the IRLS estimator...")
        _, weights = estimator.fit(images=masked_images, reference=reference)

        robust_weights_np = weights.detach().cpu().numpy().reshape(-1)
        gmm_weights_np = None

    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")

    # Write weights to metadata file, ensuring the association is correct by using
    # the item ids
    if write_metadata is not None:
        item_ids = class_data[MDL_ITEM_ID_COLUMN].to_numpy()
        class_mask = write_metadata[MDL_ITEM_ID_COLUMN].isin(item_ids)
        target_item_ids = write_metadata.loc[class_mask, MDL_ITEM_ID_COLUMN]

        robust_weights_by_id = pd.Series(robust_weights_np, index=item_ids)

        print("Saving robust weights...")
        write_metadata.loc[class_mask, "wRobust"] = target_item_ids.map(
            robust_weights_by_id
        ).to_numpy()

        if gmm_weights_np is not None:
            print("Saving GMM weights...")
            gmm_weights_by_id = pd.Series(gmm_weights_np, index=item_ids)

            write_metadata.loc[class_mask, "wRobustGmm"] = target_item_ids.map(
                gmm_weights_by_id
            ).to_numpy()

    unmasked_new_average = weighted_average(images, weights).detach().cpu().numpy()
    unmasked_original_avg = images.mean(dim=0).detach().cpu().numpy()

    return unmasked_new_average, unmasked_original_avg


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            warnings.warn(
                "Requested CUDA compute device but CUDA is unavailable. "
                "Using CPU instead."
            )
            device = "cpu"
    else:
        device = "cpu"

    metadata_df = pd.DataFrame(starfile.read(args.input_xmd))
    validate_item_ids(metadata_df, name="Input")

    write_metadata = None
    if args.out_star:
        if args.base_xmd:
            write_metadata = pd.DataFrame(starfile.read(args.base_xmd))
            validate_item_ids(write_metadata, name="Base")
        else:
            write_metadata = metadata_df

        # Initialize the weight columns we will write with NaNs, they will later be
        # overwritten but this will help catch any particles that don't get assigned
        # any weights
        for column in ESTIMATOR_WEIGHT_COLUMNS[args.estimator_type]:
            write_metadata[column] = np.nan

    stack_name = str(metadata_df["image"].to_numpy()[0]).split("@", maxsplit=1)[1]
    stack_path = Path(stack_name)

    with mrcfile.open(stack_path, header_only=True) as mrc:
        nx = mrc.header.nx
        ny = mrc.header.ny

    group_by_column = args.group_by_column
    group_by_values = sorted(metadata_df[group_by_column].unique())
    n_classes = len(group_by_values)

    corrected_averages = None
    if args.out_corrected_avg:
        corrected_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    original_averages = None
    if args.out_original_avg:
        original_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    for index, class_value in enumerate(group_by_values):
        corrected_avg, original_avg = process_class(
            data=metadata_df,
            group_by_column=group_by_column,
            group_by_value=class_value,
            device=device,
            write_metadata=write_metadata,
            estimator_type=args.estimator_type,
        )

        if corrected_averages is not None:
            corrected_averages[index] = corrected_avg

        if original_averages is not None:
            original_averages[index] = original_avg

    # Save the robust, unmasked averages if requested.
    if args.out_corrected_avg:
        mrcfile.write(name=args.out_corrected_avg, data=corrected_averages)

    # Save the original, unmasked averages if requested.
    if args.out_original_avg:
        mrcfile.write(name=args.out_original_avg, data=original_averages)

    # Save the metadata with the additional weight columns.
    if write_metadata is not None:
        weight_columns = ESTIMATOR_WEIGHT_COLUMNS[args.estimator_type]

        if write_metadata[weight_columns].isna().any().any():
            raise RuntimeError("Some particles were not assigned weights.")

        starfile.write(data=write_metadata, filename=args.out_star)


if __name__ == "__main__":
    main()
