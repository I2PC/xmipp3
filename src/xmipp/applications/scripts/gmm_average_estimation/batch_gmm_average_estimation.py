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
from xmippPyModules.gmmAverageTools.distances import (
    calculate_beta_auto,
    tagare_distance_precomputed,
)
from xmippPyModules.gmmAverageTools.masks import create_circular_mask
from xmippPyModules.gmmAverageTools.utils import weighted_average

# Estimator parameters
# NOTE: to be changed for configurable arguments in the future
ESTIMATOR_MAX_ITER = 15
ESTIMATOR_TOL = 1.0e-4
ESTIMATOR_STANDARDIZE_DISTANCES = True
ESTIMATOR_RANDOM_STATE = 42
GMM_MAX_ITER = 20


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

    return parser


def process_class(
    data: pd.DataFrame,
    class_id: int,
    device: Union[torch.device, str] = "cpu",
    write_metadata: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate robust and conventional averages for one particle class.

    Parameters
    ----------
    data : pandas.DataFrame
        Metadata describing the preprocessed particles.
    class_id : int
        Identifier of the class to process.
    device : torch.device or str, optional
        Device used for the estimation.
    write_metadata : pandas.DataFrame, optional
        Metadata table in which the calculated particle weights are stored.
        Particles are matched using their item identifiers.

    Returns
    -------
    numpy.ndarray
        Robust weighted class average.
    numpy.ndarray
        Conventional unweighted class average.
    """
    class_data = data[data[MDL_REF_COLUMN] == class_id]
    images = read_images(data=class_data, device=device)

    # Calculate the automatic scaling parameter for the distance.
    auto_beta = calculate_beta_auto(
        imgs=images,
        mult=1.0,
    )

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

    masked_images_flat = masked_images.flatten(1)
    image_norm_sq = masked_images_flat.square().sum(dim=1)
    image_norms = image_norm_sq.sqrt()

    def distance_function(
        unused_images: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return tagare_distance_precomputed(
            images_flat=masked_images_flat,
            image_norms=image_norms,
            image_norm_sq=image_norm_sq,
            reference=reference,
            beta=auto_beta,
        )

    # Initialize the estimator and calculate its starting reference.
    estimator = RecursiveGMMEstimator(
        distance_function=distance_function,
        max_iter=ESTIMATOR_MAX_ITER,
        tol=ESTIMATOR_TOL,
        standardize_distances=ESTIMATOR_STANDARDIZE_DISTANCES,
        random_state=ESTIMATOR_RANDOM_STATE,
        gmm_max_iter=GMM_MAX_ITER,
    )
    reference = masked_images.mean(dim=0)

    # Fit the recursive estimator.
    _, weights, original_distances = estimator.fit(
        images=masked_images,
        reference=reference,
    )

    # Convert weights and distances to NumPy.
    gmm_weights = weights.detach().cpu().numpy().reshape(-1)
    original_weights = -original_distances.detach().cpu().numpy().reshape(-1)

    # Write weights to metadata file, ensuring the association is correct by using
    # the item ids
    if write_metadata is not None:
        item_ids = class_data[MDL_ITEM_ID_COLUMN].to_numpy()
        original_weights_by_id = pd.Series(original_weights, index=item_ids)
        gmm_weights_by_id = pd.Series(gmm_weights, index=item_ids)

        class_mask = write_metadata[MDL_REF_COLUMN] == class_id
        target_item_ids = write_metadata.loc[class_mask, MDL_ITEM_ID_COLUMN]

        write_metadata.loc[class_mask, "wRobust"] = target_item_ids.map(
            original_weights_by_id
        ).to_numpy()
        write_metadata.loc[class_mask, "wRobustGmm"] = target_item_ids.map(
            gmm_weights_by_id
        )

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
    metadata_df["wRobustGmm"] = 1.0
    metadata_df["wRobust"] = 1.0

    write_metadata = None
    if args.out_star:
        if args.base_xmd:
            write_metadata = pd.DataFrame(starfile.read(args.base_xmd))
        else:
            write_metadata = metadata_df

    stack_name = str(metadata_df["image"].to_numpy()[0]).split("@", maxsplit=1)[1]
    stack_path = Path(stack_name)

    with mrcfile.open(stack_path, header_only=True) as mrc:
        nx = mrc.header.nx
        ny = mrc.header.ny

    class_ids = sorted(metadata_df[MDL_REF_COLUMN].unique())
    n_classes = len(class_ids)

    corrected_averages = None
    if args.out_corrected_avg:
        corrected_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    original_averages = None
    if args.out_original_avg:
        original_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    for index, class_id in enumerate(class_ids):
        corrected_avg, original_avg = process_class(
            data=metadata_df,
            class_id=class_id,
            device=device,
            write_metadata=write_metadata,
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
        starfile.write(data=write_metadata, filename=args.out_star)


if __name__ == "__main__":
    main()
