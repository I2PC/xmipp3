#!/usr/bin/env python3

import time

# Start timing before the heavy imports. The elapsed time until main() starts
# includes Python-level startup and module imports.
PROCESS_START = time.perf_counter()

import argparse
import json
from pathlib import Path
import warnings
from typing import Dict, Tuple, Union

import mrcfile
import numpy as np
import starfile
import torch
import pandas as pd

import pwem.emlib.metadata as md

from xmippPyModules.gmmAverageTools.data import (
    read_images,
    write_star_with_weights,
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
        "--out-weights",
        type=Path,
        help="Path to the output .npy file for the GMM weights",
    )
    parser.add_argument(
        "--out-distances",
        type=Path,
        help="Path to the output .npy file for the original distances",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Compute device for PyTorch",
    )

    return parser


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

    # Read the aligned images referenced by the input metadata.
    images = read_images(
        xmd_path=args.input_xmd,
        device=device,
    )

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

    # Save the robust, unmasked average if requested.
    if args.out_corrected_avg:
        unmasked_new_average = weighted_average(images, weights).detach().cpu().numpy()
        mrcfile.write(
            name=args.out_corrected_avg,
            data=unmasked_new_average,
        )

    # Save the original, unmasked average if requested.
    if args.out_original_avg:
        unmasked_original_avg = images.mean(dim=0).detach().cpu().numpy()
        mrcfile.write(
            name=args.out_original_avg,
            data=unmasked_original_avg,
        )

    # Save the metadata with the additional weight columns.
    if args.out_star:
        base_xmd = args.base_xmd if args.base_xmd is not None else args.input_xmd

        write_star_with_weights(
            input_star=base_xmd,
            output_star=args.out_star,
            weights_list=[gmm_weights, original_weights],
            column_names=["wRobustGmm", "wRobust"],
        )

    # Save optional standalone NumPy arrays.
    if args.out_weights:
        np.save(args.out_weights, gmm_weights)

    if args.out_distances:
        np.save(args.out_distances, original_weights)


if __name__ == "__main__":
    main()
