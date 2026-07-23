#!/usr/bin/env python3

import argparse
from pathlib import Path
from functools import partial
import warnings

import mrcfile
import numpy as np
import torch

from xmippPyModules.gmmAverageTools.data import read_data, write_star_with_weights
from xmippPyModules.gmmAverageTools.alignment import align_particles_batch
from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator
from xmippPyModules.gmmAverageTools.distances import (
    tagare_distance,
    calculate_beta_auto,
)
from xmippPyModules.gmmAverageTools.masks import create_circular_mask

# Estimator parameters
# NOTE: to be changed for configurable arguments in the future
ESTIMATOR_MAX_ITER = 15
ESTIMATOR_TOL = 1.0e-4
ESTIMATOR_STANDARDIZE_DISTANCES = True
ESTIMATOR_RANDOM_STATE = 42

# Batch size for the alignment
# NOTE: to be changed for configurable arguments in the future
BATCH_SIZE = 256


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-xmd",
        required=True,
        type=Path,
        help="Path to the .xmd file containing the path to the image stack and the "
        "alignment parameters",
    )
    parser.add_argument(
        "--base-xmd",
        type=Path,
        help="Path to the base .xmd file the weights should be added to. " \
        "The original file will not be modified, but a new one will be created with the " \
        "same information, plus the weights." \
        "If not specified, the input .xmd file will be used.",
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
        help="Compute device for PyTorch"
    )
    parser.add_argument(
        "--rotate-first",
        action="store_true",
        default=False,
        help="When aligning the images, apply rotation before shifts (XMIPP convention)",
    )

    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.device == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            warnings.warn("Requested CUDA compute device but CUDA is unavailable. Using CPU instead.")
            device = "cpu"
    else:
        device = "cpu"

    # Read data from the input star file
    particles, angles, shiftX, shiftY = read_data(
        xmd_path=args.input_xmd, device=device
    )

    images = align_particles_batch(
        particles,
        psi=angles,
        shiftX=shiftX,
        shiftY=shiftY,
        shift_first=not args.rotate_first,
        batch_size=BATCH_SIZE,
    )

    # Initialize distance function for the estimator
    auto_beta = calculate_beta_auto(imgs=images, mult=1.0)
    distance_function = partial(tagare_distance, beta=auto_beta)

    # Mask images for the estimator
    # NOTE: probably have to change this to avoid creating a new image array in memory
    mask_np = create_circular_mask(
        image_shape=tuple(images.shape[1:]), radius=images.shape[1] // 2
    )
    mask_tensor = torch.from_numpy(mask_np).to(device=images.device, dtype=images.dtype)
    masked_images = images * mask_tensor

    # Initialize the estimator
    estimator = RecursiveGMMEstimator(
        distance_function=distance_function,
        max_iter=ESTIMATOR_MAX_ITER,
        tol=ESTIMATOR_TOL,
        standardize_distances=ESTIMATOR_STANDARDIZE_DISTANCES,
        random_state=ESTIMATOR_RANDOM_STATE,
    )

    # Calculate the initial reference
    reference = masked_images.mean(dim=0)

    # Fit the estimator to get the corrected average and the weights
    new_avg, weights, original_distances = estimator.fit(
        images=masked_images, reference=reference
    )
    new_avg = new_avg.detach().cpu().numpy()

    # NOTE: for global GMM weights, reshaping to a flat array is enough,
    # but more intricate weight aggregation might be necessary in the future
    gmm_weights = weights.detach().cpu().numpy().reshape(-1)

    # The GMM estimator returns distances, which are the negative of the weights
    original_weights = -original_distances.detach().cpu().numpy().reshape(-1)

    # Save the new average if requested
    if args.out_corrected_avg:
        mrcfile.write(name=args.out_corrected_avg, data=new_avg)

    # Save the original average if requested
    if args.out_original_avg:
        mrcfile.write(name=args.out_original_avg, data=reference.detach().cpu().numpy())

    # Save the new star file with the weights added as a column
    if args.out_star:
        if not args.base_xmd:
            args.base_xmd = args.input_xmd
        write_star_with_weights(
            input_star=args.base_xmd,
            output_star=args.out_star,
            weights_list=[gmm_weights, original_weights],
            column_names=["wRobustGmm", "wRobust"],
        )
    
    # Save weights and distances as separate .npy files if requested
    if args.out_weights:
        np.save(args.out_weights, gmm_weights)
    if args.out_distances:
        np.save(args.out_distances, original_weights)

if __name__ == "__main__":
    main()
