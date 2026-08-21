#!/usr/bin/env python3

import time

# Start timing before the heavy imports. The elapsed time until main() starts
# includes Python-level startup and module imports.
PROCESS_START = time.perf_counter()

import argparse
import json
from pathlib import Path
import warnings

import mrcfile
import numpy as np
import torch

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
    parser.add_argument(
        "--timings-file",
        type=Path,
        help=(
            "Optional JSON file where execution times are accumulated "
            "across multiple calls to this script."
        ),
    )

    return parser


def _empty_accumulated_timings() -> dict[str, int | float]:
    """Return an empty timing summary compatible with the protocol output."""
    return {
        "n_calls": 0,
        "n_images": 0,
        "imports_and_startup": 0.0,
        "argument_parsing": 0.0,
        "device_setup": 0.0,
        "read_images": 0.0,
        "distance_setup": 0.0,
        "masking": 0.0,
        "estimator_setup": 0.0,
        "estimator_fit": 0.0,
        "result_conversion": 0.0,
        "write_corrected_average": 0.0,
        "write_original_average": 0.0,
        "write_metadata": 0.0,
        "write_optional_arrays": 0.0,
        "other": 0.0,
        "total": 0.0,
    }


def accumulate_timings(
    timings_path: Path,
    timings: dict[str, int | float],
) -> None:
    """Accumulate timings from multiple sequential script calls."""
    if timings_path.exists():
        with timings_path.open("r", encoding="utf-8") as file:
            accumulated = json.load(file)
    else:
        accumulated = _empty_accumulated_timings()

    # Keep compatibility with timing files produced by earlier versions.
    for key, default_value in _empty_accumulated_timings().items():
        accumulated.setdefault(key, default_value)

    accumulated["n_calls"] += 1
    accumulated["n_images"] += int(timings["n_images"])

    for name, elapsed in timings.items():
        if name != "n_images":
            accumulated[name] += float(elapsed)

    # Replace atomically after writing to avoid leaving a truncated JSON file.
    temporary_path = timings_path.with_suffix(timings_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(accumulated, file, indent=2)

    temporary_path.replace(timings_path)


def main() -> None:
    main_start = time.perf_counter()

    start = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()
    argument_parsing_time = time.perf_counter() - start

    timings: dict[str, int | float] = {
        "n_images": 0,
        "imports_and_startup": main_start - PROCESS_START,
        "argument_parsing": argument_parsing_time,
        "device_setup": 0.0,
        "read_images": 0.0,
        "distance_setup": 0.0,
        "masking": 0.0,
        "estimator_setup": 0.0,
        "estimator_fit": 0.0,
        "result_conversion": 0.0,
        "write_corrected_average": 0.0,
        "write_original_average": 0.0,
        "write_metadata": 0.0,
        "write_optional_arrays": 0.0,
        "other": 0.0,
        "total": 0.0,
    }

    start = time.perf_counter()

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

    timings["device_setup"] = time.perf_counter() - start

    # Read the aligned images referenced by the input metadata.
    start = time.perf_counter()
    images = read_images(
        xmd_path=args.input_xmd,
        device=device,
    )
    timings["read_images"] = time.perf_counter() - start
    timings["n_images"] = int(images.shape[0])

    # Calculate the automatic scaling parameter for the distance.
    start = time.perf_counter()
    auto_beta = calculate_beta_auto(
        imgs=images,
        mult=1.0,
    )
    timings["distance_setup"] = time.perf_counter() - start

    # Mask the images and precompute the quantities reused by the distance
    # function during every recursive-estimator iteration.
    start = time.perf_counter()

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

    timings["masking"] = time.perf_counter() - start

    # Initialize the estimator and calculate its starting reference.
    start = time.perf_counter()

    estimator = RecursiveGMMEstimator(
        distance_function=distance_function,
        max_iter=ESTIMATOR_MAX_ITER,
        tol=ESTIMATOR_TOL,
        standardize_distances=ESTIMATOR_STANDARDIZE_DISTANCES,
        random_state=ESTIMATOR_RANDOM_STATE,
        gmm_max_iter=GMM_MAX_ITER,
    )
    reference = masked_images.mean(dim=0)

    timings["estimator_setup"] = time.perf_counter() - start

    # Fit the recursive estimator.
    start = time.perf_counter()

    _, weights, original_distances = estimator.fit(
        images=masked_images,
        reference=reference,
    )

    timings["estimator_fit"] = time.perf_counter() - start

    # Convert weights and distances to NumPy.
    start = time.perf_counter()

    gmm_weights = weights.detach().cpu().numpy().reshape(-1)
    original_weights = -original_distances.detach().cpu().numpy().reshape(-1)

    timings["result_conversion"] = time.perf_counter() - start

    # Save the robust, unmasked average if requested.
    start = time.perf_counter()

    if args.out_corrected_avg:
        unmasked_new_average = weighted_average(images, weights).detach().cpu().numpy()
        mrcfile.write(
            name=args.out_corrected_avg,
            data=unmasked_new_average,
        )

    timings["write_corrected_average"] = time.perf_counter() - start

    # Save the original, unmasked average if requested.
    start = time.perf_counter()

    if args.out_original_avg:
        unmasked_original_avg = images.mean(dim=0).detach().cpu().numpy()
        mrcfile.write(
            name=args.out_original_avg,
            data=unmasked_original_avg,
        )

    timings["write_original_average"] = time.perf_counter() - start

    # Save the metadata with the additional weight columns.
    start = time.perf_counter()

    if args.out_star:
        base_xmd = args.base_xmd if args.base_xmd is not None else args.input_xmd

        write_star_with_weights(
            input_star=base_xmd,
            output_star=args.out_star,
            weights_list=[gmm_weights, original_weights],
            column_names=["wRobustGmm", "wRobust"],
        )

    timings["write_metadata"] = time.perf_counter() - start

    # Save optional standalone NumPy arrays.
    start = time.perf_counter()

    if args.out_weights:
        np.save(args.out_weights, gmm_weights)

    if args.out_distances:
        np.save(args.out_distances, original_weights)

    timings["write_optional_arrays"] = time.perf_counter() - start

    # PROCESS_START precedes the heavy imports, so total includes their cost.
    timings["total"] = time.perf_counter() - PROCESS_START

    measured_time = sum(
        float(elapsed)
        for name, elapsed in timings.items()
        if name not in {"n_images", "total", "other"}
    )
    timings["other"] = max(
        0.0,
        float(timings["total"]) - measured_time,
    )

    if args.timings_file is not None:
        accumulate_timings(
            timings_path=args.timings_file,
            timings=timings,
        )


if __name__ == "__main__":
    main()
