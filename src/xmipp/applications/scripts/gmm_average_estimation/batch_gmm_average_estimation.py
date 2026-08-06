#!/usr/bin/env python3

import time

# Start timing before the heavy imports. The elapsed time until main() starts
# includes Python-level startup and module imports.
PROCESS_START = time.perf_counter()

import argparse
import json
from pathlib import Path
import warnings
from typing import Optional, Dict, Union, TypedDict

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

# Config file format
BATCH_FORMAT_VERSION = 1
BATCH_REQUIRED_PATH_FIELDS = (
    "input_xmd",
    "base_xmd",
    "out_star",
    "out_corrected_avg",
    "out_original_avg",
)
BATCH_OPTIONAL_PATH_FIELDS = ("out_weights", "out_distances")


class ClassConfig(TypedDict):
    class_id: Optional[int]
    input_xmd: Path
    base_xmd: Optional[Path]
    out_star: Optional[Path]
    out_corrected_avg: Optional[Path]
    out_original_avg: Optional[Path]
    out_weights: Optional[Path]
    out_distances: Optional[Path]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-xmd",
        type=Path,
        help="Path to the .xmd file containing the path to the image stack",
    )
    input_group.add_argument(
        "--batch-config",
        type=Path,
        help="Path to a JSON configuration containing one or more classes",
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
        default="cpu",
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


def load_batch_config(batch_config_path: Path) -> list[ClassConfig]:
    """Load class-specific inputs and outputs from a versioned JSON file."""
    with batch_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Batch configuration must be a JSON object.")

    if config.get("format_version") != BATCH_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported batch format_version {config.get('format_version')!r}; "
            f"expected {BATCH_FORMAT_VERSION}."
        )

    classes = config.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Batch configuration must contain a non-empty 'classes' list.")

    normalized_classes: list[ClassConfig] = []
    for class_config in classes:
        if not isinstance(class_config, dict):
            raise ValueError("Each entry in 'classes' must be a JSON object.")

        required_fields = ("class_id", *BATCH_REQUIRED_PATH_FIELDS)
        missing = [field for field in required_fields if field not in class_config]
        if missing:
            raise ValueError("Missing batch fields: " + ", ".join(missing))

        normalized: ClassConfig = {
            "class_id": class_config["class_id"],
            "input_xmd": Path(class_config["input_xmd"]),
            "base_xmd": Path(class_config["base_xmd"]),
            "out_star": Path(class_config["out_star"]),
            "out_corrected_avg": Path(class_config["out_corrected_avg"]),
            "out_original_avg": Path(class_config["out_original_avg"]),
            "out_weights": None,
            "out_distances": None,
        }

        for field in BATCH_OPTIONAL_PATH_FIELDS:
            value = class_config.get(field)
            normalized[field] = Path(value) if value is not None else None

        normalized_classes.append(normalized)

    return normalized_classes


def _class_configs_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[ClassConfig]:
    """Return one uniform configuration list for either supported CLI mode."""
    class_specific_options = (
        "base_xmd",
        "out_star",
        "out_corrected_avg",
        "out_original_avg",
        "out_weights",
        "out_distances",
    )

    if args.batch_config is not None:
        conflicting = [
            f"--{name.replace('_', '-')}"
            for name in class_specific_options
            if getattr(args, name) is not None
        ]
        if conflicting:
            parser.error(
                "--batch-config cannot be combined with class-specific options: "
                + ", ".join(conflicting)
            )

        try:
            return load_batch_config(args.batch_config)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            parser.error(f"Could not load batch configuration: {exc}")

    return [
        {
            "class_id": None,
            "input_xmd": args.input_xmd,
            "base_xmd": args.base_xmd,
            "out_star": args.out_star,
            "out_corrected_avg": args.out_corrected_avg,
            "out_original_avg": args.out_original_avg,
            "out_weights": args.out_weights,
            "out_distances": args.out_distances,
        }
    ]


def _resolve_device(requested_device: str) -> str:
    """Resolve the requested PyTorch device, falling back to CPU if needed."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Requested CUDA compute device but CUDA is unavailable. Using CPU instead."
        )
        return "cpu"

    return requested_device


def _empty_accumulated_timings() -> Dict[str, Union[int, float]]:
    """Return an empty timing summary compatible with the protocol output."""
    return {
        "n_calls": 0,
        "n_classes": 0,
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


def _empty_run_timings() -> Dict[str, Union[int, float]]:
    timings = _empty_accumulated_timings()
    timings.pop("n_calls")
    return timings


def accumulate_timings(
    timings_path: Path,
    timings: dict[str, Union[int, float]],
) -> None:
    """Accumulate timings from one script invocation."""
    if timings_path.exists():
        with timings_path.open("r", encoding="utf-8") as file:
            accumulated = json.load(file)
    else:
        accumulated = _empty_accumulated_timings()

    for key, default_value in _empty_accumulated_timings().items():
        accumulated.setdefault(key, default_value)

    accumulated["n_calls"] += 1

    for name, value in timings.items():
        accumulated[name] += value

    temporary_path = timings_path.with_suffix(timings_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(accumulated, file, indent=2)

    temporary_path.replace(timings_path)


def _add_class_timings(
    total_timings: Dict[str, Union[int, float]],
    class_timings: Dict[str, Union[int, float]],
) -> None:
    total_timings["n_classes"] += 1
    for name, value in class_timings.items():
        total_timings[name] += value


def process_class(
    *,
    input_xmd: Path,
    device: str,
    base_xmd: Optional[Path] = None,
    out_star: Optional[Path] = None,
    out_corrected_avg: Optional[Path] = None,
    out_original_avg: Optional[Path] = None,
    out_weights: Optional[Path] = None,
    out_distances: Optional[Path] = None,
) -> Dict[str, Union[int, float]]:
    """
    Process one class and write the requested estimation outputs.

    The function contains all work associated with one class and is called
    sequentially for every entry in a batch configuration. Python startup,
    imports and device selection remain in :func:`main`.

    Parameters
    ----------
    input_xmd
        Metadata referencing the aligned particle stack.
    device
        Resolved PyTorch device, currently ``"cpu"`` or ``"cuda"``.
    base_xmd
        Metadata to which the output weight columns are added. If omitted,
        ``input_xmd`` is used.
    out_star
        Output metadata path for the calculated weights.
    out_corrected_avg
        Output path for the robust, weighted average.
    out_original_avg
        Output path for the unweighted average.
    out_weights
        Optional NumPy output path for the GMM weights.
    out_distances
        Optional NumPy output path for the negated estimator distances,
        preserving the current command-line behavior.

    Returns
    -------
    dict
        Timing values for the class-specific stages, including the number of
        processed images.
    """
    timings: dict[str, int | float] = {
        "n_images": 0,
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
    }

    # Read the aligned images referenced by the input metadata.
    start = time.perf_counter()
    images = read_images(
        xmd_path=input_xmd,
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
    tagare_weights = -original_distances.detach().cpu().numpy().reshape(-1)

    timings["result_conversion"] = time.perf_counter() - start

    # Save the robust, unmasked average if requested.
    start = time.perf_counter()

    if out_corrected_avg:
        unmasked_new_average = weighted_average(images, weights).detach().cpu().numpy()
        mrcfile.write(
            name=out_corrected_avg,
            data=unmasked_new_average,
        )

    timings["write_corrected_average"] = time.perf_counter() - start

    # Save the original, unmasked average if requested.
    start = time.perf_counter()

    if out_original_avg:
        unmasked_original_avg = images.mean(dim=0).detach().cpu().numpy()
        mrcfile.write(
            name=out_original_avg,
            data=unmasked_original_avg,
        )

    timings["write_original_average"] = time.perf_counter() - start

    # Save the metadata with the additional weight columns.
    start = time.perf_counter()

    if out_star:
        metadata_source = base_xmd if base_xmd is not None else input_xmd

        write_star_with_weights(
            input_star=metadata_source,
            output_star=out_star,
            weights_list=[gmm_weights, tagare_weights],
            column_names=["wRobustGmm", "wRobust"],
        )

    timings["write_metadata"] = time.perf_counter() - start

    # Save optional standalone NumPy arrays.
    start = time.perf_counter()

    if out_weights:
        np.save(out_weights, gmm_weights)

    if out_distances:
        np.save(out_distances, tagare_weights)

    timings["write_optional_arrays"] = time.perf_counter() - start

    return timings


def main() -> None:
    main_start = time.perf_counter()

    start = time.perf_counter()
    parser = build_argument_parser()
    args = parser.parse_args()
    class_configs = _class_configs_from_args(parser, args)
    argument_parsing_time = time.perf_counter() - start

    start = time.perf_counter()
    device = _resolve_device(args.device)
    device_setup_time = time.perf_counter() - start

    timings = _empty_run_timings()

    for index, class_config in enumerate(class_configs, start=1):
        class_id = class_config["class_id"]
        label = class_id if class_id is not None else index
        print(f"Processing class {index}/{len(class_configs)} (ID {label})")

        try:
            class_timings = process_class(
                input_xmd=class_config["input_xmd"],
                base_xmd=class_config["base_xmd"],
                out_star=class_config["out_star"],
                out_corrected_avg=class_config["out_corrected_avg"],
                out_original_avg=class_config["out_original_avg"],
                out_weights=class_config["out_weights"],
                out_distances=class_config["out_distances"],
                device=device,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to process class {label}.") from exc

        _add_class_timings(timings, class_timings)

    timings["imports_and_startup"] = main_start - PROCESS_START
    timings["argument_parsing"] = argument_parsing_time
    timings["device_setup"] = device_setup_time

    # PROCESS_START precedes the heavy imports, so total includes their cost.
    timings["total"] = time.perf_counter() - PROCESS_START

    measured_time = sum(
        float(elapsed)
        for name, elapsed in timings.items()
        if name not in {"n_classes", "n_images", "total", "other"}
    )
    timings["other"] = max(0.0, float(timings["total"]) - measured_time)

    if args.timings_file is not None:
        accumulate_timings(
            timings_path=args.timings_file,
            timings=timings,
        )


if __name__ == "__main__":
    main()
