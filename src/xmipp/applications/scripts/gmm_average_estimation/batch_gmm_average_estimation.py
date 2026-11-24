#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
from xmippPyModules.gmmAverageTools.admm_estimator import ADMMEstimator
from xmippPyModules.gmmAverageTools.data import (
    MDL_ITEM_ID_COLUMN,
    MDL_REF_COLUMN,
    read_images,
)
from xmippPyModules.gmmAverageTools.fourier_irls_estimator import (
    JointIRLSFourier,
    WeightApproach,
)

# Import estimator types
from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator
from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator

# Utilities: masks, weighted averages
from xmippPyModules.gmmAverageTools.masks import (
    create_circular_mask,
    create_lowpass_rfft_mask,
)
from xmippPyModules.gmmAverageTools.utils import weighted_average

# Import weight and distance functions
from xmippPyModules.gmmAverageTools.weights import (
    calculate_beta_auto,
    smooth_redescending_weights_modulus,
    smooth_redescending_weights_norm,
    tagare_weight_precomputed,
)

Estimator: TypeAlias = (
    RecursiveGMMEstimator | ADMMEstimator | JointIRLSFourier | IRLSMEstimator
)
EstimatorType = Literal["irls", "fourier_irls", "admm"]
ESTIMATOR_TYPES: tuple[str, ...] = get_args(EstimatorType)

UNASSIGNED_GROUP_VALUE = -1

DEFAULT_SMOOTH_DELTA: dict[WeightApproach, float] = {
    "per-image": 1.5,
    "per-coefficient": 1.5,
}

DEFAULT_MAX_ITERATIONS: dict[EstimatorType, int] = {
    "irls": 50,
    "fourier_irls": 50,
    "admm": 30,
}

DEFAULT_LOWPASS_MASK_CUTOFF = 0.25

ROBUST_WEIGHT_COL = "wRobust"
STD_ROBUST_WEIGHT_COL = "wRobustStd"
GMM_WEIGHT_COL = "wRobustGmm"

DEFAULT_OUTLIER_PROPORTION_FOR_AVERAGES = 0.1


@dataclass
class IOConfig:
    input_xmd: Path
    base_xmd: Path
    out_star: Path | None = None
    out_corrected_avgs: Path | None = None
    out_original_avgs: Path | None = None
    out_low_weight_avgs: Path | None = None
    out_high_weight_avgs: Path | None = None
    outlier_proportion_for_avgs: float = DEFAULT_OUTLIER_PROPORTION_FOR_AVERAGES
    group_by_column: str = MDL_REF_COLUMN
    overwrite: bool = False


@dataclass
class GMMConfig:
    external_max_iter: int
    internal_max_iter: int
    initial_bad_weight: float
    initial_bad_quantile: float
    check_degenerate: bool
    min_component_separation: float
    min_good_component_weight: float


@dataclass
class EstimatorConfig:
    estimator_type: EstimatorType
    max_iter: int
    tolerance: float
    random_state: int
    damping_coef: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    device: str
    io: IOConfig
    estimator: EstimatorConfig
    gmm: GMMConfig | None = None


@dataclass
class ClassProcessingResults:
    """
    Object containing the results of processing a class that will be used later
    in the pipeline.

    Attributes
    ----------
    unmasked_corrected_average : np.ndarray
        The robust weighted average produced by the estimator.
    unmasked_original_average : np.ndarray
        The conventional unweighted class average.
    robust_weights : np.ndarray
        Array of shape ``(n,)`` containing the weight (or aggregated per-image score)
        each image received in the robust estimation.
        For GMM estimators, this will contain the pre-GMM weights.
    gmm_weights : np.ndarray, optional
        Array of shape ``(n,)`` containing the weight each image received after
        GMM reweighting. Set to None for non-GMM estimators. Default is None.
    unmasked_low_weight_average, unmkased_high_weight_average : np.ndarray, optional
        The (unweighted) averages of the 10 percent of images with lowest
        and highest weights, respectively.
        These can be None if said subsets are not well-defined (e.g., because the
        estimator produced constant weights), or if they have not been requested.
    """

    unmasked_corrected_average: np.ndarray
    unmasked_original_average: np.ndarray
    robust_weights: np.ndarray
    gmm_weights: np.ndarray | None = None
    unmasked_low_weight_average: np.ndarray | None = None
    unmasked_high_weight_average: np.ndarray | None = None


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robust estimation pipeline for 2D classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Compute device for PyTorch",
    )

    # Input/Output parameters
    io_group = parser.add_argument_group("Input/Output parameters")
    io_group.add_argument(
        "--input-xmd",
        required=True,
        type=Path,
        help=(
            "Path to the .xmd file containing paths to the images and the "
            "classification info."
        ),
    )
    io_group.add_argument(
        "--base-xmd",
        type=Path,
        help=(
            "Path to the base .xmd file the weights should be added to. "
            "The original file will not be modified, but a new one will be "
            "created with the same information plus the weights. "
            "Defaults to --input-xmd if omitted."
        ),
    )
    io_group.add_argument(
        "--out-star",
        type=Path,
        help="Path to the output .star file",
    )
    io_group.add_argument(
        "--out-corrected-avgs",
        type=Path,
        help="Path to output .mrcs file for corrected class averages",
    )
    io_group.add_argument(
        "--out-original-avgs",
        type=Path,
        help="Path to output .mrcs file for original class averages",
    )
    io_group.add_argument(
        "--out-low-weight-avgs",
        type=Path,
        help="Path to output .mrcs file for the average of the images with lowest weights in each class",
    )
    io_group.add_argument(
        "--out-high-weight-avgs",
        type=Path,
        help="Path to output .mrcs file for the average of the images with highest weights in each class",
    )
    io_group.add_argument(
        "--outlier-proportion-for-avgs",
        type=float,
        default=DEFAULT_OUTLIER_PROPORTION_FOR_AVERAGES,
        help="Proportion of images to use to calculate averages of images with lowest or highest weights",
    )
    io_group.add_argument(
        "--group-by-column",
        type=str,
        default=MDL_REF_COLUMN,
        help=f"Column by which images are grouped (default: '{MDL_REF_COLUMN}')",
    )
    io_group.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        help="Overwrite image stacks if they already exist on the requested paths.",
        default=False,
    )

    # Shared estimator hyperparameters
    estimator_group = parser.add_argument_group("Shared estimator options")
    estimator_group.add_argument(
        "--estimator-random-state",
        type=int,
        default=42,
        help="Random seed for estimator initialization",
    )
    estimator_group.add_argument(
        "--estimator-tolerance",
        type=float,
        default=1.0e-4,
        help=(
            "Tolerance convergence threshold for the estimator. Iterative methods "
            "will stop when the relative change to the reference is below this "
            "threshold (or when they reach the specified maximum number of iterations)."
        ),
    )
    estimator_group.add_argument(
        "--estimator-max-iter",
        type=int,
        help=(
            "Maximum number of iterations for the estimator. For GMM estimators, "
            "this means the number of iterations for the most external GMM layer."
        ),
    )
    estimator_group.add_argument("--damping-coef", type=float, default=0.0)

    # GMM options
    gmm_group = parser.add_argument_group("GMM reweighting options")
    gmm_group.add_argument(
        "--gmm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply GMM reweighting after the main estimator",
    )

    gmm_group.add_argument("--gmm-external-max-iter", type=int, default=15)
    gmm_group.add_argument(
        "--gmm-internal-max-iter",
        type=int,
        default=25,
        help=(
            "Number of iterations used to fit the internal GMM model on each "
            "external iteration."
        ),
    )
    gmm_group.add_argument(
        "--gmm-initial-bad-weight",
        type=float,
        help="Initial weight for the GMM component with a lower mean.",
    )
    gmm_group.add_argument(
        "--gmm-initial-bad-quantile",
        type=float,
        help=(
            "Quantile (between 0 and 0.5) used to initialize the mean of the GMM "
            "component with a lower mean weight. This is a quantile for the weights, "
            "so the estimator will internally flip it to a value between 0.5 and 1."
        ),
    )
    gmm_group.add_argument(
        "--gmm-check-degenerate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check GMM model for degeneracy after final iteration.",
    )
    gmm_group.add_argument(
        "--gmm-min-component-sep",
        type=float,
        default=0.5,
        help="Minimum relative separation between the two GMM components.",
    )
    gmm_group.add_argument(
        "--gmm-min-good-weight",
        type=float,
        default=0.4,
        help="Minimum weight for the good (lower distance) component of the GMM model.",
    )

    # Subparsers for algorithm-specific hyperparameters
    subparsers = parser.add_subparsers(
        dest="estimator_type",
        required=True,
        help=f"Choice of estimation procedure. Options are {ESTIMATOR_TYPES}.",
    )

    # IRLS options
    subparsers.add_parser("irls", help="Standard IRLS procedure")

    # ADMM options
    admm_parser = subparsers.add_parser("admm", help="ADMM optimization procedure")
    admm_parser.add_argument("--initial-mu", type=float, default=1.0)
    admm_parser.add_argument("--fourier-multiplier", type=float, default=1.0)
    admm_parser.add_argument(
        "--internal-max-iter",
        type=int,
        default=15,
        help="Maximum number of iterations for the real-space and Fourier-space sub-estimators",
    )
    admm_parser.add_argument(
        "--fourier-delta",
        type=float,
        help="Delta scale parameter for the Fourier estimator",
    )
    admm_parser.add_argument(
        "--fourier-weight-approach",
        type=str,
        choices=get_args(WeightApproach),
        default="per-coefficient",
        help="Weight approach for the Fourier estimator",
    )
    admm_parser.add_argument(
        "--fourier-lowpass-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a lowpass mask in the Fourier estimator",
    )
    admm_parser.add_argument(
        "--fourier-lowpass-mask-cutoff",
        type=float,
        help=(
            "Normalized lowpass mask cutoff frequency. Only used if "
            "'--fourier-lowpass-mask' is also passed as an argument "
            f"(default is {DEFAULT_LOWPASS_MASK_CUTOFF})."
        ),
    )

    # Fourier IRLS options
    fourier_irls_parser = subparsers.add_parser(
        "fourier_irls", help="IRLS procedure on Fourier coefficients"
    )
    fourier_irls_parser.add_argument(
        "--delta",
        type=float,
        help=(
            "Delta scale parameter for redescending weights (defaults based "
            "on --weight-approach)"
        ),
    )
    fourier_irls_parser.add_argument(
        "--weight-approach",
        type=str,
        choices=get_args(WeightApproach),
        default="per-coefficient",
    )
    fourier_irls_parser.add_argument(
        "--lowpass-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use lowpass mask",
    )
    fourier_irls_parser.add_argument(
        "--lowpass-mask-cutoff",
        type=float,
        help=(
            "Normalized lowpass mask cutoff frequency. Only used if '--lowpass-mask' "
            f"is also passed as an argument (default is {DEFAULT_LOWPASS_MASK_CUTOFF})."
        ),
    )

    return parser


def parse_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    io_cfg = IOConfig(
        input_xmd=args.input_xmd,
        base_xmd=args.base_xmd or args.input_xmd,
        out_star=args.out_star,
        out_corrected_avgs=args.out_corrected_avgs,
        out_original_avgs=args.out_original_avgs,
        out_low_weight_avgs=args.out_low_weight_avgs,
        out_high_weight_avgs=args.out_high_weight_avgs,
        outlier_proportion_for_avgs=args.outlier_proportion_for_avgs,
        group_by_column=args.group_by_column,
        overwrite=args.overwrite,
    )

    if args.estimator_type == "fourier_irls":
        if args.lowpass_mask and args.lowpass_mask_cutoff is None:
            args.lowpass_mask_cutoff = DEFAULT_LOWPASS_MASK_CUTOFF
        method_params = {
            "delta": args.delta,
            "weight_approach": args.weight_approach,
            "lowpass_mask": args.lowpass_mask,
            "lowpass_mask_cutoff": args.lowpass_mask_cutoff,
        }
    elif args.estimator_type == "admm":
        if args.fourier_lowpass_mask and args.fourier_lowpass_mask_cutoff is None:
            args.fourier_lowpass_mask_cutoff = DEFAULT_LOWPASS_MASK_CUTOFF
        method_params = {
            "initial_mu": args.initial_mu,
            "fourier_multiplier": args.fourier_multiplier,
            "internal_max_iter": args.internal_max_iter,
            "fourier_delta": args.fourier_delta,
            "fourier_weight_approach": args.fourier_weight_approach,
            "fourier_lowpass_mask": args.fourier_lowpass_mask,
            "fourier_lowpass_mask_cutoff": args.fourier_lowpass_mask_cutoff,
        }
    else:
        method_params = {}

    # Resolve dynamic default for 'delta' and 'fourier_delta'
    for param_prefix in ["", "fourier_"]:
        param_name = param_prefix + "delta"
        if method_params.get(param_name) is None:
            weight_approach_name = param_prefix + "weight_approach"
            approach = method_params.get(weight_approach_name)
            if approach in DEFAULT_SMOOTH_DELTA:
                method_params[param_name] = DEFAULT_SMOOTH_DELTA[approach]

    # Resolve dynamic default for max_iter
    max_iter = args.estimator_max_iter
    if max_iter is None:
        max_iter = DEFAULT_MAX_ITERATIONS.get(args.estimator_type, 50)

    estimator_cfg = EstimatorConfig(
        estimator_type=args.estimator_type,
        max_iter=max_iter,
        tolerance=args.estimator_tolerance,
        random_state=args.estimator_random_state,
        damping_coef=args.damping_coef,
        params=method_params,
    )

    gmm_cfg = None
    if args.gmm:
        gmm_cfg = GMMConfig(
            external_max_iter=args.gmm_external_max_iter,
            internal_max_iter=args.gmm_internal_max_iter,
            initial_bad_weight=args.gmm_initial_bad_weight,
            # GMM works with distances internally, so the quantile has to be flipped
            initial_bad_quantile=(
                None
                if args.gmm_initial_bad_quantile is None
                else 1.0 - args.gmm_initial_bad_quantile
            ),
            check_degenerate=args.gmm_check_degenerate,
            min_component_separation=args.gmm_min_component_sep,
            min_good_component_weight=args.gmm_min_good_weight,
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Requested CUDA compute device but CUDA is unavailable. Using CPU instead."
        )
        device = "cpu"

    return PipelineConfig(
        device=device, io=io_cfg, estimator=estimator_cfg, gmm=gmm_cfg
    )


def validate_item_ids(data: pd.DataFrame, name: str) -> None:
    """Validate that metadata contains unique, nonmissing particle IDs."""
    if MDL_ITEM_ID_COLUMN not in data.columns:
        raise ValueError(f"{name} metadata does not contain '{MDL_ITEM_ID_COLUMN}'.")

    if data[MDL_ITEM_ID_COLUMN].isna().any():
        raise ValueError(f"{name} metadata contains missing particle identifiers.")

    if data[MDL_ITEM_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"{name} metadata contains duplicated '{MDL_ITEM_ID_COLUMN}' values."
        )


def validate_assignments(data: pd.DataFrame, group_by_column: str) -> None:
    """Validate particle IDs, nonnegative integer groups, and unique assignments."""
    keys = [MDL_ITEM_ID_COLUMN, group_by_column]

    missing_columns = [key for key in keys if key not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Input metadata is missing required columns: {missing_columns}."
        )

    if data[keys].isna().any().any():
        raise ValueError("Input metadata contains missing particle or group IDs.")

    # Normalize group values for validation, without modifying the input.
    pairs = data[keys].copy()
    pairs[group_by_column] = pd.to_numeric(
        pairs[group_by_column],
        errors="raise",
    )
    groups = pairs[group_by_column].to_numpy()

    if (
        not np.isfinite(groups).all()
        or (groups < 0).any()
        or (groups != np.floor(groups)).any()
    ):
        raise ValueError("Group identifiers must be finite, nonnegative integers.")

    if pairs.duplicated().any():
        raise ValueError(
            "Input metadata contains repeated particle assignments "
            "within the same group."
        )


def _initialize_irls(
    config: EstimatorConfig, unmasked_images: torch.Tensor, masked_images: torch.Tensor
) -> IRLSMEstimator:
    # Calculate the automatic scaling parameter for the distance or weight functions.
    auto_beta = calculate_beta_auto(imgs=unmasked_images, mult=1.0)

    # Calculate norms and flatten images for precomputed versions of
    # distance or weight functions
    masked_images_flat = masked_images.flatten(1)
    image_norm_sq = masked_images_flat.square().sum(dim=1)
    image_norms = image_norm_sq.sqrt()

    def tagare_weight_function(
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

    return IRLSMEstimator(
        weight_function=tagare_weight_function,
        max_iter=config.max_iter,
        tol=config.tolerance,
        damping_coef=config.damping_coef,
    )


def _initialize_fourier_irls(
    config: EstimatorConfig, unmasked_images: torch.Tensor, masked_images: torch.Tensor
) -> JointIRLSFourier:
    params = config.params

    if params["weight_approach"] == "per-coefficient":
        weight_function = partial(
            smooth_redescending_weights_modulus, delta=params["delta"]
        )
    else:
        weight_function = partial(
            smooth_redescending_weights_norm, delta=params["delta"]
        )

    use_mask = params.get("lowpass_mask")
    mask = None
    if use_mask:
        mask = create_lowpass_rfft_mask(
            image_shape=unmasked_images.shape[1:],
            cutoff=params["lowpass_mask_cutoff"],
            unit="normalized",
        )

    irls_solver = IRLSMEstimator(
        weight_function=weight_function,
        max_iter=config.max_iter,
        tol=config.tolerance,
        damping_coef=config.damping_coef,
    )

    return JointIRLSFourier(
        irls_solver=irls_solver, weight_approach=params["weight_approach"], mask=mask
    )


def _initialize_admm(
    config: EstimatorConfig, unmasked_images: torch.Tensor, masked_images: torch.Tensor
) -> ADMMEstimator:
    params = config.params

    real_irls_config = replace(config, max_iter=params["internal_max_iter"])
    real_irls = _initialize_irls(real_irls_config, unmasked_images, masked_images)

    fourier_params = {
        "delta": params["fourier_delta"],
        "weight_approach": params["fourier_weight_approach"],
        "lowpass_mask": params["fourier_lowpass_mask"],
        "lowpass_mask_cutoff": params["fourier_lowpass_mask_cutoff"],
    }
    fourier_irls_config = replace(
        config, max_iter=params["internal_max_iter"], params=fourier_params
    )
    fourier_irls = _initialize_fourier_irls(
        fourier_irls_config, unmasked_images, masked_images
    )

    return ADMMEstimator(
        irls_real=real_irls,
        irls_fourier=fourier_irls,
        max_iter=config.max_iter,
        initial_mu=params["initial_mu"],
        fourier_multiplier=params["fourier_multiplier"],
    )


ESTIMATOR_INITIALIZERS = {
    "irls": _initialize_irls,
    "fourier_irls": _initialize_fourier_irls,
    "admm": _initialize_admm,
}


def initialize_estimator(
    pipeline_config: PipelineConfig,
    unmasked_images: torch.Tensor,
    masked_images: torch.Tensor,
):
    gmm_config = pipeline_config.gmm
    estimator_config = pipeline_config.estimator
    estimator_type = estimator_config.estimator_type

    estimator = ESTIMATOR_INITIALIZERS[estimator_type](
        estimator_config, unmasked_images, masked_images
    )

    if gmm_config is None:
        return estimator

    def distance_function(images, reference):
        weights = estimator.fit(images, reference=reference).weights
        # Weights are aggregated per-image and have shape (n, 1, ..., 1)
        return -weights.reshape(-1)

    return RecursiveGMMEstimator(
        distance_function=distance_function,
        max_iter=gmm_config.external_max_iter,
        tol=estimator_config.tolerance,
        initial_bad_mean_quantile=gmm_config.initial_bad_quantile,
        initial_bad_weight=gmm_config.initial_bad_weight,
        random_state=estimator_config.random_state,
        gmm_max_iter=gmm_config.internal_max_iter,
        check_degenerate_model=gmm_config.check_degenerate,
        min_component_separation=gmm_config.min_component_separation,
        min_good_component_weight=gmm_config.min_good_component_weight,
    )


def process_class(
    class_data: pd.DataFrame,
    pipeline_config: PipelineConfig,
) -> ClassProcessingResults:
    """
    Estimate robust and conventional averages for one particle class.

    Parameters
    ----------
    class_data : pandas.DataFrame
        Metadata describing the preprocessed particles for the requested class.
    pipeline_config : PipelineConfig
        Pipeline configuration object

    Returns
    -------
    ClassProcessingResults
        Results object (see ``ClassProcessingResults`` for details)
    """
    images = read_images(data=class_data, device=pipeline_config.device)

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
        pipeline_config, unmasked_images=images, masked_images=masked_images
    )
    reference = masked_images.mean(dim=0)

    result = estimator.fit(images=masked_images, reference=reference)
    fit_weights_np = result.weights.detach().cpu().numpy().reshape(-1)

    gmm_distances = result.gmm_distances
    if gmm_distances is None:
        robust_weights_np = fit_weights_np
        gmm_weights_np = None
    else:
        robust_weights_np = -gmm_distances.detach().cpu().numpy().reshape(-1)
        gmm_weights_np = fit_weights_np

    io = pipeline_config.io

    unmasked_corrected_average = (
        weighted_average(images, result.weights).detach().cpu().numpy()
    )
    unmasked_original_average = images.mean(dim=0).detach().cpu().numpy()

    weights = fit_weights_np

    high_weight_average = None
    if io.out_high_weight_avgs:
        outlier_proportion = io.outlier_proportion_for_avgs
        top_weight = np.quantile(weights, 1.0 - outlier_proportion)
        high_mask = weights >= top_weight

        if high_mask.any():
            high_weight_average = images[high_mask].mean(dim=0).detach().cpu().numpy()

    low_weight_average = None
    if io.out_low_weight_avgs:
        outlier_proportion = io.outlier_proportion_for_avgs
        bottom_weight = np.quantile(weights, outlier_proportion)
        low_mask = weights <= bottom_weight

        if low_mask.any():
            low_weight_average = images[low_mask].mean(dim=0).detach().cpu().numpy()

    return ClassProcessingResults(
        unmasked_corrected_average=unmasked_corrected_average,
        unmasked_original_average=unmasked_original_average,
        unmasked_low_weight_average=low_weight_average,
        unmasked_high_weight_average=high_weight_average,
        robust_weights=robust_weights_np,
        gmm_weights=gmm_weights_np,
    )


def write_weights_to_dataframe(
    write_metadata: pd.DataFrame,
    robust_weights_np: np.ndarray,
    gmm_weights_np: np.ndarray | None,
    item_ids: np.ndarray,
    group_by_column: str,
    group_by_value: int,
    eps_std: float = 1.0e-6,
) -> None:
    """Store each particle's group and all weights from its best assignment."""
    std = robust_weights_np.std()
    standardized_weights = (
        (robust_weights_np - robust_weights_np.mean()) / std
        if std > eps_std
        else np.zeros_like(robust_weights_np)
    )

    weights_by_column = {
        ROBUST_WEIGHT_COL: robust_weights_np,
        STD_ROBUST_WEIGHT_COL: standardized_weights,
    }
    if gmm_weights_np is not None:
        weights_by_column[GMM_WEIGHT_COL] = gmm_weights_np

    for column, values in weights_by_column.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Group {group_by_value} produced nonfinite {column} values."
            )

    primary_column = GMM_WEIGHT_COL if gmm_weights_np is not None else ROBUST_WEIGHT_COL
    output_ids = write_metadata[MDL_ITEM_ID_COLUMN]
    matched_ids = output_ids.loc[output_ids.isin(item_ids)]

    incoming = pd.DataFrame(weights_by_column, index=item_ids).reindex(
        matched_ids.to_numpy()
    )
    incoming.index = matched_ids.index

    previous_primary = write_metadata.loc[matched_ids.index, primary_column]
    better_group = previous_primary.isna() | (
        incoming[primary_column] > previous_primary
    )
    winners = incoming.loc[better_group]

    write_metadata.loc[winners.index, list(weights_by_column)] = winners
    write_metadata.loc[winners.index, group_by_column] = group_by_value


def get_output_buffers(
    io_config: IOConfig,
    input_metadata_df: pd.DataFrame,
    weight_columns: Iterable[str],
    n_classes: int,
) -> tuple[
    pd.DataFrame | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    group_by_column = io_config.group_by_column

    write_metadata = None
    if io_config.out_star:
        # Metadata files can be large, avoid reading the same file twice if possible
        if io_config.base_xmd != io_config.input_xmd:
            write_metadata = pd.DataFrame(starfile.read(io_config.base_xmd))
        else:
            write_metadata = input_metadata_df.copy()

        validate_item_ids(write_metadata, name="Base")

        input_ids = pd.Index(input_metadata_df[MDL_ITEM_ID_COLUMN].unique())
        base_ids = pd.Index(write_metadata[MDL_ITEM_ID_COLUMN])

        missing_ids = input_ids.difference(base_ids)
        if len(missing_ids):
            raise ValueError(
                f"{len(missing_ids)} input particles are absent from the base metadata."
            )

        # Filter out particles not present in input xmd
        write_metadata = write_metadata.loc[
            write_metadata[MDL_ITEM_ID_COLUMN].isin(input_ids)
        ].copy()

        # Initialize the weight to nan to later catch particles without weights
        for column in weight_columns:
            write_metadata[column] = np.nan
        write_metadata[group_by_column] = UNASSIGNED_GROUP_VALUE

    stack_path = Path(
        str(input_metadata_df["image"].to_numpy()[0]).split("@", maxsplit=1)[1]
    )

    with mrcfile.open(stack_path, header_only=True) as mrc:
        nx = mrc.header.nx
        ny = mrc.header.ny

    average_stacks_shape = (n_classes, ny, nx)
    corrected_averages = None
    if io_config.out_corrected_avgs:
        corrected_averages = np.empty(shape=average_stacks_shape, dtype=np.float32)

    original_averages = None
    if io_config.out_original_avgs:
        original_averages = np.empty(shape=average_stacks_shape, dtype=np.float32)

    low_weight_averages = None
    if io_config.out_low_weight_avgs:
        low_weight_averages = np.empty(shape=average_stacks_shape, dtype=np.float32)

    high_weight_averages = None
    if io_config.out_high_weight_avgs:
        high_weight_averages = np.empty(shape=average_stacks_shape, dtype=np.float32)

    return (
        write_metadata,
        corrected_averages,
        original_averages,
        low_weight_averages,
        high_weight_averages,
    )


def get_weight_columns(pipeline_config: PipelineConfig):
    base_weight_cols = [ROBUST_WEIGHT_COL, STD_ROBUST_WEIGHT_COL]
    if pipeline_config.gmm is None:
        return base_weight_cols
    return base_weight_cols + [GMM_WEIGHT_COL]


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    pipeline_config = parse_pipeline_config(args)

    group_by_column = pipeline_config.io.group_by_column
    input_metadata_df = pd.DataFrame(starfile.read(pipeline_config.io.input_xmd))

    validate_assignments(input_metadata_df, group_by_column)

    input_metadata_df[group_by_column] = pd.to_numeric(
        input_metadata_df[group_by_column], errors="raise"
    ).astype(np.int64)

    groups = input_metadata_df.groupby(group_by_column, sort=True)

    weight_columns = get_weight_columns(pipeline_config)
    out_md, robust_avgs, original_avgs, low_w_avgs, high_w_avgs = get_output_buffers(
        io_config=pipeline_config.io,
        input_metadata_df=input_metadata_df,
        weight_columns=weight_columns,
        n_classes=groups.ngroups,
    )

    for index, (class_value, class_data) in enumerate(groups):
        class_results = process_class(
            class_data=class_data, pipeline_config=pipeline_config
        )

        if robust_avgs is not None:
            robust_avgs[index] = class_results.unmasked_corrected_average

        if original_avgs is not None:
            original_avgs[index] = class_results.unmasked_original_average

        if low_w_avgs is not None:
            low_w_avgs[index] = (
                class_results.unmasked_low_weight_average
                if class_results.unmasked_low_weight_average is not None
                else 0.0
            )

        if high_w_avgs is not None:
            high_w_avgs[index] = (
                class_results.unmasked_high_weight_average
                if class_results.unmasked_high_weight_average is not None
                else 0.0
            )

        if out_md is not None:
            write_weights_to_dataframe(
                group_by_column=group_by_column,
                group_by_value=class_value,
                write_metadata=out_md,
                item_ids=class_data[MDL_ITEM_ID_COLUMN].to_numpy(),
                robust_weights_np=class_results.robust_weights,
                gmm_weights_np=class_results.gmm_weights,
            )

    if robust_avgs is not None:
        mrcfile.write(
            name=pipeline_config.io.out_corrected_avgs,
            data=robust_avgs,
            overwrite=pipeline_config.io.overwrite,
        )
    if original_avgs is not None:
        mrcfile.write(
            name=pipeline_config.io.out_original_avgs,
            data=original_avgs,
            overwrite=pipeline_config.io.overwrite,
        )
    if low_w_avgs is not None:
        mrcfile.write(
            name=pipeline_config.io.out_low_weight_avgs,
            data=low_w_avgs,
            overwrite=pipeline_config.io.overwrite,
        )
    if high_w_avgs is not None:
        mrcfile.write(
            name=pipeline_config.io.out_high_weight_avgs,
            data=high_w_avgs,
            overwrite=pipeline_config.io.overwrite,
        )

    if out_md is not None:
        if out_md[weight_columns].isna().any().any():
            raise RuntimeError("Some particles were not assigned weights.")

        output_groups = out_md[group_by_column]
        if output_groups.isna().any() or (output_groups < 0).any():
            raise RuntimeError("Some particles were not assigned a valid group.")

        out_md[group_by_column] = output_groups.astype(np.int64)

        starfile.write(data=out_md, filename=args.out_star)


if __name__ == "__main__":
    main()
