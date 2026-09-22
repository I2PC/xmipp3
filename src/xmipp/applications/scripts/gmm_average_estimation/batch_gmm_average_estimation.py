#!/usr/bin/env python3

import argparse
from pathlib import Path
import warnings
from typing import Tuple, Union, Optional, Dict, Literal, get_args, Any, Iterable
from dataclasses import dataclass, field, replace
from functools import partial

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

# Import estimator types
from xmippPyModules.gmmAverageTools.gmm_estimator import RecursiveGMMEstimator
from xmippPyModules.gmmAverageTools.irls_estimator import IRLSMEstimator
from xmippPyModules.gmmAverageTools.fourier_irls_estimator import (
    JointIRLSFourier,
    WeightApproach,
)
from xmippPyModules.gmmAverageTools.admm_estimator import ADMMEstimator

# Import weight and distance functions
from xmippPyModules.gmmAverageTools.weights import (
    calculate_beta_auto,
    tagare_weight_precomputed,
    smooth_redescending_weights_modulus,
    smooth_redescending_weights_norm,
)

# Utilities: masks, weighted averages
from xmippPyModules.gmmAverageTools.masks import (
    create_circular_mask,
    create_lowpass_rfft_mask,
)
from xmippPyModules.gmmAverageTools.utils import weighted_average

Estimator = Union[
    RecursiveGMMEstimator, ADMMEstimator, JointIRLSFourier, IRLSMEstimator
]
EstimatorType = Literal["irls", "fourier_irls", "admm"]
ESTIMATOR_TYPES: Tuple[str, ...] = get_args(EstimatorType)

UNASSIGNED_GROUP_VALUE = -1

DEFAULT_SMOOTH_DELTA: Dict[WeightApproach, float] = {
    "per-image": 0.05,
    "per-coefficient": 1.5,
}

DEFAULT_MAX_ITERATIONS: Dict[EstimatorType, int] = {
    "irls": 50,
    "fourier_irls": 50,
    "admm": 30,
}

DEFAULT_LOWPASS_MASK_CUTOFF = 0.25

ROBUST_WEIGHT_COL = "wRobust"
STD_ROBUST_WEIGHT_COL = "wRobustStd"
GMM_WEIGHT_COL = "wRobustGmm"


@dataclass
class IOConfig:
    input_xmd: Path
    base_xmd: Path
    out_star: Optional[Path] = None
    out_corrected_avgs: Optional[Path] = None
    out_original_avgs: Optional[Path] = None
    group_by_column: str = MDL_REF_COLUMN


@dataclass
class GMMConfig:
    external_max_iter: int = 15
    internal_max_iter: int = 25
    standardize_distances: bool = True
    check_degenerate: bool = True
    min_component_separation: float = 0.5
    min_good_component_weight: float = 0.4


@dataclass
class EstimatorConfig:
    estimator_type: EstimatorType
    max_iter: int
    tolerance: float
    random_state: int
    damping_coef: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    device: str
    io: IOConfig
    estimator: EstimatorConfig
    gmm: Optional[GMMConfig] = None


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
        "--group-by-column",
        type=str,
        default=MDL_REF_COLUMN,
        help=f"Column by which images are grouped (default: '{MDL_REF_COLUMN}')",
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
        "--gmm-standardize-distances",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Standardize the distances before fitting the GMM model to them.",
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
    irls_parser = subparsers.add_parser("irls", help="Standard IRLS procedure")

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
        help="Normalized lowpass mask cutoff frequency",
    )
    fourier_irls_parser.add_argument(
        "--lowpass-mask-cutoff",
        type=float,
        default=DEFAULT_LOWPASS_MASK_CUTOFF,
        help="Normalized lowpass mask cutoff frequency",
    )

    return parser


def parse_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    io_cfg = IOConfig(
        input_xmd=args.input_xmd,
        base_xmd=args.base_xmd or args.input_xmd,
        out_star=args.out_star,
        out_corrected_avgs=args.out_corrected_avgs,
        out_original_avgs=args.out_original_avgs,
        group_by_column=args.group_by_column,
    )

    if args.estimator_type == "fourier_irls":
        method_params = {
            "delta": args.delta,
            "weight_approach": args.weight_approach,
            "lowpass_mask_cutoff": args.lowpass_mask_cutoff,
        }
    elif args.estimator_type == "admm":
        method_params = {
            "initial_mu": args.initial_mu,
            "fourier_multiplier": args.fourier_multiplier,
            "internal_max_iter": args.internal_max_iter,
        }
    else:
        method_params = {}

    # Resolve dynamic default for 'delta'
    if method_params.get("delta") is None:
        approach = method_params.get("weight_approach")
        if approach in DEFAULT_SMOOTH_DELTA:
            method_params["delta"] = DEFAULT_SMOOTH_DELTA[approach]

    # Resolve dynamic defaeult for max_iter
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
        if estimator_cfg.estimator_type == "admm":
            raise ValueError("GMM reweighting is not supported for ADMM estimator")

        gmm_cfg = GMMConfig(
            external_max_iter=args.gmm_external_max_iter,
            internal_max_iter=args.gmm_internal_max_iter,
            standardize_distances=args.gmm_standardize_distances,
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
    """Validate that a metadata table contains unique particle item identifiers."""
    if MDL_ITEM_ID_COLUMN not in data.columns:
        raise ValueError(
            f"{name} metadata does not contain " f"'{MDL_ITEM_ID_COLUMN}'."
        )

    if data[MDL_ITEM_ID_COLUMN].duplicated().any():
        raise ValueError(
            f"{name} metadata contains duplicated " f"'{MDL_ITEM_ID_COLUMN}' values."
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

    internal_config = replace(config, max_iter=params["internal_max_iter"])

    real_irls = _initialize_irls(internal_config, unmasked_images, masked_images)

    fourier_irls = _initialize_fourier_irls(
        internal_config, unmasked_images, masked_images
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
        _, weights = estimator.fit(images, reference=reference)
        return -weights.flatten(start_dim=1).mean(dim=1)

    return RecursiveGMMEstimator(
        distance_function=distance_function,
        max_iter=gmm_config.external_max_iter,
        tol=estimator_config.tolerance,
        standardize_distances=gmm_config.standardize_distances,
        random_state=estimator_config.random_state,
        gmm_max_iter=gmm_config.internal_max_iter,
        check_degenerate_model=gmm_config.check_degenerate,
        min_component_separation=gmm_config.min_component_separation,
        min_good_component_weight=gmm_config.min_good_component_weight,
    )


def fit_estimator(
    estimator: Estimator,
    masked_images: torch.Tensor,
    unmasked_images: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Fit the estimator and return a set of per-image weights and an optional set of
    per-image GMM weights for GMM-type estimators.
    Also returns an estimate using the unmasked images and the estimator's weights.
    """
    n_images = unmasked_images.shape[0]

    gmm_weights = None
    if isinstance(estimator, RecursiveGMMEstimator):
        _, weights, original_distances = estimator.fit(
            images=masked_images, reference=reference
        )

        unmasked_new_average = weighted_average(unmasked_images, weights)

        robust_weights = -original_distances
        gmm_weights = weights

    elif isinstance(estimator, ADMMEstimator):
        _, weights_real, weights_fourier = estimator.fit(images=masked_images)

        estimate_real = weighted_average(unmasked_images, weights_real)
        images_fourier = torch.fft.rfft2(unmasked_images)
        estimate_fourier = torch.fft.irfft2(
            weighted_average(images_fourier, weights_fourier)
        )
        unmasked_new_average = 0.5 * (estimate_real + estimate_fourier)

        weights_real = weights_real.view(n_images, -1).mean(dim=1)
        weights_fourier = weights_fourier.view(n_images, -1).mean(dim=1)

        robust_weights = 0.5 * (weights_real + weights_fourier)

    elif isinstance(estimator, JointIRLSFourier):
        _, weights = estimator.fit(
            images=masked_images, reference=reference, fourier_transform_images=True
        )

        unmasked_images_fourier = torch.fft.rfft2(unmasked_images)
        fourier_unmasked_average = weighted_average(unmasked_images_fourier, weights)
        unmasked_new_average = torch.fft.irfft2(fourier_unmasked_average)

        robust_weights = weights.view(n_images, -1).mean(dim=1)

    else:
        _, weights = estimator.fit(images=masked_images, reference=reference)

        unmasked_new_average = weighted_average(unmasked_images, weights)

        # Aggregate possibly local weights into global per-image scores
        robust_weights = weights.view(n_images, -1).mean(dim=1)

    return (
        robust_weights.detach().cpu().numpy().reshape(-1),
        None if gmm_weights is None else gmm_weights.detach().cpu().numpy().reshape(-1),
        unmasked_new_average.detach().cpu().numpy(),
    )


def process_class(
    data: pd.DataFrame,
    pipeline_config: PipelineConfig,
    group_by_value: int,
    write_metadata: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate robust and conventional averages for one particle class.

    Parameters
    ----------
    data : pandas.DataFrame
        Metadata describing the preprocessed particles.
    pipeline_config : PipelineConfig
        Pipeline configuration object
    group_by_value : int
        Identifier of the class to process.
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
    group_by_column = pipeline_config.io.group_by_column
    class_data = data[data[group_by_column] == group_by_value]
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

    robust_weights_np, gmm_weights_np, unmasked_new_average = fit_estimator(
        estimator,
        masked_images=masked_images,
        unmasked_images=images,
        reference=reference,
    )
    unmasked_original_average = images.mean(dim=0).detach().cpu().numpy()

    if write_metadata is not None:
        write_weights_to_dataframe(
            group_by_column=group_by_column,
            group_by_value=group_by_value,
            write_metadata=write_metadata,
            item_ids=class_data[MDL_ITEM_ID_COLUMN].to_numpy(),
            robust_weights_np=robust_weights_np,
            gmm_weights_np=gmm_weights_np,
        )

    return unmasked_new_average, unmasked_original_average


def write_weights_to_dataframe(
    write_metadata: pd.DataFrame,
    robust_weights_np: np.ndarray,
    gmm_weights_np: Optional[np.ndarray],
    item_ids: np.ndarray,
    group_by_column: str,
    group_by_value: int,
):
    """
    Write weights to metadata file, ensuring the association is correct by using
    the item ids from the input metadata file
    """
    class_mask = write_metadata[MDL_ITEM_ID_COLUMN].isin(item_ids)
    target_item_ids = write_metadata.loc[class_mask, MDL_ITEM_ID_COLUMN]

    robust_weights_by_id = pd.Series(robust_weights_np, index=item_ids)

    write_metadata.loc[class_mask, ROBUST_WEIGHT_COL] = target_item_ids.map(
        robust_weights_by_id
    ).to_numpy()

    # Per-class standardized weights are useful because different classes might have
    # different weight distributions
    robust_weights_std = (
        robust_weights_np - robust_weights_np.mean()
    ) / robust_weights_np.std()
    weights_std_by_id = pd.Series(robust_weights_std, index=item_ids)

    write_metadata.loc[class_mask, STD_ROBUST_WEIGHT_COL] = target_item_ids.map(
        weights_std_by_id
    ).to_numpy()

    if gmm_weights_np is not None:
        gmm_weights_by_id = pd.Series(gmm_weights_np, index=item_ids)

        write_metadata.loc[class_mask, GMM_WEIGHT_COL] = target_item_ids.map(
            gmm_weights_by_id
        ).to_numpy()

    write_metadata.loc[class_mask, group_by_column] = group_by_value


def get_output_buffers(
    io_config: IOConfig,
    input_metadata_df: pd.DataFrame,
    weight_columns: Iterable[str],
    n_classes: int,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
    group_by_column = io_config.group_by_column

    write_metadata = None
    if io_config.out_star:
        # Metadata files can be large, avoid reading the same file twice if possible
        if io_config.base_xmd != io_config.input_xmd:
            write_metadata = pd.DataFrame(starfile.read(io_config.base_xmd))
        else:
            write_metadata = input_metadata_df

        # Initialize the weight and group columns with invalid values to help catch
        # any particles that don't get assigned a group or weights
        for column in weight_columns:
            write_metadata[column] = np.nan

        write_metadata[group_by_column] = pd.Series(
            UNASSIGNED_GROUP_VALUE, index=write_metadata.index, dtype=int
        )

    stack_path = Path(
        str(input_metadata_df["image"].to_numpy()[0]).split("@", maxsplit=1)[1]
    )

    with mrcfile.open(stack_path, header_only=True) as mrc:
        nx = mrc.header.nx
        ny = mrc.header.ny

    corrected_averages = None
    if io_config.out_corrected_avgs:
        corrected_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    original_averages = None
    if io_config.out_original_avgs:
        original_averages = np.empty(shape=(n_classes, ny, nx), dtype=np.float32)

    return write_metadata, corrected_averages, original_averages


def get_weight_columns(pipeline_config: PipelineConfig):
    base_weight_cols = [ROBUST_WEIGHT_COL, STD_ROBUST_WEIGHT_COL]
    if pipeline_config.gmm is None:
        return base_weight_cols
    return base_weight_cols + [GMM_WEIGHT_COL]


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    pipeline_config = parse_pipeline_config(args)

    input_metadata_df = pd.DataFrame(starfile.read(args.input_xmd))
    validate_item_ids(input_metadata_df, name="Input")

    group_by_values = sorted(
        input_metadata_df[pipeline_config.io.group_by_column].unique()
    )

    weight_columns = get_weight_columns(pipeline_config)
    write_metadata, corrected_averages, original_averages = get_output_buffers(
        io_config=pipeline_config.io,
        input_metadata_df=input_metadata_df,
        weight_columns=weight_columns,
        n_classes=len(group_by_values),
    )

    for index, class_value in enumerate(group_by_values):
        corrected_avg, original_avg = process_class(
            data=input_metadata_df,
            pipeline_config=pipeline_config,
            group_by_value=class_value,
            write_metadata=write_metadata,
        )

        if corrected_averages is not None:
            corrected_averages[index] = corrected_avg

        if original_averages is not None:
            original_averages[index] = original_avg

    if corrected_averages is not None:
        mrcfile.write(
            name=pipeline_config.io.out_corrected_avgs, data=corrected_averages
        )
    if original_averages is not None:
        mrcfile.write(name=pipeline_config.io.out_original_avgs, data=original_averages)
    if write_metadata is not None:
        if write_metadata[weight_columns].isna().any().any():
            raise RuntimeError("Some particles were not assigned weights.")

        starfile.write(data=write_metadata, filename=args.out_star)


if __name__ == "__main__":
    main()
