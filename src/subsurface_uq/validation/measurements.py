from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sampling.calibration import (
    SUPPORTED_CALIBRATION_MODELS,
    calibrate_point_covariance_candidates,
)
from ..sampling.exact_kriging import exact_simple_kriging_predict_points

Array = np.ndarray


@dataclass(frozen=True)
class SpatialMeasurementCVResult:
    summary_rows: tuple[dict[str, object], ...]
    fold_rows: tuple[dict[str, object], ...]
    fold_assignment: Array


def spatial_block_fold_assignment(
    coordinates_xy_m: Array,
    *,
    n_folds: int = 5,
    block_size_m: float = 2000.0,
    seed: int = 2907,
) -> Array:
    """Assign all points in a rectangular spatial block to the same CV fold."""

    coordinates = np.asarray(coordinates_xy_m, dtype=np.float64)
    n_folds = int(n_folds)
    block_size_m = float(block_size_m)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates_xy_m must have shape [n,2] with n>=2")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if block_size_m <= 0.0:
        raise ValueError("block_size_m must be positive")

    origin = np.min(coordinates, axis=0)
    block_x = np.floor((coordinates[:, 0] - origin[0]) / block_size_m).astype(np.int64)
    block_y = np.floor((coordinates[:, 1] - origin[1]) / block_size_m).astype(np.int64)
    block_keys = list(dict.fromkeys(zip(block_x.tolist(), block_y.tolist())))
    if len(block_keys) < n_folds:
        raise ValueError(
            f"only {len(block_keys)} occupied spatial blocks for {n_folds} folds; "
            "reduce n_folds or block_size_m"
        )
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(block_keys))

    # Greedy balanced assignment by point count keeps entire blocks together
    # while avoiding one fold accidentally receiving most measurements.
    block_counts = {key: int(np.count_nonzero((block_x == key[0]) & (block_y == key[1]))) for key in block_keys}
    ordered = sorted(
        (block_keys[index] for index in permutation),
        key=lambda key: block_counts[key],
        reverse=True,
    )
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    block_to_fold: dict[tuple[int, int], int] = {}
    for key in ordered:
        fold = int(np.argmin(fold_sizes))
        block_to_fold[key] = fold
        fold_sizes[fold] += block_counts[key]

    assignment = np.asarray(
        [block_to_fold[(int(bx), int(by))] for bx, by in zip(block_x, block_y)],
        dtype=np.int64,
    )
    if set(assignment.tolist()) != set(range(n_folds)):
        raise RuntimeError("spatial block assignment produced an empty fold")
    return assignment


def _predictive_metrics(truth: Array, mean: Array, std: Array) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    residual = truth - mean
    variance = std**2
    positive = variance > np.finfo(float).eps
    if not np.all(positive):
        raise ValueError("held-out predictive variance must be strictly positive")
    z = residual / std
    z90 = 1.6448536269514722
    coverage = np.mean(np.abs(z) <= z90)
    nlpd = 0.5 * np.log(2.0 * np.pi * variance) + 0.5 * residual**2 / variance
    return {
        "rmse_log10_k": float(np.sqrt(np.mean(residual**2))),
        "mae_log10_k": float(np.mean(np.abs(residual))),
        "coverage_90": float(coverage),
        "standardized_residual_mean": float(np.mean(z)),
        "standardized_residual_std": float(np.std(z, ddof=1)) if z.size > 1 else 0.0,
        "gaussian_nlpd": float(np.mean(nlpd)),
    }


def spatial_block_cross_validate_measurements(
    coordinates_xy_m: Array,
    hydraulic_conductivity_m_s: Array,
    *,
    models=SUPPORTED_CALIBRATION_MODELS,
    n_folds: int = 5,
    block_size_m: float = 2000.0,
    fold_seed: int = 2907,
    lag_bin_m: float = 250.0,
    max_lag_m: float = 3000.0,
    angle_tolerance_deg: float = 22.5,
    min_pairs_per_bin: int = 8,
    observation_std_log10_k: float = 0.0,
    fit_nugget: bool = True,
    pair_count_weighted_fit: bool = True,
) -> SpatialMeasurementCVResult:
    """Spatial block CV with fold-wise variogram recalibration and exact kriging."""

    coordinates = np.asarray(coordinates_xy_m, dtype=np.float64)
    conductivity = np.asarray(hydraulic_conductivity_m_s, dtype=np.float64).reshape(-1)
    if coordinates.shape != (conductivity.size, 2):
        raise ValueError("coordinates and conductivity values must align")
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("hydraulic conductivity values must be finite and positive")
    log_values = np.log10(conductivity)
    assignment = spatial_block_fold_assignment(
        coordinates, n_folds=n_folds, block_size_m=block_size_m, seed=fold_seed
    )

    requested = tuple(str(model).strip().lower() for model in models)
    predictions: dict[str, list[tuple[Array, Array, Array]]] = {model: [] for model in requested}
    fold_rows: list[dict[str, object]] = []
    for fold in range(int(n_folds)):
        test = assignment == fold
        train = ~test
        if np.count_nonzero(train) < 10 or np.count_nonzero(test) == 0:
            raise ValueError(f"fold {fold} has insufficient train/test measurements")
        calibration, _, _ = calibrate_point_covariance_candidates(
            coordinates[train],
            conductivity[train],
            models=requested,
            lag_bin_m=lag_bin_m,
            max_lag_m=max_lag_m,
            angle_tolerance_deg=angle_tolerance_deg,
            min_pairs_per_bin=min_pairs_per_bin,
            fit_nugget=fit_nugget,
            pair_count_weighted_fit=pair_count_weighted_fit,
        )
        by_model = {result.covariance_model: result for result in calibration}
        for model in requested:
            result = by_model[model]
            posterior = exact_simple_kriging_predict_points(
                observation_coordinates_xy_m=coordinates[train],
                observation_log10_k=log_values[train],
                query_coordinates_xy_m=coordinates[test],
                covariance_model=model,
                mean_log10_k=result.mean_log10_k,
                std_log10_k=result.std_log10_k,
                length_scale_y_m=result.length_scale_y_m,
                length_scale_x_m=result.length_scale_x_m,
                observation_std_log10_k=observation_std_log10_k,
                structured_std_log10_k=(
                    result.structured_std_log10_k
                    if result.structured_std_log10_k is not None
                    else result.std_log10_k
                ),
                nugget_std_log10_k=result.nugget_std_log10_k,
                include_query_nugget=True,
                angle_rad=result.angle_rad,
            )
            metrics = _predictive_metrics(
                log_values[test], posterior.mean_log10_k, posterior.std_log10_k
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "covariance_model": model,
                    "n_train": int(np.count_nonzero(train)),
                    "n_test": int(np.count_nonzero(test)),
                    "length_scale_x_m": result.length_scale_x_m,
                    "length_scale_y_m": result.length_scale_y_m,
                    "ell_x_over_ell_y": result.length_scale_x_m / result.length_scale_y_m,
                    "variogram_rmse": result.variogram_rmse,
                    "variogram_weighted_rmse": result.variogram_weighted_rmse,
                    "structured_std_log10_k": result.structured_std_log10_k,
                    "nugget_std_log10_k": result.nugget_std_log10_k,
                    "nugget_fraction": result.nugget_fraction,
                    **metrics,
                }
            )
            predictions[model].append(
                (
                    log_values[test].copy(),
                    posterior.mean_log10_k.copy(),
                    posterior.std_log10_k.copy(),
                )
            )

    summary_rows: list[dict[str, object]] = []
    for model in requested:
        truth = np.concatenate([item[0] for item in predictions[model]])
        mean = np.concatenate([item[1] for item in predictions[model]])
        std = np.concatenate([item[2] for item in predictions[model]])
        metrics = _predictive_metrics(truth, mean, std)
        model_folds = [row for row in fold_rows if row["covariance_model"] == model]
        summary_rows.append(
            {
                "covariance_model": model,
                "n_predictions": int(truth.size),
                "mean_fold_variogram_rmse": float(
                    np.mean([row["variogram_rmse"] for row in model_folds])
                ),
                "mean_fold_weighted_variogram_rmse": float(
                    np.mean([row["variogram_weighted_rmse"] for row in model_folds])
                ),
                "mean_fold_length_scale_x_m": float(
                    np.mean([row["length_scale_x_m"] for row in model_folds])
                ),
                "mean_fold_length_scale_y_m": float(
                    np.mean([row["length_scale_y_m"] for row in model_folds])
                ),
                "mean_fold_nugget_fraction": float(
                    np.mean([row["nugget_fraction"] for row in model_folds])
                ),
                "std_fold_nugget_fraction": float(
                    np.std([row["nugget_fraction"] for row in model_folds], ddof=1)
                    if len(model_folds) > 1
                    else 0.0
                ),
                **metrics,
            }
        )
    return SpatialMeasurementCVResult(
        summary_rows=tuple(summary_rows),
        fold_rows=tuple(fold_rows),
        fold_assignment=assignment,
    )
