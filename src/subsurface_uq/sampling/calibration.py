from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

Array = np.ndarray

SUPPORTED_CALIBRATION_MODELS = (
    "matern32",
    "exponential",
    "radial_exponential",
)


@dataclass(frozen=True)
class CovarianceCalibrationResult:
    """Compact calibration result for log10-permeability covariance models."""

    covariance_model: str
    mean_log10_k: float
    std_log10_k: float
    between_field_mean_std_log10_k: float
    length_scale_y_m: float
    length_scale_x_m: float
    angle_rad: float
    variogram_rmse: float
    variogram_rmse_y: float
    variogram_rmse_x: float
    variogram_rmse_diag: float
    cell_size_m: float | None
    spatial_stride: int | None
    max_lag_cells: int | None
    calibration_source: str = "regular_grid"
    lag_bin_m: float | None = None
    max_lag_m: float | None = None
    angle_tolerance_deg: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "covariance_model": self.covariance_model,
            "mean_log10_k": self.mean_log10_k,
            "std_log10_k": self.std_log10_k,
            "between_field_mean_std_log10_k": self.between_field_mean_std_log10_k,
            "length_scale_y_m": self.length_scale_y_m,
            "length_scale_x_m": self.length_scale_x_m,
            "angle_rad": self.angle_rad,
            "variogram_rmse": self.variogram_rmse,
            "variogram_rmse_y": self.variogram_rmse_y,
            "variogram_rmse_x": self.variogram_rmse_x,
            "variogram_rmse_diag": self.variogram_rmse_diag,
            "cell_size_m": self.cell_size_m,
            "spatial_stride": self.spatial_stride,
            "max_lag_cells": self.max_lag_cells,
            "calibration_source": self.calibration_source,
            "lag_bin_m": self.lag_bin_m,
            "max_lag_m": self.max_lag_m,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "log_space": "log10",
        }


def _as_positive_fields(fields: Array) -> Array:
    values = np.asarray(fields)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("permeability fields must have shape [H,W] or [N,H,W]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("permeability fields must be finite and strictly positive")
    return values


def estimate_directional_variograms(
    fields: Array,
    *,
    cell_size_m: float,
    max_lag_cells: int = 64,
    spatial_stride: int = 1,
) -> dict[str, tuple[Array, Array]]:
    """Estimate axis and diagonal semivariograms of log10(K).

    The calculation uses regular-grid increments rather than all point pairs, so
    it remains practical for large DaRUS-style fields. spatial_stride may be
    increased for exploratory calibration on very large grids; distances are
    adjusted accordingly.
    """

    values = _as_positive_fields(fields)
    cell_size_m = float(cell_size_m)
    max_lag_cells = int(max_lag_cells)
    spatial_stride = int(spatial_stride)
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    if max_lag_cells <= 0:
        raise ValueError("max_lag_cells must be positive")
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")

    log_fields = np.log10(values)
    sampled = log_fields[:, ::spatial_stride, ::spatial_stride]
    h, w = sampled.shape[1:]
    maximum = min(max_lag_cells, h - 1, w - 1)
    if maximum < 1:
        raise ValueError("field is too small for the requested variogram settings")

    spacing = cell_size_m * spatial_stride
    lags = np.arange(1, maximum + 1, dtype=np.float64)
    gamma_y = np.empty(maximum, dtype=np.float64)
    gamma_x = np.empty(maximum, dtype=np.float64)
    gamma_diag = np.empty(maximum, dtype=np.float64)

    for index, lag in enumerate(range(1, maximum + 1)):
        dy = sampled[:, lag:, :] - sampled[:, :-lag, :]
        dx = sampled[:, :, lag:] - sampled[:, :, :-lag]
        dd = sampled[:, lag:, lag:] - sampled[:, :-lag, :-lag]
        gamma_y[index] = 0.5 * float(np.mean(np.asarray(dy, dtype=np.float64) ** 2))
        gamma_x[index] = 0.5 * float(np.mean(np.asarray(dx, dtype=np.float64) ** 2))
        gamma_diag[index] = 0.5 * float(np.mean(np.asarray(dd, dtype=np.float64) ** 2))

    return {
        "y": (lags * spacing, gamma_y),
        "x": (lags * spacing, gamma_x),
        "diag": (lags * spacing * np.sqrt(2.0), gamma_diag),
    }


def estimate_point_directional_variograms(
    coordinates_xy_m: Array,
    log10_values: Array,
    *,
    lag_bin_m: float = 250.0,
    max_lag_m: float = 3000.0,
    angle_tolerance_deg: float = 22.5,
    min_pairs_per_bin: int = 8,
) -> tuple[dict[str, tuple[Array, Array]], dict[str, Array]]:
    """Estimate directional semivariograms from irregular point measurements.

    Directions are defined by the absolute pair-vector angle: x around 0/180 deg,
    y around 90 deg, and diagonal around 45/135 deg. Both diagonal orientations
    are combined to improve support for sparse irregular observations.
    """

    coordinates = np.asarray(coordinates_xy_m, dtype=np.float64)
    values = np.asarray(log10_values, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_xy_m must have shape [n,2] as x,y")
    if coordinates.shape[0] != values.size or values.size < 3:
        raise ValueError("at least three point values with matching coordinates are required")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(values)):
        raise ValueError("point coordinates and log10 values must be finite")
    lag_bin_m = float(lag_bin_m)
    max_lag_m = float(max_lag_m)
    angle_tolerance_deg = float(angle_tolerance_deg)
    min_pairs_per_bin = int(min_pairs_per_bin)
    if lag_bin_m <= 0.0 or max_lag_m <= lag_bin_m:
        raise ValueError("lag_bin_m must be positive and smaller than max_lag_m")
    if not (0.0 < angle_tolerance_deg < 45.0):
        raise ValueError("angle_tolerance_deg must lie in (0,45)")
    if min_pairs_per_bin <= 0:
        raise ValueError("min_pairs_per_bin must be positive")

    i, j = np.triu_indices(values.size, k=1)
    dx = coordinates[j, 0] - coordinates[i, 0]
    dy = coordinates[j, 1] - coordinates[i, 1]
    distance = np.hypot(dx, dy)
    semivariance = 0.5 * (values[j] - values[i]) ** 2
    valid_distance = (distance > 0.0) & (distance <= max_lag_m)

    angle = np.mod(np.arctan2(np.abs(dy), np.abs(dx)), np.pi / 2.0)
    tolerance = np.deg2rad(angle_tolerance_deg)
    direction_masks = {
        "x": angle <= tolerance,
        "y": np.abs(angle - np.pi / 2.0) <= tolerance,
        "diag": np.abs(angle - np.pi / 4.0) <= tolerance,
    }
    edges = np.arange(0.0, max_lag_m + lag_bin_m, lag_bin_m, dtype=np.float64)
    if edges[-1] < max_lag_m:
        edges = np.append(edges, max_lag_m)

    variograms: dict[str, tuple[Array, Array]] = {}
    counts_out: dict[str, Array] = {}
    for direction, direction_mask in direction_masks.items():
        mask = valid_distance & direction_mask
        distances = distance[mask]
        gammas = semivariance[mask]
        bin_index = np.digitize(distances, edges, right=False) - 1
        lag_centers: list[float] = []
        estimates: list[float] = []
        counts: list[int] = []
        for index in range(len(edges) - 1):
            selected = bin_index == index
            count = int(np.count_nonzero(selected))
            if count < min_pairs_per_bin:
                continue
            lag_centers.append(float(np.mean(distances[selected])))
            estimates.append(float(np.mean(gammas[selected])))
            counts.append(count)
        if len(lag_centers) < 3:
            raise ValueError(
                f"insufficient supported {direction} variogram bins; "
                "increase angle tolerance/max lag or lower min_pairs_per_bin"
            )
        variograms[direction] = (
            np.asarray(lag_centers, dtype=np.float64),
            np.asarray(estimates, dtype=np.float64),
        )
        counts_out[direction] = np.asarray(counts, dtype=np.int64)
    return variograms, counts_out


def calibrate_point_covariance_candidates(
    coordinates_xy_m: Array,
    hydraulic_conductivity_m_s: Array,
    *,
    models: Sequence[str] = SUPPORTED_CALIBRATION_MODELS,
    lag_bin_m: float = 250.0,
    max_lag_m: float = 3000.0,
    angle_tolerance_deg: float = 22.5,
    min_pairs_per_bin: int = 8,
) -> tuple[tuple[CovarianceCalibrationResult, ...], dict[str, tuple[Array, Array]], dict[str, Array]]:
    """Fit covariance candidates directly to irregular real measurements."""

    coordinates = np.asarray(coordinates_xy_m, dtype=np.float64)
    conductivity = np.asarray(hydraulic_conductivity_m_s, dtype=np.float64).reshape(-1)
    if coordinates.ndim != 2 or coordinates.shape != (conductivity.size, 2):
        raise ValueError("coordinates and conductivity values must have shapes [n,2] and [n]")
    if conductivity.size < 3 or not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("at least three finite positive conductivity measurements are required")
    log_values = np.log10(conductivity)
    mean = float(np.mean(log_values))
    std = float(np.std(log_values, ddof=1))
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("log10 measurement variance must be positive")
    sill = std**2

    requested = tuple(str(model).strip().lower() for model in models)
    if not requested:
        raise ValueError("at least one covariance model must be requested")
    unknown = sorted(set(requested).difference(SUPPORTED_CALIBRATION_MODELS))
    if unknown:
        raise ValueError(f"unsupported covariance models: {unknown}")

    variograms, counts = estimate_point_directional_variograms(
        coordinates,
        log_values,
        lag_bin_m=lag_bin_m,
        max_lag_m=max_lag_m,
        angle_tolerance_deg=angle_tolerance_deg,
        min_pairs_per_bin=min_pairs_per_bin,
    )
    x_span = max(float(np.ptp(coordinates[:, 0])), float(lag_bin_m))
    y_span = max(float(np.ptp(coordinates[:, 1])), float(lag_bin_m))
    initial_x = max(float(lag_bin_m), 0.1 * x_span)
    initial_y = max(float(lag_bin_m), 0.1 * y_span)
    lower = max(float(lag_bin_m) * 0.25, np.finfo(float).eps)
    upper = max(x_span, y_span, float(max_lag_m)) * 4.0

    results: list[CovarianceCalibrationResult] = []
    for model in requested:
        ly, lx, errors = _fit_candidate(
            covariance_model=model,
            variograms=variograms,
            sill=sill,
            initial_y_m=initial_y,
            initial_x_m=initial_x,
            lower_m=lower,
            upper_m=upper,
        )
        results.append(
            CovarianceCalibrationResult(
                covariance_model=model,
                mean_log10_k=mean,
                std_log10_k=std,
                between_field_mean_std_log10_k=0.0,
                length_scale_y_m=ly,
                length_scale_x_m=lx,
                angle_rad=0.0,
                variogram_rmse=errors["all"],
                variogram_rmse_y=errors["y"],
                variogram_rmse_x=errors["x"],
                variogram_rmse_diag=errors["diag"],
                cell_size_m=None,
                spatial_stride=None,
                max_lag_cells=None,
                calibration_source="irregular_real_measurements",
                lag_bin_m=float(lag_bin_m),
                max_lag_m=float(max_lag_m),
                angle_tolerance_deg=float(angle_tolerance_deg),
            )
        )
    return tuple(sorted(results, key=lambda item: item.variogram_rmse)), variograms, counts


def _matern32_rho(scaled_distance: Array) -> Array:
    root3 = np.sqrt(3.0) * np.asarray(scaled_distance, dtype=np.float64)
    return (1.0 + root3) * np.exp(-root3)


def correlation_for_offsets(
    covariance_model: str,
    *,
    delta_y_m: Array,
    delta_x_m: Array,
    length_scale_y_m: float,
    length_scale_x_m: float,
) -> Array:
    """Correlation for candidate covariance models at grid-aligned offsets."""

    model = str(covariance_model).strip().lower()
    dy = np.abs(np.asarray(delta_y_m, dtype=np.float64))
    dx = np.abs(np.asarray(delta_x_m, dtype=np.float64))
    ly = float(length_scale_y_m)
    lx = float(length_scale_x_m)
    if ly <= 0.0 or lx <= 0.0:
        raise ValueError("length scales must be positive")

    if model == "matern32":
        return _matern32_rho(dy / ly) * _matern32_rho(dx / lx)
    if model == "exponential":
        return np.exp(-(dy / ly + dx / lx))
    if model == "radial_exponential":
        return np.exp(-np.sqrt((dy / ly) ** 2 + (dx / lx) ** 2))
    raise ValueError(
        "covariance_model must be one of "
        f"{', '.join(SUPPORTED_CALIBRATION_MODELS)}"
    )


def _fit_candidate(
    *,
    covariance_model: str,
    variograms: Mapping[str, tuple[Array, Array]],
    sill: float,
    initial_y_m: float,
    initial_x_m: float,
    lower_m: float,
    upper_m: float,
) -> tuple[float, float, dict[str, float]]:
    y_dist, y_emp = variograms["y"]
    x_dist, x_emp = variograms["x"]
    diag_dist, diag_emp = variograms["diag"]

    def prediction(direction: str, distances: Array, ly: float, lx: float) -> Array:
        if direction == "y":
            rho = correlation_for_offsets(
                covariance_model,
                delta_y_m=distances,
                delta_x_m=np.zeros_like(distances),
                length_scale_y_m=ly,
                length_scale_x_m=lx,
            )
        elif direction == "x":
            rho = correlation_for_offsets(
                covariance_model,
                delta_y_m=np.zeros_like(distances),
                delta_x_m=distances,
                length_scale_y_m=ly,
                length_scale_x_m=lx,
            )
        else:
            component = distances / np.sqrt(2.0)
            rho = correlation_for_offsets(
                covariance_model,
                delta_y_m=component,
                delta_x_m=component,
                length_scale_y_m=ly,
                length_scale_x_m=lx,
            )
        return sill * (1.0 - rho)

    scale = max(float(sill), np.finfo(float).eps)

    def residual(log_scales: Array) -> Array:
        ly, lx = np.exp(log_scales)
        return np.concatenate(
            (
                (prediction("y", y_dist, ly, lx) - y_emp) / scale,
                (prediction("x", x_dist, ly, lx) - x_emp) / scale,
                (prediction("diag", diag_dist, ly, lx) - diag_emp) / scale,
            )
        )

    fit = least_squares(
        residual,
        x0=np.log([initial_y_m, initial_x_m]),
        bounds=(np.log([lower_m, lower_m]), np.log([upper_m, upper_m])),
        method="trf",
    )
    ly, lx = np.exp(fit.x)
    errors: dict[str, float] = {}
    for direction, distances, empirical in (
        ("y", y_dist, y_emp),
        ("x", x_dist, x_emp),
        ("diag", diag_dist, diag_emp),
    ):
        difference = prediction(direction, distances, ly, lx) - empirical
        errors[direction] = float(np.sqrt(np.mean(difference * difference)))
    all_differences = np.concatenate(
        (
            prediction("y", y_dist, ly, lx) - y_emp,
            prediction("x", x_dist, ly, lx) - x_emp,
            prediction("diag", diag_dist, ly, lx) - diag_emp,
        )
    )
    errors["all"] = float(np.sqrt(np.mean(all_differences * all_differences)))
    return float(ly), float(lx), errors


def calibrate_covariance_candidates(
    fields: Array,
    *,
    cell_size_m: float,
    models: Sequence[str] = SUPPORTED_CALIBRATION_MODELS,
    max_lag_cells: int = 64,
    spatial_stride: int = 1,
) -> tuple[CovarianceCalibrationResult, ...]:
    """Fit candidate covariance models to empirical log10(K) variograms.

    Results are sorted by joint x/y/diagonal variogram RMSE. The ordering is a
    calibration diagnostic and should not be presented as proof that the first
    model is the unique geological truth.
    """

    values = _as_positive_fields(fields)
    log_fields = np.log10(values)
    field_means = np.mean(log_fields, axis=(1, 2), dtype=np.float64)
    mean = float(np.mean(field_means))
    centered = np.asarray(log_fields, dtype=np.float64) - field_means[:, None, None]
    dof = centered.size - centered.shape[0]
    if dof <= 0:
        raise ValueError("not enough cells to estimate within-field log10 variance")
    std = float(np.sqrt(np.sum(centered * centered) / dof))
    between_std = (
        float(np.std(field_means, ddof=1)) if field_means.size >= 2 else 0.0
    )
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("log10 permeability must have positive within-field variance")
    sill = std * std

    requested = tuple(str(model).strip().lower() for model in models)
    if not requested:
        raise ValueError("at least one covariance model must be requested")
    unknown = sorted(set(requested).difference(SUPPORTED_CALIBRATION_MODELS))
    if unknown:
        raise ValueError(f"unsupported covariance models: {unknown}")

    variograms = estimate_directional_variograms(
        values,
        cell_size_m=cell_size_m,
        max_lag_cells=max_lag_cells,
        spatial_stride=spatial_stride,
    )
    h, w = values.shape[1:]
    spacing = float(cell_size_m) * int(spatial_stride)
    initial_y = max(spacing, 0.1 * h * float(cell_size_m))
    initial_x = max(spacing, 0.1 * w * float(cell_size_m))
    lower = max(spacing * 0.25, np.finfo(float).eps)
    upper = max(h, w) * float(cell_size_m) * 4.0

    results: list[CovarianceCalibrationResult] = []
    for model in requested:
        ly, lx, errors = _fit_candidate(
            covariance_model=model,
            variograms=variograms,
            sill=sill,
            initial_y_m=initial_y,
            initial_x_m=initial_x,
            lower_m=lower,
            upper_m=upper,
        )
        results.append(
            CovarianceCalibrationResult(
                covariance_model=model,
                mean_log10_k=mean,
                std_log10_k=std,
                between_field_mean_std_log10_k=between_std,
                length_scale_y_m=ly,
                length_scale_x_m=lx,
                angle_rad=0.0,
                variogram_rmse=errors["all"],
                variogram_rmse_y=errors["y"],
                variogram_rmse_x=errors["x"],
                variogram_rmse_diag=errors["diag"],
                cell_size_m=float(cell_size_m),
                spatial_stride=int(spatial_stride),
                max_lag_cells=int(max_lag_cells),
            )
        )
    return tuple(sorted(results, key=lambda item: item.variogram_rmse))
