from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import SUPPORTED_CALIBRATION_MODELS, correlation_for_offsets

Array = np.ndarray


@dataclass(frozen=True)
class ExactSimpleKrigingResult:
    """Exact gridwise Gaussian simple-kriging posterior in log10(K)."""

    mean_log10_k: Array
    variance_log10_k: Array
    std_log10_k: Array
    conditioning_max_abs_log10_residual: float
    conditioning_rms_log10_residual: float


def _validate_observations(
    *,
    shape: tuple[int, int],
    observation_indices: Array,
    observation_log10_k: Array,
    observation_std_log10_k: float | Array,
) -> tuple[Array, Array, Array]:
    raw = np.asarray(observation_indices)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
        raise ValueError("observation_indices must have shape [n,2]")
    if not np.all(np.isfinite(raw)) or not np.array_equal(raw, np.rint(raw)):
        raise ValueError("observation_indices must contain finite integer grid cells")
    indices = np.rint(raw).astype(np.int64)
    if len({tuple(row) for row in indices.tolist()}) != indices.shape[0]:
        raise ValueError("duplicate observation cells are not supported")
    rows, cols = indices[:, 0], indices[:, 1]
    if np.any(rows < 0) or np.any(rows >= shape[0]):
        raise ValueError("observation row lies outside the grid")
    if np.any(cols < 0) or np.any(cols >= shape[1]):
        raise ValueError("observation column lies outside the grid")

    values = np.asarray(observation_log10_k, dtype=np.float64).reshape(-1)
    if values.shape[0] != indices.shape[0] or not np.all(np.isfinite(values)):
        raise ValueError("one finite log10 permeability value is required per observation")

    noise = np.asarray(observation_std_log10_k, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.full(values.shape, float(noise), dtype=np.float64)
    else:
        noise = noise.reshape(-1)
    if noise.shape != values.shape:
        raise ValueError("observation_std_log10_k must be scalar or length n_observations")
    if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
        raise ValueError("observation_std_log10_k must be finite and non-negative")
    return indices, values, noise


def _symmetric_psd_inverse(matrix: Array, *, rank_tolerance: float) -> Array:
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    values, vectors = np.linalg.eigh(symmetric)
    scale = max(float(np.max(np.abs(values))), 1.0)
    tolerance = float(rank_tolerance) * scale
    if float(np.min(values)) < -10.0 * tolerance:
        raise RuntimeError("observation covariance is not positive semidefinite")
    positive = values > tolerance
    if not np.any(positive):
        raise ValueError("observation covariance has no positive direction")
    return (vectors[:, positive] * (1.0 / values[positive])[None, :]) @ vectors[:, positive].T


def exact_simple_kriging_posterior(
    *,
    shape: tuple[int, int],
    cell_size_m: float,
    covariance_model: str,
    mean_log10_k: float,
    std_log10_k: float,
    length_scale_y_m: float,
    length_scale_x_m: float,
    observation_indices: Array,
    observation_log10_k: Array,
    observation_std_log10_k: float | Array = 0.0,
    angle_rad: float = 0.0,
    chunk_rows: int = 64,
    rank_tolerance: float = 1e-10,
) -> ExactSimpleKrigingResult:
    """Evaluate the full-covariance simple-kriging posterior on a regular grid.

    Only the observation covariance ``K_DD`` and chunked cross-covariances
    ``K_xD`` are formed. No dense ``(H*W) x (H*W)`` covariance is materialized.
    This makes covariance-family validation independent of any KL truncation.

    The current covariance calibration is grid-aligned. A non-zero radial
    anisotropy angle is therefore rejected here rather than silently using a
    convention that might differ from the GSTools rotation convention.
    """

    shape = (int(shape[0]), int(shape[1]))
    if any(value <= 0 for value in shape):
        raise ValueError("shape must contain two positive entries")
    cell_size_m = float(cell_size_m)
    mean_log10_k = float(mean_log10_k)
    std_log10_k = float(std_log10_k)
    length_scale_y_m = float(length_scale_y_m)
    length_scale_x_m = float(length_scale_x_m)
    chunk_rows = int(chunk_rows)
    rank_tolerance = float(rank_tolerance)
    model = str(covariance_model).strip().lower()
    if model not in SUPPORTED_CALIBRATION_MODELS:
        raise ValueError(f"unsupported covariance model: {model}")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    if not np.isfinite(mean_log10_k):
        raise ValueError("mean_log10_k must be finite")
    if not np.isfinite(std_log10_k) or std_log10_k <= 0.0:
        raise ValueError("std_log10_k must be finite and positive")
    if length_scale_y_m <= 0.0 or length_scale_x_m <= 0.0:
        raise ValueError("length scales must be positive")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if rank_tolerance <= 0.0 or not np.isfinite(rank_tolerance):
        raise ValueError("rank_tolerance must be finite and positive")
    if abs(float(angle_rad)) > 1e-14:
        raise NotImplementedError(
            "exact full-covariance evaluation currently supports grid-aligned anisotropy only"
        )

    indices, values, noise = _validate_observations(
        shape=shape,
        observation_indices=observation_indices,
        observation_log10_k=observation_log10_k,
        observation_std_log10_k=observation_std_log10_k,
    )
    obs_y = (indices[:, 0].astype(np.float64) + 0.5) * cell_size_m
    obs_x = (indices[:, 1].astype(np.float64) + 0.5) * cell_size_m
    dy_dd = obs_y[:, None] - obs_y[None, :]
    dx_dd = obs_x[:, None] - obs_x[None, :]
    sill = std_log10_k**2
    rho_dd = correlation_for_offsets(
        model,
        delta_y_m=dy_dd,
        delta_x_m=dx_dd,
        length_scale_y_m=length_scale_y_m,
        length_scale_x_m=length_scale_x_m,
    )
    covariance_dd = sill * rho_dd + np.diag(noise**2)
    inverse_dd = _symmetric_psd_inverse(covariance_dd, rank_tolerance=rank_tolerance)
    alpha = inverse_dd @ (values - mean_log10_k)

    h, w = shape
    x_centers = (np.arange(w, dtype=np.float64) + 0.5) * cell_size_m
    mean = np.empty(shape, dtype=np.float64)
    variance = np.empty(shape, dtype=np.float64)
    for row_start in range(0, h, chunk_rows):
        row_stop = min(h, row_start + chunk_rows)
        y_centers = (np.arange(row_start, row_stop, dtype=np.float64) + 0.5) * cell_size_m
        grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing="ij")
        flat_y = grid_y.reshape(-1, 1)
        flat_x = grid_x.reshape(-1, 1)
        rho_xd = correlation_for_offsets(
            model,
            delta_y_m=flat_y - obs_y[None, :],
            delta_x_m=flat_x - obs_x[None, :],
            length_scale_y_m=length_scale_y_m,
            length_scale_x_m=length_scale_x_m,
        )
        covariance_xd = sill * rho_xd
        posterior_mean = mean_log10_k + covariance_xd @ alpha
        solved = covariance_xd @ inverse_dd
        posterior_variance = sill - np.sum(solved * covariance_xd, axis=1)
        posterior_variance = np.maximum(posterior_variance, 0.0)
        block_shape = (row_stop - row_start, w)
        mean[row_start:row_stop] = posterior_mean.reshape(block_shape)
        variance[row_start:row_stop] = posterior_variance.reshape(block_shape)

    residuals = mean[indices[:, 0], indices[:, 1]] - values
    return ExactSimpleKrigingResult(
        mean_log10_k=mean.astype(np.float32),
        variance_log10_k=variance.astype(np.float32),
        std_log10_k=np.sqrt(variance).astype(np.float32),
        conditioning_max_abs_log10_residual=float(np.max(np.abs(residuals))),
        conditioning_rms_log10_residual=float(np.sqrt(np.mean(residuals * residuals))),
    )


@dataclass(frozen=True)
class ExactPointKrigingResult:
    """Exact simple-kriging posterior at arbitrary query coordinates."""

    mean_log10_k: Array
    variance_log10_k: Array
    std_log10_k: Array


def exact_simple_kriging_predict_points(
    *,
    observation_coordinates_xy_m: Array,
    observation_log10_k: Array,
    query_coordinates_xy_m: Array,
    covariance_model: str,
    mean_log10_k: float,
    std_log10_k: float,
    length_scale_y_m: float,
    length_scale_x_m: float,
    observation_std_log10_k: float | Array = 0.0,
    structured_std_log10_k: float | None = None,
    nugget_std_log10_k: float = 0.0,
    include_query_nugget: bool = False,
    angle_rad: float = 0.0,
    rank_tolerance: float = 1e-10,
    chunk_size: int = 100000,
) -> ExactPointKrigingResult:
    """Evaluate the full-covariance posterior at continuous XY coordinates."""

    observation_coordinates = np.asarray(observation_coordinates_xy_m, dtype=np.float64)
    queries = np.asarray(query_coordinates_xy_m, dtype=np.float64)
    values = np.asarray(observation_log10_k, dtype=np.float64).reshape(-1)
    if observation_coordinates.ndim != 2 or observation_coordinates.shape[1] != 2:
        raise ValueError("observation_coordinates_xy_m must have shape [n,2] as x,y")
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query_coordinates_xy_m must have shape [m,2] as x,y")
    if observation_coordinates.shape[0] != values.size or values.size == 0:
        raise ValueError("one observation value is required per observation coordinate")
    if not np.all(np.isfinite(observation_coordinates)) or not np.all(np.isfinite(queries)):
        raise ValueError("observation and query coordinates must be finite")
    if not np.all(np.isfinite(values)):
        raise ValueError("observation_log10_k must be finite")
    if abs(float(angle_rad)) > 1e-14:
        raise NotImplementedError(
            "continuous exact kriging currently supports grid-aligned anisotropy only"
        )
    model = str(covariance_model).strip().lower()
    if model not in SUPPORTED_CALIBRATION_MODELS:
        raise ValueError(f"unsupported covariance model: {model}")
    mean = float(mean_log10_k)
    std = float(std_log10_k)
    structured_std = (
        std if structured_std_log10_k is None else float(structured_std_log10_k)
    )
    nugget_std = float(nugget_std_log10_k)
    ly = float(length_scale_y_m)
    lx = float(length_scale_x_m)
    if not np.isfinite(mean):
        raise ValueError("mean_log10_k must be finite")
    if (
        not np.isfinite(std)
        or std <= 0.0
        or not np.isfinite(structured_std)
        or structured_std <= 0.0
        or not np.isfinite(nugget_std)
        or nugget_std < 0.0
        or ly <= 0.0
        or lx <= 0.0
    ):
        raise ValueError(
            "total/structured standard deviations and length scales must be positive; "
            "nugget_std_log10_k must be non-negative"
        )
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    noise = np.asarray(observation_std_log10_k, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.full(values.shape, float(noise), dtype=np.float64)
    else:
        noise = noise.reshape(-1)
    if noise.shape != values.shape:
        raise ValueError("observation_std_log10_k must be scalar or length n_observations")
    if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
        raise ValueError("observation_std_log10_k must be finite and non-negative")

    obs_x = observation_coordinates[:, 0]
    obs_y = observation_coordinates[:, 1]
    dy_dd = obs_y[:, None] - obs_y[None, :]
    dx_dd = obs_x[:, None] - obs_x[None, :]
    structured_variance = structured_std**2
    nugget_variance = nugget_std**2
    covariance_dd = structured_variance * correlation_for_offsets(
        model,
        delta_y_m=dy_dd,
        delta_x_m=dx_dd,
        length_scale_y_m=ly,
        length_scale_x_m=lx,
    ) + np.diag(nugget_variance + noise**2)
    inverse_dd = _symmetric_psd_inverse(
        covariance_dd, rank_tolerance=float(rank_tolerance)
    )
    alpha = inverse_dd @ (values - mean)

    posterior_mean = np.empty(queries.shape[0], dtype=np.float64)
    posterior_variance = np.empty(queries.shape[0], dtype=np.float64)
    for start in range(0, queries.shape[0], chunk_size):
        stop = min(queries.shape[0], start + chunk_size)
        block = queries[start:stop]
        rho_qd = correlation_for_offsets(
            model,
            delta_y_m=block[:, 1, None] - obs_y[None, :],
            delta_x_m=block[:, 0, None] - obs_x[None, :],
            length_scale_y_m=ly,
            length_scale_x_m=lx,
        )
        covariance_qd = structured_variance * rho_qd
        posterior_mean[start:stop] = mean + covariance_qd @ alpha
        solved = covariance_qd @ inverse_dd
        query_variance = structured_variance + (
            nugget_variance if include_query_nugget else 0.0
        )
        posterior_variance[start:stop] = np.maximum(
            query_variance - np.sum(solved * covariance_qd, axis=1), 0.0
        )

    return ExactPointKrigingResult(
        mean_log10_k=posterior_mean,
        variance_log10_k=posterior_variance,
        std_log10_k=np.sqrt(posterior_variance),
    )


def exact_simple_kriging_grid_from_points(
    *,
    observation_coordinates_xy_m: Array,
    observation_log10_k: Array,
    x_min_m: float,
    y_min_m: float,
    nx: int,
    ny: int,
    cell_size_x_m: float,
    cell_size_y_m: float,
    covariance_model: str,
    mean_log10_k: float,
    std_log10_k: float,
    length_scale_y_m: float,
    length_scale_x_m: float,
    observation_std_log10_k: float | Array = 0.0,
    structured_std_log10_k: float | None = None,
    nugget_std_log10_k: float = 0.0,
    include_query_nugget: bool = False,
    angle_rad: float = 0.0,
    active_mask: Array | None = None,
    chunk_size: int = 100000,
) -> ExactSimpleKrigingResult:
    """Evaluate a continuous-point posterior on a regular reference XY grid."""

    nx = int(nx)
    ny = int(ny)
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive")
    x = float(x_min_m) + np.arange(nx, dtype=np.float64) * float(cell_size_x_m)
    y = float(y_min_m) + np.arange(ny, dtype=np.float64) * float(cell_size_y_m)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    query = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    point_result = exact_simple_kriging_predict_points(
        observation_coordinates_xy_m=observation_coordinates_xy_m,
        observation_log10_k=observation_log10_k,
        query_coordinates_xy_m=query,
        covariance_model=covariance_model,
        mean_log10_k=mean_log10_k,
        std_log10_k=std_log10_k,
        length_scale_y_m=length_scale_y_m,
        length_scale_x_m=length_scale_x_m,
        observation_std_log10_k=observation_std_log10_k,
        structured_std_log10_k=structured_std_log10_k,
        nugget_std_log10_k=nugget_std_log10_k,
        include_query_nugget=include_query_nugget,
        angle_rad=angle_rad,
        chunk_size=chunk_size,
    )
    mean = point_result.mean_log10_k.reshape(ny, nx)
    variance = point_result.variance_log10_k.reshape(ny, nx)
    std = point_result.std_log10_k.reshape(ny, nx)
    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != (ny, nx):
            raise ValueError("active_mask must have shape [ny,nx]")
        mean = np.where(mask, mean, np.nan)
        variance = np.where(mask, variance, np.nan)
        std = np.where(mask, std, np.nan)

    observations = np.asarray(observation_coordinates_xy_m, dtype=np.float64)
    observed_values = np.asarray(observation_log10_k, dtype=np.float64).reshape(-1)
    at_observations = exact_simple_kriging_predict_points(
        observation_coordinates_xy_m=observations,
        observation_log10_k=observed_values,
        query_coordinates_xy_m=observations,
        covariance_model=covariance_model,
        mean_log10_k=mean_log10_k,
        std_log10_k=std_log10_k,
        length_scale_y_m=length_scale_y_m,
        length_scale_x_m=length_scale_x_m,
        observation_std_log10_k=observation_std_log10_k,
        structured_std_log10_k=structured_std_log10_k,
        nugget_std_log10_k=nugget_std_log10_k,
        include_query_nugget=False,
        angle_rad=angle_rad,
        chunk_size=chunk_size,
    )
    residuals = at_observations.mean_log10_k - observed_values
    return ExactSimpleKrigingResult(
        mean_log10_k=mean.astype(np.float32),
        variance_log10_k=variance.astype(np.float32),
        std_log10_k=std.astype(np.float32),
        conditioning_max_abs_log10_residual=float(np.max(np.abs(residuals))),
        conditioning_rms_log10_residual=float(np.sqrt(np.mean(residuals**2))),
    )
