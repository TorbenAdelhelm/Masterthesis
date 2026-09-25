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
