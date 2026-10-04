from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.lib.format import open_memmap

Array = np.ndarray


def scalar_summary(
    values: Array,
    *,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, float | int]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("scalar samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(sample)):
        raise ValueError("scalar samples must be finite")
    q = np.quantile(sample, tuple(float(v) for v in quantiles))
    return {
        "count": int(sample.size),
        "mean": float(np.mean(sample)),
        "sd": float(np.std(sample, ddof=1)) if sample.size >= 2 else 0.0,
        "q05": float(q[0]),
        "q50": float(q[1]),
        "q95": float(q[2]),
    }


def global_uncertainty_metrics(std: Array) -> dict[str, float]:
    values = np.asarray(std, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("temperature standard-deviation field must be finite and non-negative")
    return {
        "U_RMS": float(np.sqrt(np.mean(values * values))),
        "U95": float(np.quantile(values, 0.95)),
        "mean_std": float(np.mean(values)),
        "max_std": float(np.max(values)),
    }


def write_temperature_field_products(
    *,
    temperature_store: str | Path,
    final_mean: Array,
    final_std: Array,
    output_directory: str | Path,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    chunk_rows: int = 16,
) -> dict[str, Path]:
    """Write the finalized RQ1 mean/std/empirical-quantile field products."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    q = tuple(float(v) for v in quantiles)
    if len(q) != 3 or not np.allclose(q, (0.05, 0.5, 0.95)):
        raise ValueError("RQ1 field quantiles must be 0.05, 0.5 and 0.95")

    source = np.load(Path(temperature_store).expanduser().resolve(), mmap_mode="r")
    if source.ndim != 3 or source.shape[0] < 2:
        raise ValueError("temperature store must have shape [N,H,W] with N>=2")
    h, w = int(source.shape[1]), int(source.shape[2])
    mean = np.asarray(final_mean, dtype=np.float32)
    std = np.asarray(final_std, dtype=np.float32)
    if mean.shape != (h, w) or std.shape != (h, w):
        raise ValueError("final mean/std do not match disk-backed temperature shape")

    root = Path(output_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "temperature_mean": root / "temperature_mean.npy",
        "temperature_std": root / "temperature_std.npy",
        "temperature_q05": root / "temperature_q05.npy",
        "temperature_q50": root / "temperature_q50.npy",
        "temperature_q95": root / "temperature_q95.npy",
        "temperature_width90": root / "temperature_width90.npy",
    }
    np.save(paths["temperature_mean"], mean)
    np.save(paths["temperature_std"], std)

    q05 = open_memmap(paths["temperature_q05"], mode="w+", dtype=np.float32, shape=(h, w))
    q50 = open_memmap(paths["temperature_q50"], mode="w+", dtype=np.float32, shape=(h, w))
    q95 = open_memmap(paths["temperature_q95"], mode="w+", dtype=np.float32, shape=(h, w))
    width = open_memmap(
        paths["temperature_width90"], mode="w+", dtype=np.float32, shape=(h, w)
    )
    for start in range(0, h, chunk_rows):
        stop = min(start + chunk_rows, h)
        block = np.asarray(source[:, start:stop, :], dtype=np.float32)
        values = np.quantile(block, q, axis=0)
        q05[start:stop] = values[0].astype(np.float32)
        q50[start:stop] = values[1].astype(np.float32)
        q95[start:stop] = values[2].astype(np.float32)
        width[start:stop] = (values[2] - values[0]).astype(np.float32)
    for array in (q05, q50, q95, width):
        array.flush()
    return paths


def prefix_field_convergence(
    *,
    temperature_store: str | Path,
    checkpoints: Sequence[int],
    reference_mean: Array,
    reference_std: Array,
    ddof: int = 1,
    chunk_rows: int = 16,
) -> list[dict[str, float | int]]:
    """Compute nested-prefix field convergence against an empirical reference."""

    source = np.load(Path(temperature_store).expanduser().resolve(), mmap_mode="r")
    if source.ndim != 3:
        raise ValueError("temperature store must have shape [N,H,W]")
    h, w = int(source.shape[1]), int(source.shape[2])
    reference_mean = np.asarray(reference_mean, dtype=np.float64)
    reference_std = np.asarray(reference_std, dtype=np.float64)
    if reference_mean.shape != (h, w) or reference_std.shape != (h, w):
        raise ValueError("reference fields do not match temperature store")

    points: list[dict[str, float | int]] = []
    cell_count = h * w
    for n_raw in checkpoints:
        n = int(n_raw)
        if n <= ddof or n > source.shape[0]:
            raise ValueError(f"invalid convergence checkpoint N={n}")
        sse_mean = 0.0
        sse_std = 0.0
        variance_sum = 0.0
        std_sum = 0.0
        std_max = 0.0
        std_flat = np.empty(cell_count, dtype=np.float32)
        cursor = 0

        for start in range(0, h, chunk_rows):
            stop = min(start + chunk_rows, h)
            block = np.asarray(source[:n, start:stop, :], dtype=np.float64)
            current_mean = np.mean(block, axis=0)
            current_std = np.std(block, axis=0, ddof=ddof)
            diff_mean = current_mean - reference_mean[start:stop]
            diff_std = current_std - reference_std[start:stop]
            sse_mean += float(np.sum(diff_mean * diff_mean))
            sse_std += float(np.sum(diff_std * diff_std))
            variance_sum += float(np.sum(current_std * current_std))
            std_sum += float(np.sum(current_std))
            std_max = max(std_max, float(np.max(current_std)))
            flat = current_std.astype(np.float32, copy=False).ravel()
            std_flat[cursor : cursor + flat.size] = flat
            cursor += flat.size

        points.append(
            {
                "budget": n,
                "e_mu": float(np.sqrt(sse_mean / cell_count)),
                "e_sigma": float(np.sqrt(sse_std / cell_count)),
                "U_RMS": float(np.sqrt(variance_sum / cell_count)),
                "U95": float(np.quantile(std_flat, 0.95)),
                "mean_std": float(std_sum / cell_count),
                "max_std": float(std_max),
            }
        )
    return points


def scalar_prefix_convergence(
    values: Array,
    *,
    checkpoints: Sequence[int],
    reference: Mapping[str, float | int],
    target: str,
) -> list[dict[str, float | int | str]]:
    sample = np.asarray(values, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for n_raw in checkpoints:
        n = int(n_raw)
        if n > sample.size:
            raise ValueError(f"checkpoint {n} exceeds scalar sample count {sample.size}")
        current = scalar_summary(sample[:n])
        for statistic in ("mean", "sd", "q05", "q50", "q95"):
            estimate = float(current[statistic])
            reference_value = float(reference[statistic])
            error = estimate - reference_value
            rows.append(
                {
                    "budget": n,
                    "target": target,
                    "statistic": statistic,
                    "estimate": estimate,
                    "reference": reference_value,
                    "error": error,
                    "abs_error": abs(error),
                }
            )
    return rows
