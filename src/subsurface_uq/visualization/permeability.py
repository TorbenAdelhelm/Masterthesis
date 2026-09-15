from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..sampling.diagnostics import PermeabilityDiagnosticsResult

Array = np.ndarray


def _validate_observation_indices(
    observation_indices: Array | None,
    *,
    shape: tuple[int, int],
) -> Array | None:
    if observation_indices is None:
        return None
    indices = np.asarray(observation_indices)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("observation_indices must have shape [n,2]")
    if not np.all(np.isfinite(indices)):
        raise ValueError("observation_indices must be finite")
    rounded = np.rint(indices)
    if not np.array_equal(indices, rounded):
        raise ValueError("observation_indices must contain integer grid cells")
    indices = rounded.astype(np.int64)
    h, w = shape
    if np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= h):
        raise ValueError("observation row index lies outside the field")
    if np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= w):
        raise ValueError("observation column index lies outside the field")
    return indices


def _save_log10k_map(
    field: Array,
    destination: Path,
    *,
    title: str,
    cell_size_m: float,
    observation_indices: Array | None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"permeability plot field must be 2-D, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("permeability plot field contains non-finite values")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    h, w = values.shape
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="lower",
        extent=(0.0, w * cell_size_m, 0.0, h * cell_size_m),
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    if observation_indices is not None and len(observation_indices) > 0:
        rows = observation_indices[:, 0]
        cols = observation_indices[:, 1]
        axis.scatter(
            (cols + 0.5) * cell_size_m,
            (rows + 0.5) * cell_size_m,
            marker="x",
            label="conditioning cell",
        )
        axis.legend(loc="best")
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("log10(K / m²)")
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def _save_std_map(
    field: Array,
    destination: Path,
    *,
    title: str,
    cell_size_m: float,
    observation_indices: Array | None,
) -> Path:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"permeability std field must be 2-D, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("permeability std field must be finite and non-negative")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    h, w = values.shape
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="lower",
        extent=(0.0, w * cell_size_m, 0.0, h * cell_size_m),
        aspect="equal",
    )
    if observation_indices is not None and len(observation_indices) > 0:
        rows = observation_indices[:, 0]
        cols = observation_indices[:, 1]
        axis.scatter(
            (cols + 0.5) * cell_size_m,
            (rows + 0.5) * cell_size_m,
            marker="x",
            label="conditioning cell",
        )
        axis.legend(loc="best")
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("std. dev. log10(K / m²)")
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_permeability_diagnostics(
    diagnostics: PermeabilityDiagnosticsResult,
    directory: str | Path,
    *,
    cell_size_m: float = 5.0,
    observation_indices: Array | None = None,
) -> dict[str, Path]:
    """Save streaming GRF input diagnostics without storing the full ensemble."""

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(value) for value in diagnostics.mean_log10_k.shape)
    observations = _validate_observation_indices(observation_indices, shape=shape)

    saved: dict[str, Path] = {}
    saved["log10k_mean"] = _save_log10k_map(
        diagnostics.mean_log10_k,
        root / "log10k_mean.png",
        title=f"Mean log10 permeability (N={diagnostics.count})",
        cell_size_m=cell_size_m,
        observation_indices=observations,
    )
    saved["log10k_std"] = _save_std_map(
        diagnostics.std_log10_k,
        root / "log10k_std.png",
        title=f"Std. dev. of log10 permeability (N={diagnostics.count})",
        cell_size_m=cell_size_m,
        observation_indices=observations,
    )

    if diagnostics.preview_log10_k:
        preview_min = min(float(np.min(field)) for field in diagnostics.preview_log10_k)
        preview_max = max(float(np.max(field)) for field in diagnostics.preview_log10_k)
        if preview_min == preview_max:
            preview_min = None
            preview_max = None
        for index, field in enumerate(diagnostics.preview_log10_k, start=1):
            key = f"log10k_sample_{index:03d}"
            saved[key] = _save_log10k_map(
                field,
                root / f"{key}.png",
                title=f"log10 permeability — sample {index}",
                cell_size_m=cell_size_m,
                observation_indices=observations,
                vmin=preview_min,
                vmax=preview_max,
            )

    return saved
