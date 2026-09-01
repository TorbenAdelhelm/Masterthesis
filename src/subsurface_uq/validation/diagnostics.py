from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .temperature import TemperatureComparison

Array = np.ndarray


@dataclass(frozen=True)
class TemperatureDiagnostics:
    """Distributional diagnostics for one temperature comparison."""

    p50_absolute_error: float
    p90_absolute_error: float
    p95_absolute_error: float
    p99_absolute_error: float
    p999_absolute_error: float
    fraction_above_0p1: float
    fraction_above_0p5: float
    fraction_above_1p0: float


def summarize_temperature_errors(
    comparison: TemperatureComparison,
) -> TemperatureDiagnostics:
    """Summarize the spatial distribution of absolute temperature error.

    Fractions are returned as unit fractions in [0, 1]. Thresholds are in °C.
    """

    absolute = np.asarray(comparison.absolute_error, dtype=np.float64)
    if absolute.ndim != 2:
        raise ValueError(f"absolute error must be 2-D, got {absolute.shape}")
    if not np.all(np.isfinite(absolute)):
        raise ValueError("absolute error contains non-finite values")

    percentiles = np.percentile(absolute, [50.0, 90.0, 95.0, 99.0, 99.9])
    return TemperatureDiagnostics(
        p50_absolute_error=float(percentiles[0]),
        p90_absolute_error=float(percentiles[1]),
        p95_absolute_error=float(percentiles[2]),
        p99_absolute_error=float(percentiles[3]),
        p999_absolute_error=float(percentiles[4]),
        fraction_above_0p1=float(np.mean(absolute > 0.1)),
        fraction_above_0p5=float(np.mean(absolute > 0.5)),
        fraction_above_1p0=float(np.mean(absolute > 1.0)),
    )


def _save_field_plot(
    field: Array,
    path: Path,
    *,
    title: str,
    colorbar_label: str,
    symmetric: bool = False,
) -> None:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"field plot expects 2-D data, got {values.shape}")

    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    image_kwargs: dict[str, float | str] = {"origin": "lower", "cmap": "viridis"}
    if symmetric:
        extent = float(np.max(np.abs(values)))
        if extent == 0.0:
            extent = 1.0
        image_kwargs.update({"cmap": "coolwarm", "vmin": -extent, "vmax": extent})
    image = axis.imshow(values, **image_kwargs)
    axis.set_title(title)
    axis.set_xlabel("x pixel")
    axis.set_ylabel("y pixel")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_temperature_diagnostic_plots(
    comparison: TemperatureComparison,
    output_dir: str | Path,
    *,
    prefix: str = "temperature_validation",
) -> dict[str, Path]:
    """Save four diagnostic PNG maps and return their paths."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    paths = {
        "prediction": destination / f"{prefix}_prediction.png",
        "reference": destination / f"{prefix}_reference.png",
        "signed_error": destination / f"{prefix}_signed_error.png",
        "absolute_error": destination / f"{prefix}_absolute_error.png",
    }
    _save_field_plot(
        comparison.prediction,
        paths["prediction"],
        title="Predicted temperature",
        colorbar_label="Temperature [°C]",
    )
    _save_field_plot(
        comparison.reference,
        paths["reference"],
        title="Reference temperature",
        colorbar_label="Temperature [°C]",
    )
    _save_field_plot(
        comparison.error,
        paths["signed_error"],
        title="Signed temperature error (prediction - reference)",
        colorbar_label="Error [°C]",
        symmetric=True,
    )
    _save_field_plot(
        comparison.absolute_error,
        paths["absolute_error"],
        title="Absolute temperature error",
        colorbar_label="Absolute error [°C]",
    )
    return paths
