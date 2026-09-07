from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MonteCarloPlotData:
    """Validated fields loaded from a saved Monte Carlo result archive."""

    count: int
    mean: Array
    std: Array
    minimum: Array
    maximum: Array
    metadata: Mapping[str, object]
    exceedance_thresholds: tuple[float, ...] = ()
    exceedance_probabilities: Array | None = None

    @property
    def field_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.mean.shape)

    @property
    def temperature_range(self) -> Array:
        return self.maximum - self.minimum


def _validated_2d(value: Array, name: str) -> Array:
    field = np.asarray(value, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError(f"{name} contains non-finite values")
    return field


def load_monte_carlo_plot_data(path: str | Path) -> MonteCarloPlotData:
    """Load one UQ result archive produced by ``save_monte_carlo_result``.

    Older archives without online exceedance fields remain supported; mean,
    standard deviation, minimum and maximum are sufficient for the core plots.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=False) as archive:
        required = {"count", "mean", "std", "minimum", "maximum"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Monte Carlo archive is missing fields: {missing}")

        count = int(np.asarray(archive["count"]).item())
        if count <= 0:
            raise ValueError("Monte Carlo sample count must be positive")

        mean = _validated_2d(archive["mean"], "mean")
        std = _validated_2d(archive["std"], "std")
        minimum = _validated_2d(archive["minimum"], "minimum")
        maximum = _validated_2d(archive["maximum"], "maximum")
        shape = mean.shape
        for name, field in (("std", std), ("minimum", minimum), ("maximum", maximum)):
            if field.shape != shape:
                raise ValueError(f"{name} shape {field.shape} does not match mean {shape}")
        if np.any(std < 0.0):
            raise ValueError("standard deviation must be non-negative")
        if np.any(maximum < minimum):
            raise ValueError("maximum temperature must be >= minimum temperature")

        metadata: dict[str, object] = {}
        if "metadata_json" in archive.files:
            raw = str(np.asarray(archive["metadata_json"]).item())
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("metadata_json must encode a JSON object")
            metadata = parsed

        has_thresholds = "exceedance_thresholds" in archive.files
        has_probabilities = "exceedance_probabilities" in archive.files
        if has_thresholds != has_probabilities:
            raise ValueError(
                "exceedance_thresholds and exceedance_probabilities must either both be present or both be absent"
            )

        thresholds: tuple[float, ...] = ()
        probabilities: Array | None = None
        if has_thresholds:
            threshold_values = np.asarray(archive["exceedance_thresholds"], dtype=np.float64)
            probabilities = np.asarray(archive["exceedance_probabilities"], dtype=np.float32)
            if threshold_values.ndim != 1:
                raise ValueError("exceedance_thresholds must be one-dimensional")
            if probabilities.ndim != 3:
                raise ValueError("exceedance_probabilities must have shape [K,H,W]")
            if probabilities.shape != (len(threshold_values), *shape):
                raise ValueError(
                    "exceedance probability shape does not match thresholds/temperature field"
                )
            if not np.all(np.isfinite(probabilities)):
                raise ValueError("exceedance probabilities contain non-finite values")
            if np.any((probabilities < 0.0) | (probabilities > 1.0)):
                raise ValueError("exceedance probabilities must lie in [0, 1]")
            thresholds = tuple(float(value) for value in threshold_values)

    return MonteCarloPlotData(
        count=count,
        mean=mean,
        std=std,
        minimum=minimum,
        maximum=maximum,
        metadata=metadata,
        exceedance_thresholds=thresholds,
        exceedance_probabilities=probabilities,
    )


def _extent_km(shape: tuple[int, int], cell_size_m: float) -> tuple[float, float, float, float]:
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    h, w = shape
    return (0.0, w * cell_size_m / 1000.0, 0.0, h * cell_size_m / 1000.0)


def _save_map(
    field: Array,
    destination: Path,
    *,
    title: str,
    colorbar_label: str,
    cell_size_m: float,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    norm=None,
) -> Path:
    values = _validated_2d(field, title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="lower",
        extent=_extent_km(values.shape, cell_size_m),
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel("x [km]")
    axis.set_ylabel("y [km]")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def _threshold_token(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _background_from_metadata(metadata: Mapping[str, object]) -> float | None:
    value = metadata.get("background_temperature")
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("background_temperature in metadata must be finite")
    return result


def save_monte_carlo_uq_plots(
    data: MonteCarloPlotData,
    directory: str | Path,
    *,
    prefix: str = "monte_carlo",
    cell_size_m: float = 5.0,
    background_temperature: float | None = None,
) -> dict[str, Path]:
    """Save publication-ready spatial diagnostics for a Monte Carlo UQ result.

    Core outputs are mean temperature, standard deviation, sample range and a
    mean/std overlay. If a background temperature is supplied (or stored in the
    metadata), a mean Delta-T map is also produced. Online exceedance maps are
    rendered when present in the archive.
    """

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if background_temperature is None:
        background_temperature = _background_from_metadata(data.metadata)
    elif not np.isfinite(background_temperature):
        raise ValueError("background_temperature must be finite")

    saved: dict[str, Path] = {}
    saved["temperature_mean"] = _save_map(
        data.mean,
        root / f"{prefix}_temperature_mean.png",
        title=f"Monte Carlo mean temperature (N={data.count})",
        colorbar_label="temperature [degC]",
        cell_size_m=cell_size_m,
        cmap="viridis",
    )
    saved["temperature_std"] = _save_map(
        data.std,
        root / f"{prefix}_temperature_std.png",
        title=f"Monte Carlo temperature standard deviation (N={data.count})",
        colorbar_label="standard deviation [degC]",
        cell_size_m=cell_size_m,
        cmap="magma",
        vmin=0.0,
    )
    saved["temperature_range"] = _save_map(
        data.temperature_range,
        root / f"{prefix}_temperature_range.png",
        title=f"Monte Carlo temperature range (N={data.count})",
        colorbar_label="max - min [degC]",
        cell_size_m=cell_size_m,
        cmap="magma",
        vmin=0.0,
    )

    extent = _extent_km(data.field_shape, cell_size_m)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        data.mean,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
    )
    positive_std = data.std[data.std > 0.0]
    if positive_std.size:
        levels = np.unique(np.quantile(positive_std, [0.5, 0.75, 0.9]))
        levels = levels[(levels > float(np.min(data.std))) & (levels < float(np.max(data.std)))]
        if levels.size:
            x = np.linspace(extent[0], extent[1], data.std.shape[1])
            y = np.linspace(extent[2], extent[3], data.std.shape[0])
            contours = axis.contour(
                x,
                y,
                data.std,
                levels=levels,
                colors="white",
                linewidths=0.7,
            )
            axis.clabel(contours, inline=True, fontsize=7, fmt="%.3g degC")
    axis.set_title(f"Mean temperature with uncertainty contours (N={data.count})")
    axis.set_xlabel("x [km]")
    axis.set_ylabel("y [km]")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("mean temperature [degC]")
    overlay_path = root / f"{prefix}_temperature_mean_std_overlay.png"
    figure.savefig(overlay_path, dpi=210, bbox_inches="tight")
    plt.close(figure)
    saved["temperature_mean_std_overlay"] = overlay_path

    if background_temperature is not None:
        delta_mean = data.mean.astype(np.float64) - float(background_temperature)
        magnitude = max(abs(float(np.min(delta_mean))), abs(float(np.max(delta_mean))))
        if magnitude == 0.0:
            magnitude = 1.0
        saved["delta_temperature_mean"] = _save_map(
            delta_mean,
            root / f"{prefix}_delta_temperature_mean.png",
            title=f"Monte Carlo mean temperature change (N={data.count})",
            colorbar_label="Delta T [degC]",
            cell_size_m=cell_size_m,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-magnitude, vcenter=0.0, vmax=magnitude),
        )

    if data.exceedance_probabilities is not None:
        for index, threshold in enumerate(data.exceedance_thresholds):
            token = _threshold_token(threshold)
            key = f"probability_deltaT_ge_{token}"
            saved[key] = _save_map(
                data.exceedance_probabilities[index],
                root / f"{prefix}_{key}.png",
                title=(
                    f"P(Delta T >= {threshold:g} degC), empirical Monte Carlo "
                    f"(N={data.count})"
                ),
                colorbar_label="probability",
                cell_size_m=cell_size_m,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )

    return saved


def plot_monte_carlo_archive(
    source: str | Path,
    directory: str | Path,
    *,
    prefix: str | None = None,
    cell_size_m: float = 5.0,
    background_temperature: float | None = None,
) -> dict[str, Path]:
    """Load a saved MC archive and write all available UQ plots."""

    source_path = Path(source).expanduser().resolve()
    data = load_monte_carlo_plot_data(source_path)
    return save_monte_carlo_uq_plots(
        data,
        directory,
        prefix=source_path.stem if prefix is None else prefix,
        cell_size_m=cell_size_m,
        background_temperature=background_temperature,
    )
