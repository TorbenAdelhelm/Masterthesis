from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


def _field_map(
    values: Array,
    destination: Path,
    *,
    title: str,
    label: str,
    cell_size_m: float,
) -> Path:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 2 or not np.all(np.isfinite(field)):
        raise ValueError("RQ1 map must be a finite 2-D field")
    h, w = field.shape
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(
        field,
        origin="lower",
        extent=(0.0, w * cell_size_m / 1000.0, 0.0, h * cell_size_m / 1000.0),
        aspect="equal",
    )
    axis.set_title(title)
    axis.set_xlabel("x [km]")
    axis.set_ylabel("y [km]")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(label)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_rq1_temperature_fields(
    fields_directory: str | Path,
    figures_directory: str | Path,
    *,
    cell_size_m: float,
    title_prefix: str,
) -> dict[str, Path]:
    fields = Path(fields_directory).expanduser().resolve()
    figures = Path(figures_directory).expanduser().resolve()
    figures.mkdir(parents=True, exist_ok=True)
    specs = (
        ("temperature_mean", "Mean temperature", "temperature [degC]"),
        ("temperature_std", "Temperature standard deviation", "standard deviation [degC]"),
        ("temperature_width90", "90% empirical interval width", "q95 - q05 [degC]"),
    )
    saved: dict[str, Path] = {}
    for key, title, label in specs:
        values = np.load(fields / f"{key}.npy", mmap_mode="r")
        saved[key] = _field_map(
            values,
            figures / f"{key}.png",
            title=f"{title_prefix}: {title}",
            label=label,
            cell_size_m=cell_size_m,
        )
    return saved


def plot_rq1_qoi_distributions(
    *,
    mean_anomaly: Array | None,
    receptor_values: Array,
    receptor_indices: Sequence[tuple[int, int]],
    directory: str | Path,
    title_prefix: str,
) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    series: list[tuple[str, Array, str]] = []
    if mean_anomaly is not None:
        series.append(
            (
                "mean_anomaly",
                np.asarray(mean_anomaly),
                "mean temperature anomaly [degC]",
            )
        )
    receptor_values = np.asarray(receptor_values)
    for index, location in enumerate(receptor_indices):
        series.append(
            (
                f"receptor_{index + 1:03d}",
                receptor_values[:, index],
                f"temperature at receptor {location} [degC]",
            )
        )

    for key, values, label in series:
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        axis.hist(np.asarray(values, dtype=np.float64), bins="auto")
        axis.set_title(f"{title_prefix}: {key.replace('_', ' ')}")
        axis.set_xlabel(label)
        axis.set_ylabel("count")
        destination = root / f"{key}_distribution.png"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        saved[key] = destination
    return saved


def plot_rq1_convergence(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
) -> dict[str, Path]:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for statistic, label in (("e_mu", "mean-field RMS error [degC]"), ("e_sigma", "std-field RMS error [degC]")):
        figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        variants = sorted(
            {
                str(row["variant"])
                for row in rows
                if row.get("statistic") == statistic and row.get("repetition") == "main"
            }
        )
        for variant in variants:
            selected = [
                row
                for row in rows
                if row.get("variant") == variant
                and row.get("statistic") == statistic
                and row.get("repetition") == "main"
            ]
            selected.sort(key=lambda row: int(row["budget"]))
            axis.plot(
                [int(row["budget"]) for row in selected],
                [float(row["error"]) for row in selected],
                marker="o",
                label=variant,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("LGCNN evaluations N")
        axis.set_ylabel(label)
        axis.set_title(f"RQ1 convergence: {statistic}")
        if variants:
            axis.legend()
        destination = root / f"convergence_{statistic}.png"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        saved[statistic] = destination
    return saved


def plot_mc_rqmc_comparison(
    rows: Sequence[Mapping[str, object]],
    directory: str | Path,
) -> Path:
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for statistic in ("e_mu", "e_sigma"):
        selected = [row for row in rows if row.get("statistic") == statistic]
        selected.sort(key=lambda row: int(row["budget"]))
        if not selected:
            continue
        axis.plot(
            [int(row["budget"]) for row in selected],
            [float(row["mc_rmse"]) for row in selected],
            marker="o",
            label=f"MC {statistic}",
        )
        axis.plot(
            [int(row["budget"]) for row in selected],
            [float(row["rqmc_rmse"]) for row in selected],
            marker="s",
            label=f"RQMC {statistic}",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("LGCNN evaluations N")
    axis.set_ylabel("RMSE against empirical MC reference [degC]")
    axis.set_title("Conditional GRF: MC vs randomized QMC")
    axis.legend()
    destination = root / "mc_vs_rqmc_rmse.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
