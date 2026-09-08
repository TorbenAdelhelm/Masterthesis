from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..spatial import center_crop_2d, extent_km, heat_pump_centers, to_2d_array
from ..surrogates.release25_runtime import Release25Runtime

Array = np.ndarray
Tensor = torch.Tensor


def build_release25_validation_overlay_context(
    *,
    release25_repo: str | Path,
    cnn1_dir: str | Path,
    cnn2_dir: str | Path,
    prepared_pki_dir: str | Path,
    run_id: str,
    target_shape: tuple[int, int],
    device: str | torch.device = "cpu",
    random_k: bool = True,
    streamline_method: str = "RK45",
) -> dict[str, Array]:
    """Regenerate central streamlines and heat-pump locations for validation plots."""

    runtime = Release25Runtime.from_paths(
        release25_repo=release25_repo,
        cnn1_dir=cnn1_dir,
        cnn2_dir=cnn2_dir,
        prepared_pki_dir=prepared_pki_dir,
        run_id=run_id,
        device=device,
        random_k=random_k,
        streamline_method=streamline_method,
    )
    outputs = runtime.deterministic()
    streamlines = outputs["streamlines"]
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.as_tensor(streamlines)
    if streamlines.ndim != 3 or streamlines.shape[0] != 2:
        raise ValueError(f"streamlines must have shape [2,H,W], got {tuple(streamlines.shape)}")

    center = to_2d_array(streamlines[0], "central streamline field")
    if center.shape != target_shape:
        center = center_crop_2d(center, target_shape)
    pumps = heat_pump_centers(runtime.scenario.fixed.material_id, target_shape)
    return {
        "central_streamline": center,
        "heat_pump_centers": pumps,
    }


def _save_overlay(
    field: Array,
    central_streamline: Array,
    heat_pump_centers_array: Array,
    destination: Path,
    *,
    title: str,
    colorbar_label: str,
    cell_size_m: float,
    streamline_threshold: float,
) -> Path:
    values = to_2d_array(field, "overlay field")
    center = to_2d_array(central_streamline, "central streamline field")
    if center.shape != values.shape:
        center = center_crop_2d(center, values.shape)
    if not (0.0 <= streamline_threshold <= 1.0):
        raise ValueError("streamline_threshold must lie in [0, 1]")

    extent = extent_km(values.shape, cell_size_m)
    mask = center >= streamline_threshold
    x = np.linspace(extent[0], extent[1], center.shape[1])
    y = np.linspace(extent[2], extent[3], center.shape[0])

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, origin="lower", extent=extent, aspect="equal")
    if np.any(mask):
        ax.contour(x, y, mask.astype(np.float32), levels=[0.5], colors="white", linewidths=0.55)
    if len(heat_pump_centers_array):
        ax.scatter(
            heat_pump_centers_array[:, 1] * cell_size_m / 1000.0,
            heat_pump_centers_array[:, 0] * cell_size_m / 1000.0,
            s=22,
            facecolors="none",
            edgecolors="red",
            linewidths=0.9,
            label="heat pump",
        )
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_validation_overlay_plots(
    *,
    prediction: Array,
    reference: Array,
    absolute_error: Array,
    overlay_context: Mapping[str, Array],
    directory: str | Path,
    prefix: str,
    cell_size_m: float = 5.0,
    streamline_threshold: float = 0.05,
) -> dict[str, Path]:
    """Save aligned prediction/reference/error maps with streamlines and heat pumps."""

    center = np.asarray(overlay_context["central_streamline"], dtype=np.float32)
    pumps = np.asarray(overlay_context["heat_pump_centers"], dtype=np.float64)
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    saved["prediction_overlay"] = _save_overlay(
        prediction,
        center,
        pumps,
        root / f"{prefix}_prediction_streamlines_heatpumps.png",
        title="Predicted temperature, streamlines, and heat pumps",
        colorbar_label="temperature [degC]",
        cell_size_m=cell_size_m,
        streamline_threshold=streamline_threshold,
    )
    saved["reference_overlay"] = _save_overlay(
        reference,
        center,
        pumps,
        root / f"{prefix}_reference_streamlines_heatpumps.png",
        title="Reference temperature, streamlines, and heat pumps",
        colorbar_label="temperature [degC]",
        cell_size_m=cell_size_m,
        streamline_threshold=streamline_threshold,
    )
    saved["absolute_error_overlay"] = _save_overlay(
        absolute_error,
        center,
        pumps,
        root / f"{prefix}_absolute_error_streamlines_heatpumps.png",
        title="Absolute temperature error, streamlines, and heat pumps",
        colorbar_label="absolute error [degC]",
        cell_size_m=cell_size_m,
        streamline_threshold=streamline_threshold,
    )
    return saved
