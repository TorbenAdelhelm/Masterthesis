from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy import ndimage
import torch

Tensor = torch.Tensor


def _to_2d_array(value: Tensor | np.ndarray, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D after removing singleton leading axes, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _center_crop_array(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError(f"center crop expects a 2-D field, got {values.shape}")
    h, w = values.shape
    th, tw = target_shape
    if th > h or tw > w:
        raise ValueError(f"cannot center-crop {values.shape} to {target_shape}")
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return values[y0 : y0 + th, x0 : x0 + tw]


def _extent_km(shape: tuple[int, int], cell_size_m: float) -> tuple[float, float, float, float]:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    h, w = shape
    return (0.0, w * cell_size_m / 1000.0, 0.0, h * cell_size_m / 1000.0)


def _save_field(
    field: np.ndarray,
    destination: Path,
    *,
    title: str,
    colorbar_label: str,
    cell_size_m: float = 5.0,
    diverging_zero: bool = False,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    kwargs: dict[str, object] = {}
    if diverging_zero and float(np.nanmin(field)) < 0.0 < float(np.nanmax(field)):
        limit = max(abs(float(np.nanmin(field))), abs(float(np.nanmax(field))))
        kwargs["norm"] = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        kwargs["cmap"] = "coolwarm"
    image = ax.imshow(
        field,
        origin="lower",
        extent=_extent_km(field.shape, cell_size_m),
        aspect="equal",
        **kwargs,
    )
    ax.set_title(title)
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return destination


def _heat_pump_centers(
    material_id: Tensor | np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Return one [y,x] centroid per connected heat-pump region in the target crop."""

    material = _to_2d_array(material_id, "material id")
    if material.shape != target_shape:
        material = _center_crop_array(material, target_shape)
    mask = np.isclose(material, 2.0)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    centers = ndimage.center_of_mass(mask, labels, range(1, count + 1))
    return np.asarray(centers, dtype=np.float64)


def save_release25_output_plots(
    outputs: Mapping[str, Tensor],
    directory: str | Path,
    *,
    prefix: str = "release25",
    include_velocity: bool = True,
    cell_size_m: float = 5.0,
) -> dict[str, Path]:
    """Save diagnostic maps from one deterministic release25 LGCNN forward pass.

    The streamline images visualize the rasterized feature fields that are passed
    to the Step-3 CNN. They are not geometric polyline renderings of individual
    ODE trajectories. ``cell_size_m=5`` matches the release25 synthetic baseline;
    callers can override it for another grid.
    """

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if "streamlines" not in outputs or "temperature" not in outputs:
        raise KeyError("release25 outputs must contain 'streamlines' and 'temperature'")

    streamlines = outputs["streamlines"]
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.as_tensor(streamlines)
    if streamlines.ndim != 3 or streamlines.shape[0] != 2:
        raise ValueError(
            "streamlines must have shape [2,H,W] with [center, outer], "
            f"got {tuple(streamlines.shape)}"
        )

    center = _to_2d_array(streamlines[0], "central streamline field")
    outer = _to_2d_array(streamlines[1], "outer streamline field")
    temperature = _to_2d_array(outputs["temperature"], "temperature")

    saved: dict[str, Path] = {}
    saved["streamline_center"] = _save_field(
        center,
        root / f"{prefix}_streamline_center.png",
        title="LGCNN central streamline feature",
        colorbar_label="streamline feature",
        cell_size_m=cell_size_m,
    )
    saved["streamline_outer"] = _save_field(
        outer,
        root / f"{prefix}_streamline_outer.png",
        title="LGCNN outer streamline feature",
        colorbar_label="streamline feature",
        cell_size_m=cell_size_m,
    )
    saved["temperature"] = _save_field(
        temperature,
        root / f"{prefix}_temperature.png",
        title="LGCNN predicted temperature",
        colorbar_label="temperature [degC]",
        cell_size_m=cell_size_m,
    )

    if include_velocity:
        if "velocity" not in outputs:
            raise KeyError("include_velocity=True requires a 'velocity' output")
        velocity = outputs["velocity"]
        if not isinstance(velocity, torch.Tensor):
            velocity = torch.as_tensor(velocity)
        if velocity.ndim != 3 or velocity.shape[0] != 2:
            raise ValueError(f"velocity must have shape [2,H,W], got {tuple(velocity.shape)}")
        vx = _to_2d_array(velocity[0], "velocity x")
        vy = _to_2d_array(velocity[1], "velocity y")
        magnitude = np.sqrt(vx.astype(np.float64) ** 2 + vy.astype(np.float64) ** 2).astype(
            np.float32
        )
        saved["velocity_x"] = _save_field(
            vx,
            root / f"{prefix}_velocity_x.png",
            title="LGCNN predicted velocity x",
            colorbar_label="velocity x [release25 physical units]",
            cell_size_m=cell_size_m,
            diverging_zero=True,
        )
        saved["velocity_y"] = _save_field(
            vy,
            root / f"{prefix}_velocity_y.png",
            title="LGCNN predicted velocity y",
            colorbar_label="velocity y [release25 physical units]",
            cell_size_m=cell_size_m,
            diverging_zero=True,
        )
        saved["velocity_magnitude"] = _save_field(
            magnitude,
            root / f"{prefix}_velocity_magnitude.png",
            title="LGCNN predicted velocity magnitude",
            colorbar_label="velocity magnitude [release25 physical units]",
            cell_size_m=cell_size_m,
        )

    return saved


def save_release25_overlay_plots(
    outputs: Mapping[str, Tensor],
    material_id: Tensor | np.ndarray,
    directory: str | Path,
    *,
    prefix: str = "release25",
    cell_size_m: float = 5.0,
    streamline_threshold: float = 0.05,
) -> dict[str, Path]:
    """Save combined temperature/streamline/heat-pump figures for one run.

    Streamline contours are obtained from the actual rasterized central feature
    field used by CNN3. Heat-pump markers are connected-component centroids of
    Material-ID==2 regions after alignment to the temperature output.
    """

    if not (0.0 <= streamline_threshold <= 1.0):
        raise ValueError("streamline_threshold must lie in [0, 1]")
    if "streamlines" not in outputs or "temperature" not in outputs:
        raise KeyError("release25 outputs must contain 'streamlines' and 'temperature'")

    streamlines = outputs["streamlines"]
    if not isinstance(streamlines, torch.Tensor):
        streamlines = torch.as_tensor(streamlines)
    if streamlines.ndim != 3 or streamlines.shape[0] != 2:
        raise ValueError("streamlines must have shape [2,H,W]")

    temperature = _to_2d_array(outputs["temperature"], "temperature")
    center = _to_2d_array(streamlines[0], "central streamline field")
    if center.shape != temperature.shape:
        center = _center_crop_array(center, temperature.shape)
    pumps = _heat_pump_centers(material_id, temperature.shape)
    extent = _extent_km(temperature.shape, cell_size_m)

    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def base_figure(title: str):
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(
            temperature,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("temperature [degC]")
        return fig, ax

    saved: dict[str, Path] = {}
    mask = center >= streamline_threshold
    x = np.linspace(extent[0], extent[1], center.shape[1])
    y = np.linspace(extent[2], extent[3], center.shape[0])

    fig, ax = base_figure("LGCNN temperature with central streamline feature")
    if np.any(mask):
        ax.contour(x, y, mask.astype(np.float32), levels=[0.5], colors="white", linewidths=0.6)
    path = root / f"{prefix}_temperature_streamlines_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved["temperature_streamlines"] = path

    fig, ax = base_figure("LGCNN temperature with heat-pump locations")
    if len(pumps):
        ax.scatter(
            pumps[:, 1] * cell_size_m / 1000.0,
            pumps[:, 0] * cell_size_m / 1000.0,
            s=20,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
            label="heat pump",
        )
        ax.legend(loc="upper right")
    path = root / f"{prefix}_temperature_heatpumps_overlay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved["temperature_heatpumps"] = path

    fig, ax = base_figure("LGCNN temperature, streamlines, and heat pumps")
    if np.any(mask):
        ax.contour(x, y, mask.astype(np.float32), levels=[0.5], colors="white", linewidths=0.5)
    if len(pumps):
        ax.scatter(
            pumps[:, 1] * cell_size_m / 1000.0,
            pumps[:, 0] * cell_size_m / 1000.0,
            s=22,
            facecolors="none",
            edgecolors="red",
            linewidths=0.9,
            label="heat pump",
        )
        ax.legend(loc="upper right")
    path = root / f"{prefix}_temperature_streamlines_heatpumps.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved["temperature_streamlines_heatpumps"] = path

    return saved
