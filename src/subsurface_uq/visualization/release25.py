from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
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
        raise ValueError(f"{name} must be 2-D after removing singleton leading axes, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return np.asarray(array, dtype=np.float32)


def _save_field(
    field: np.ndarray,
    destination: Path,
    *,
    title: str,
    colorbar_label: str,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(field, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("x cell")
    ax.set_ylabel("y cell")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def save_release25_output_plots(
    outputs: Mapping[str, Tensor],
    directory: str | Path,
    *,
    prefix: str = "release25",
    include_velocity: bool = True,
) -> dict[str, Path]:
    """Save diagnostic maps from one deterministic release25 LGCNN forward pass.

    Expected keys are the standard ``Release25RuntimeAdapter.predict`` outputs:
    ``velocity`` with shape [2,H,W], ``streamlines`` with shape [2,H,W]
    (central and outer streamline feature fields), and ``temperature`` with
    shape [1,H,W] or [H,W].

    The streamline images visualize the actual rasterized feature fields passed
    to the Step-3 CNN. They are not geometric polyline renderings of individual
    ODE trajectories.
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
    )
    saved["streamline_outer"] = _save_field(
        outer,
        root / f"{prefix}_streamline_outer.png",
        title="LGCNN outer streamline feature",
        colorbar_label="streamline feature",
    )
    saved["temperature"] = _save_field(
        temperature,
        root / f"{prefix}_temperature.png",
        title="LGCNN predicted temperature",
        colorbar_label="temperature [degC]",
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
            colorbar_label="velocity x",
        )
        saved["velocity_y"] = _save_field(
            vy,
            root / f"{prefix}_velocity_y.png",
            title="LGCNN predicted velocity y",
            colorbar_label="velocity y",
        )
        saved["velocity_magnitude"] = _save_field(
            magnitude,
            root / f"{prefix}_velocity_magnitude.png",
            title="LGCNN predicted velocity magnitude",
            colorbar_label="velocity magnitude",
        )

    return saved
