from __future__ import annotations

import numpy as np
from scipy import ndimage
import torch

Array = np.ndarray
Tensor = torch.Tensor


def to_2d_array(value: Tensor | Array, name: str, *, dtype=np.float32) -> Array:
    """Return one finite 2-D NumPy field after removing singleton leading axes."""

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
    return np.asarray(array, dtype=dtype)


def center_crop_2d(field: Array, target_shape: tuple[int, int]) -> Array:
    """Return the centered ``target_shape`` window of one 2-D field."""

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError(f"center crop expects a 2-D field, got {values.shape}")
    th, tw = (int(target_shape[0]), int(target_shape[1]))
    h, w = values.shape
    if th <= 0 or tw <= 0:
        raise ValueError("target shape must be positive")
    if th > h or tw > w:
        raise ValueError(f"cannot center-crop {values.shape} to {(th, tw)}")
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return values[y0 : y0 + th, x0 : x0 + tw]


def extent_km(
    shape: tuple[int, int], cell_size_m: float
) -> tuple[float, float, float, float]:
    """Return an ``imshow`` extent in kilometres for a regular cell-centered grid."""

    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    h, w = (int(shape[0]), int(shape[1]))
    if h <= 0 or w <= 0:
        raise ValueError("shape entries must be positive")
    return (0.0, w * cell_size_m / 1000.0, 0.0, h * cell_size_m / 1000.0)


def heat_pump_centers(
    material_id: Tensor | Array,
    target_shape: tuple[int, int],
) -> Array:
    """Return one ``[y,x]`` centroid per connected ``Material ID == 2`` region."""

    material = to_2d_array(material_id, "material id")
    if material.shape != target_shape:
        material = center_crop_2d(material, target_shape)
    mask = np.isclose(material, 2.0)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    centers = ndimage.center_of_mass(mask, labels, range(1, count + 1))
    return np.asarray(centers, dtype=np.float64)
