from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .training_provenance import RealisticRunMetadata

Array = np.ndarray

HISTORICAL_GENERATOR_REPOSITORY = "JuliaPelzer/Dataset-generation-with-Pflotran"
HISTORICAL_GENERATOR_BRANCH = "windowed_real_vals_v2"
HISTORICAL_GENERATOR_COMMIT = "c13ccea3fccd8180ea682ea30d5622fac0c333eb"
# The exact parent raster supplied with the historical data is in the explicit
# Easting/Northing form of DHDN / 3-degree Gauss-Kruger zone 4.
HISTORICAL_PARENT_CRS = "EPSG:5678"
HISTORICAL_SOURCE_RESOLUTION_M = 20.0
HISTORICAL_PARENT_SHAPE_YX = (3797, 3583)
HISTORICAL_PARENT_SHA256 = (
    "6d50f2c6f9b96e3136fe77ad177ef4f70cc2d4cfc632f33ff655e59bdcc096ee"
)
HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR = 7.5e6
HISTORICAL_PARENT_HYDRAULIC_CONDUCTIVITY_FILENAME = (
    "Hydraulic_conductivity_20m_resolution.tif"
)


@dataclass(frozen=True)
class HistoricalReconstructionResult:
    run_name: str
    reconstructed_permeability_m2: Array
    source_window_hydraulic_conductivity_m_s: Array
    source_x_indices: Array
    source_y_indices: Array


def historical_hydraulic_conductivity_to_training_permeability(
    hydraulic_conductivity_m_s: Array,
) -> Array:
    """Apply the conversion used by the historical realistic-data generator."""

    values = np.asarray(hydraulic_conductivity_m_s, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("hydraulic conductivity must be finite and positive")
    return values / HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR


def historical_rotated_source_indices(
    metadata: RealisticRunMetadata,
    *,
    window_shape_source_cells: tuple[int, int],
) -> tuple[Array, Array]:
    """Reproduce ``calc_rotated_box`` from the historical generator.

    Historical semantics recovered from commit c13ccea:

    - YAML start position is ``start_pos * orig_resolution``;
    - the extraction mesh starts at (0, 0);
    - rotation uses ``deg2rad(rotation_angle + 90)``;
    - the rotated floating coordinates are translated by the start cell;
    - source values are indexed after NumPy ``astype(int)`` truncation.
    """

    resolution = float(metadata.original_resolution_m)
    start_x = float(metadata.start_position_m[0]) / resolution
    start_y = float(metadata.start_position_m[1]) / resolution
    ny_like = int(window_shape_source_cells[0])
    nx_like = int(window_shape_source_cells[1])
    if ny_like <= 0 or nx_like <= 0:
        raise ValueError("window_shape_source_cells must be positive")
    x, y = np.meshgrid(
        np.arange(0, nx_like, dtype=np.float64),
        np.arange(0, ny_like, dtype=np.float64),
    )
    angle = np.deg2rad(float(metadata.rotation_angle_deg) + 90.0)
    rot_x = x * np.cos(angle) - y * np.sin(angle) + start_x
    rot_y = x * np.sin(angle) + y * np.cos(angle) + start_y
    return rot_x.astype(np.int64), rot_y.astype(np.int64)


def historical_extract_source_window(
    parent_hydraulic_conductivity_m_s: Array,
    metadata: RealisticRunMetadata,
    *,
    window_shape_source_cells: tuple[int, int],
) -> tuple[Array, Array, Array]:
    parent = np.asarray(parent_hydraulic_conductivity_m_s, dtype=np.float64)
    if parent.ndim != 2:
        raise ValueError("parent hydraulic-conductivity map must be 2-D")
    x_idx, y_idx = historical_rotated_source_indices(
        metadata,
        window_shape_source_cells=window_shape_source_cells,
    )
    if (
        np.any(x_idx < 0)
        or np.any(y_idx < 0)
        or np.any(x_idx >= parent.shape[1])
        or np.any(y_idx >= parent.shape[0])
    ):
        raise ValueError("historical window extends outside parent map")
    window = parent[y_idx, x_idx]
    if not np.all(np.isfinite(window)) or np.any(window <= 0.0):
        raise ValueError("historical source window contains invalid hydraulic conductivity")
    return window, x_idx, y_idx


def historical_interpolate_window(
    source_window: Array,
    *,
    original_resolution_m: float,
    destination_resolution_m: float,
    destination_shape_yx: tuple[int, int],
) -> Array:
    """Reproduce the historical RegularGridInterpolator sampling convention.

    The source code used the first array axis as the first interpolation coordinate
    and queried the interpolator with PFLOTRAN cell-center ``(x, y)`` positions.
    The TODO in the original code explicitly notes that no additional +0.5 source-
    pixel correction was applied.
    """

    source = np.asarray(source_window, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("source_window must be 2-D")
    original_resolution_m = float(original_resolution_m)
    destination_resolution_m = float(destination_resolution_m)
    ny, nx = (int(destination_shape_yx[0]), int(destination_shape_yx[1]))
    if original_resolution_m <= 0.0 or destination_resolution_m <= 0.0:
        raise ValueError("resolutions must be positive")
    if ny <= 0 or nx <= 0:
        raise ValueError("destination_shape_yx must be positive")

    axis0 = np.arange(source.shape[0], dtype=np.float64) * original_resolution_m
    axis1 = np.arange(source.shape[1], dtype=np.float64) * original_resolution_m
    interpolator = RegularGridInterpolator(
        (axis0, axis1),
        source,
        bounds_error=False,
        fill_value=None,
    )
    # Historical mesh cells are generated with x varying fastest, then y.
    x = (np.arange(nx, dtype=np.float64) + 0.5) * destination_resolution_m
    y = (np.arange(ny, dtype=np.float64) + 0.5) * destination_resolution_m
    yy, xx = np.meshgrid(y, x, indexing="ij")
    query = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    values = interpolator(query)
    return values.reshape(ny, nx)


def reconstruct_historical_training_permeability(
    parent_hydraulic_conductivity_m_s: Array,
    metadata: RealisticRunMetadata,
    *,
    destination_shape_yx: tuple[int, int],
    destination_resolution_m: float = 5.0,
) -> HistoricalReconstructionResult:
    """Rebuild one historical training permeability field from its parent map."""

    physical_size_y = int(destination_shape_yx[0]) * float(destination_resolution_m)
    physical_size_x = int(destination_shape_yx[1]) * float(destination_resolution_m)
    source_shape = (
        int(round(physical_size_y / metadata.original_resolution_m)),
        int(round(physical_size_x / metadata.original_resolution_m)),
    )
    if not np.isclose(source_shape[0] * metadata.original_resolution_m, physical_size_y):
        raise ValueError("destination y-size is not compatible with original resolution")
    if not np.isclose(source_shape[1] * metadata.original_resolution_m, physical_size_x):
        raise ValueError("destination x-size is not compatible with original resolution")

    source_kh, x_idx, y_idx = historical_extract_source_window(
        parent_hydraulic_conductivity_m_s,
        metadata,
        window_shape_source_cells=source_shape,
    )
    source_perm = historical_hydraulic_conductivity_to_training_permeability(source_kh)
    reconstructed = historical_interpolate_window(
        source_perm,
        original_resolution_m=metadata.original_resolution_m,
        destination_resolution_m=destination_resolution_m,
        destination_shape_yx=destination_shape_yx,
    )
    return HistoricalReconstructionResult(
        run_name=metadata.run_name,
        reconstructed_permeability_m2=np.asarray(reconstructed, dtype=np.float32),
        source_window_hydraulic_conductivity_m_s=np.asarray(source_kh, dtype=np.float32),
        source_x_indices=x_idx,
        source_y_indices=y_idx,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_parent_hydraulic_conductivity_tif(
    path: str | Path,
) -> tuple[Array, dict[str, object]]:
    """Load the historical parent GeoTIFF while retaining provenance metadata.

    Nodata values are returned as NaN. This mirrors the role of the historical
    ``align_holes`` preprocessing for the hydraulic-conductivity channel while
    preserving the original positive values used in valid simulation windows.
    """

    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "Historical parent-map reconstruction requires rasterio. Install "
            "'subsurface-uq[geostat]' before using "
            "--parent-hydraulic-conductivity-tif."
        ) from exc
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    sha256 = _sha256_file(source)
    with rasterio.open(source) as dataset:
        masked = dataset.read(1, masked=True)
        values = np.asarray(masked.filled(np.nan), dtype=np.float64)
        crs = None if dataset.crs is None else dataset.crs.to_string()
        transform = tuple(float(v) for v in dataset.transform)[:6]
        bounds = {
            "left": float(dataset.bounds.left),
            "bottom": float(dataset.bounds.bottom),
            "right": float(dataset.bounds.right),
            "top": float(dataset.bounds.top),
        }
        resolution = (float(abs(dataset.res[0])), float(abs(dataset.res[1])))
        nodata = None if dataset.nodata is None else float(dataset.nodata)
    return values, {
        "path": str(source),
        "sha256": sha256,
        "shape_yx": [int(values.shape[0]), int(values.shape[1])],
        "crs": crs,
        "affine_transform": list(transform),
        "bounds": bounds,
        "resolution_m": list(resolution),
        "nodata": nodata,
        "historical_expected_crs": HISTORICAL_PARENT_CRS,
        "historical_expected_shape_yx": list(HISTORICAL_PARENT_SHAPE_YX),
        "historical_expected_sha256": HISTORICAL_PARENT_SHA256,
        "historical_expected_filename": HISTORICAL_PARENT_HYDRAULIC_CONDUCTIVITY_FILENAME,
    }


def sample_parent_hydraulic_conductivity_at_projected_points(
    parent_hydraulic_conductivity_m_s: Array,
    parent_metadata: dict[str, object],
    x_m: Array,
    y_m: Array,
    *,
    method: str = "linear",
) -> tuple[Array, Array]:
    """Sample the historical parent raster at projected measurement coordinates.

    The uploaded historical raster is north-up with an affine transform in
    EPSG:5678. Sampling uses pixel-centre coordinates and either bilinear
    (``method='linear'``) or nearest-neighbour interpolation. Returned values
    remain hydraulic conductivity in m/s.
    """

    parent = np.asarray(parent_hydraulic_conductivity_m_s, dtype=np.float64)
    if parent.ndim != 2:
        raise ValueError("parent hydraulic-conductivity map must be 2-D")
    transform = tuple(float(v) for v in parent_metadata["affine_transform"])
    if len(transform) != 6:
        raise ValueError("parent affine transform must contain six coefficients")
    a, b, c, d, e, f = transform
    if not np.isclose(b, 0.0, atol=1.0e-12) or not np.isclose(d, 0.0, atol=1.0e-12):
        raise ValueError("rotated/sheared parent rasters are not supported")
    if np.isclose(a, 0.0) or np.isclose(e, 0.0):
        raise ValueError("invalid parent affine transform")
    method = str(method).lower()
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")

    x = np.asarray(x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(y_m, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x_m and y_m must be finite aligned one-dimensional arrays")

    col = (x - c) / a - 0.5
    row = (y - f) / e - 0.5
    inside = (
        (row >= 0.0)
        & (row <= parent.shape[0] - 1)
        & (col >= 0.0)
        & (col <= parent.shape[1] - 1)
    )
    sampled = np.full(x.shape, np.nan, dtype=np.float64)
    if np.any(inside):
        interpolator = RegularGridInterpolator(
            (
                np.arange(parent.shape[0], dtype=np.float64),
                np.arange(parent.shape[1], dtype=np.float64),
            ),
            parent,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        sampled[inside] = interpolator(
            np.column_stack((row[inside], col[inside]))
        )
    valid = inside & np.isfinite(sampled) & (sampled > 0.0)
    return sampled, valid
