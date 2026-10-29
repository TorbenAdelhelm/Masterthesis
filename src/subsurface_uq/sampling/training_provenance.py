from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

from .geospatial import LGCNNDomainGeoreference, orient_raw_field

Array = np.ndarray


@dataclass(frozen=True)
class RealisticRunMetadata:
    """Historical extraction metadata stored beside one DaRUS-5065 run."""

    run_name: str
    original_resolution_m: float
    rotation_angle_deg: float
    start_position_m: tuple[float, float]
    source_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "run_name": self.run_name,
            "original_resolution_m": self.original_resolution_m,
            "rotation_angle_deg": self.rotation_angle_deg,
            "start_position_m": [
                float(self.start_position_m[0]),
                float(self.start_position_m[1]),
            ],
            "source_path": self.source_path,
        }


def load_realistic_run_metadata(
    dataset_root: str | Path,
    run_name: str,
    *,
    filename: str = "realistic_params.yaml",
) -> RealisticRunMetadata:
    """Load the per-run historical permeability-map extraction metadata.

    The keys intentionally follow the DaRUS YAML verbatim. No geometric meaning
    is assigned to ``start position`` or ``rotation angle`` here because the
    historical crop/rotation convention must be validated rather than guessed.
    """

    root = Path(dataset_root).expanduser().resolve()
    source = root / str(run_name) / filename
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain a YAML mapping")
    required = (
        "orig resolution [m]",
        "rotation angle [°]",
        "start position [m]",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{source} is missing required keys: {missing}")
    resolution = float(payload["orig resolution [m]"])
    rotation = float(payload["rotation angle [°]"])
    start_raw = payload["start position [m]"]
    if not isinstance(start_raw, (list, tuple)) or len(start_raw) != 2:
        raise ValueError(f"{source}: 'start position [m]' must contain two numbers")
    start = (float(start_raw[0]), float(start_raw[1]))
    values = np.asarray([resolution, rotation, *start], dtype=np.float64)
    if not np.all(np.isfinite(values)) or resolution <= 0.0:
        raise ValueError(f"{source} contains invalid/non-finite metadata")
    return RealisticRunMetadata(
        run_name=str(run_name),
        original_resolution_m=resolution,
        rotation_angle_deg=rotation,
        start_position_m=start,
        source_path=str(source),
    )


def discover_realistic_runs(
    dataset_root: str | Path,
    *,
    metadata_filename: str = "realistic_params.yaml",
    require_h5: bool = True,
) -> tuple[str, ...]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    runs: list[tuple[int, str]] = []
    for folder in root.iterdir():
        if not folder.is_dir() or not folder.name.startswith("RUN_"):
            continue
        if not (folder / metadata_filename).is_file():
            continue
        if require_h5 and not (folder / "pflotran.h5").is_file():
            continue
        try:
            number = int(folder.name.removeprefix("RUN_"))
        except ValueError:
            continue
        runs.append((number, folder.name))
    runs.sort(key=lambda item: item[0])
    if not runs:
        raise ValueError(
            f"no RUN_* folders with {metadata_filename!r}"
            + (" and pflotran.h5" if require_h5 else "")
            + f" found below {root}"
        )
    return tuple(name for _, name in runs)


def sample_training_field_at_projected_points(
    field: Array,
    georeference: LGCNNDomainGeoreference,
    x_m: Array,
    y_m: Array,
) -> tuple[Array, Array]:
    """Bilinearly sample a georeferenced training field in log10 permeability.

    Returns ``(sampled_permeability_m2, inside_mask)``. Sampling is performed in
    log10-space because permeability varies over orders of magnitude.
    """

    raw = np.asarray(field, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("field must be a finite positive 2-D array")
    geo = np.asarray(orient_raw_field(raw, georeference.transform), dtype=np.float64)
    if geo.shape != (georeference.ny, georeference.nx):
        raise ValueError(
            "oriented field shape differs from georeference: "
            f"{geo.shape} != {(georeference.ny, georeference.nx)}"
        )
    x = np.asarray(x_m, dtype=np.float64).reshape(-1)
    y = np.asarray(y_m, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x_m and y_m must be finite aligned one-dimensional arrays")
    row, col = georeference.xy_to_fractional_indices(x, y)
    inside = (
        (row >= 0.0)
        & (row <= geo.shape[0] - 1)
        & (col >= 0.0)
        & (col <= geo.shape[1] - 1)
    )
    sampled = np.full(x.shape, np.nan, dtype=np.float64)
    if np.any(inside):
        interpolator = RegularGridInterpolator(
            (
                np.arange(geo.shape[0], dtype=np.float64),
                np.arange(geo.shape[1], dtype=np.float64),
            ),
            np.log10(geo),
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        sampled_log = interpolator(np.column_stack((row[inside], col[inside])))
        sampled[inside] = np.power(10.0, sampled_log)
    return sampled, inside


def measurement_training_consistency(
    field: Array,
    georeference: LGCNNDomainGeoreference,
    measurement_x_m: Array,
    measurement_y_m: Array,
    measurement_permeability_m2: Array,
) -> tuple[dict[str, object], Array, Array]:
    """Compare intrinsic-permeability measurements with one training cutout."""

    measured = np.asarray(measurement_permeability_m2, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(measured)) or np.any(measured <= 0.0):
        raise ValueError("measurement permeability must be finite and positive")
    sampled, inside = sample_training_field_at_projected_points(
        field,
        georeference,
        measurement_x_m,
        measurement_y_m,
    )
    if measured.shape != sampled.shape:
        raise ValueError("measurement arrays have incompatible shapes")
    valid = inside & np.isfinite(sampled) & (sampled > 0.0)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return (
            {
                "measurement_count_in_run": 0,
                "rmse_log10": None,
                "mae_log10": None,
                "bias_training_minus_measurement_log10": None,
                "correlation_log10": None,
                "fraction_within_0_05_log10": None,
                "fraction_within_0_10_log10": None,
            },
            sampled,
            valid,
        )
    measured_log = np.log10(measured[valid])
    training_log = np.log10(sampled[valid])
    residual = training_log - measured_log
    corr = None
    if count >= 2 and np.std(measured_log) > 0.0 and np.std(training_log) > 0.0:
        corr = float(np.corrcoef(measured_log, training_log)[0, 1])
    return (
        {
            "measurement_count_in_run": count,
            "rmse_log10": float(np.sqrt(np.mean(residual * residual))),
            "mae_log10": float(np.mean(np.abs(residual))),
            "bias_training_minus_measurement_log10": float(np.mean(residual)),
            "correlation_log10": corr,
            "fraction_within_0_05_log10": float(np.mean(np.abs(residual) <= 0.05)),
            "fraction_within_0_10_log10": float(np.mean(np.abs(residual) <= 0.10)),
        },
        sampled,
        valid,
    )


def summarize_metadata_georeference_relation(
    metadata: Iterable[RealisticRunMetadata],
    georeferences: dict[str, LGCNNDomainGeoreference],
    *,
    tolerance_m: float = 100.0,
) -> dict[str, object]:
    """Test simple translation interpretations of historical ``start position``.

    This does not choose a historical convention. It reports whether any of three
    deliberately simple interpretations yields one nearly constant parent-map
    origin across runs. Rotation semantics remain unresolved unless recovered
    from the historical extraction code.
    """

    items = list(metadata)
    if not items:
        raise ValueError("metadata is empty")
    tolerance = float(tolerance_m)
    if tolerance < 0.0:
        raise ValueError("tolerance_m must be non-negative")

    conventions = {
        "start_plus_constant_to_first_cell_center": lambda g: (
            g.first_cell_center_x_m,
            g.first_cell_center_y_m,
        ),
        "start_plus_constant_to_west_south_edge": lambda g: (
            g.west_edge_m,
            g.south_edge_m,
        ),
        "start_plus_constant_to_domain_center": lambda g: (
            0.5 * (g.west_edge_m + g.east_edge_m),
            0.5 * (g.south_edge_m + g.north_edge_m),
        ),
    }
    rows: list[dict[str, object]] = []
    for name, target_fn in conventions.items():
        offsets = []
        per_run = []
        for item in items:
            georef = georeferences.get(item.run_name)
            if georef is None:
                continue
            target_x, target_y = target_fn(georef)
            dx = float(target_x - item.start_position_m[0])
            dy = float(target_y - item.start_position_m[1])
            offsets.append((dx, dy))
            per_run.append(
                {
                    "run_name": item.run_name,
                    "offset_x_m": dx,
                    "offset_y_m": dy,
                }
            )
        if not offsets:
            continue
        arr = np.asarray(offsets, dtype=np.float64)
        spread_x = float(np.ptp(arr[:, 0]))
        spread_y = float(np.ptp(arr[:, 1]))
        rows.append(
            {
                "candidate_convention": name,
                "run_count": int(arr.shape[0]),
                "mean_offset_x_m": float(np.mean(arr[:, 0])),
                "mean_offset_y_m": float(np.mean(arr[:, 1])),
                "spread_x_m": spread_x,
                "spread_y_m": spread_y,
                "translation_consistent_within_tolerance": bool(
                    spread_x <= tolerance and spread_y <= tolerance
                ),
                "per_run": per_run,
            }
        )
    consistent = [
        row for row in rows if row["translation_consistent_within_tolerance"]
    ]
    return {
        "tolerance_m": tolerance,
        "candidate_translation_models": rows,
        "consistent_candidate_count": len(consistent),
        "unique_simple_translation_candidate": len(consistent) == 1,
        "selected_simple_translation_candidate": (
            consistent[0] if len(consistent) == 1 else None
        ),
        "rotation_convention_resolved": False,
        "interpretation": (
            "The per-run YAML proves that historical extraction position, rotation, "
            "and original resolution were recorded. This diagnostic intentionally "
            "does not infer the rotation center/sign or crop ordering without either "
            "historical code or independent reconstruction evidence."
        ),
    }
