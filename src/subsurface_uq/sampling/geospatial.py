from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

Array = np.ndarray

REFERENCE_COLUMNS = {"K_P10": 3, "K_P50": 4, "K_P90": 5}
RAW_TO_GEO_TRANSFORMS = (
    "identity",
    "flip_y",
    "flip_x",
    "flip_xy",
    "transpose",
    "transpose_flip_y",
    "transpose_flip_x",
    "transpose_flip_xy",
)


@dataclass(frozen=True)
class ReferencePermeabilitySurface:
    """One horizontal Munich reference surface on its projected XY grid."""

    x_m: Array
    y_m: Array
    values: Array
    active_mask: Array
    value_name: str
    z_mode: str
    z_value_m: float | None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.values.shape)

    @property
    def spacing_x_m(self) -> float:
        return float(np.median(np.diff(self.x_m)))

    @property
    def spacing_y_m(self) -> float:
        return float(np.median(np.diff(self.y_m)))


@dataclass(frozen=True)
class LGCNNDomainGeoreference:
    """Mapping from a release25 raw 2-D array to projected Munich coordinates."""

    run_name: str
    transform: str
    crs_assumption: str
    first_cell_center_x_m: float
    first_cell_center_y_m: float
    cell_size_m: float
    nx: int
    ny: int
    reference_column: str
    reference_z_mode: str
    reference_z_m: float | None
    log10_unit_shift_raw_minus_reference: float
    unit_shift_interpretation: str
    centered_rmse_log10: float
    correlation: float
    reference_coverage_fraction: float
    matched_points: int
    coarse_origin_ix: int
    coarse_origin_iy: int
    validated: bool

    @property
    def west_edge_m(self) -> float:
        return self.first_cell_center_x_m - 0.5 * self.cell_size_m

    @property
    def south_edge_m(self) -> float:
        return self.first_cell_center_y_m - 0.5 * self.cell_size_m

    @property
    def east_edge_m(self) -> float:
        return self.west_edge_m + self.nx * self.cell_size_m

    @property
    def north_edge_m(self) -> float:
        return self.south_edge_m + self.ny * self.cell_size_m

    def xy_to_fractional_indices(self, x_m: Array, y_m: Array) -> tuple[Array, Array]:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        col = (x - self.first_cell_center_x_m) / self.cell_size_m
        row = (y - self.first_cell_center_y_m) / self.cell_size_m
        return row, col

    def contains_xy(self, x_m: Array, y_m: Array, *, include_boundary: bool = True) -> Array:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if include_boundary:
            return (
                (x >= self.west_edge_m)
                & (x <= self.east_edge_m)
                & (y >= self.south_edge_m)
                & (y <= self.north_edge_m)
            )
        return (
            (x > self.west_edge_m)
            & (x < self.east_edge_m)
            & (y > self.south_edge_m)
            & (y < self.north_edge_m)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_name": self.run_name,
            "transform": self.transform,
            "crs_assumption": self.crs_assumption,
            "first_cell_center_x_m": self.first_cell_center_x_m,
            "first_cell_center_y_m": self.first_cell_center_y_m,
            "west_edge_m": self.west_edge_m,
            "south_edge_m": self.south_edge_m,
            "east_edge_m": self.east_edge_m,
            "north_edge_m": self.north_edge_m,
            "cell_size_m": self.cell_size_m,
            "nx": self.nx,
            "ny": self.ny,
            "reference_column": self.reference_column,
            "reference_z_mode": self.reference_z_mode,
            "reference_z_m": self.reference_z_m,
            "log10_unit_shift_raw_minus_reference": self.log10_unit_shift_raw_minus_reference,
            "unit_shift_interpretation": self.unit_shift_interpretation,
            "centered_rmse_log10": self.centered_rmse_log10,
            "correlation": self.correlation,
            "reference_coverage_fraction": self.reference_coverage_fraction,
            "matched_points": self.matched_points,
            "coarse_origin_ix": self.coarse_origin_ix,
            "coarse_origin_iy": self.coarse_origin_iy,
            "validated": self.validated,
        }


def orient_raw_field(field: Array, transform: str) -> Array:
    """Return a view/copy whose axes are geographic [y increasing, x increasing]."""

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError("raw field must be 2-D")
    transform = str(transform)
    if transform == "identity":
        return values
    if transform == "flip_y":
        return values[::-1, :]
    if transform == "flip_x":
        return values[:, ::-1]
    if transform == "flip_xy":
        return values[::-1, ::-1]
    transposed = values.T
    if transform == "transpose":
        return transposed
    if transform == "transpose_flip_y":
        return transposed[::-1, :]
    if transform == "transpose_flip_x":
        return transposed[:, ::-1]
    if transform == "transpose_flip_xy":
        return transposed[::-1, ::-1]
    raise ValueError(f"unknown raw-to-geographic transform {transform!r}")


def geographic_to_raw_field(field_yx: Array, transform: str) -> Array:
    """Invert orient_raw_field for a geographic [y,x] field."""

    inverse = {
        "identity": "identity",
        "flip_y": "flip_y",
        "flip_x": "flip_x",
        "flip_xy": "flip_xy",
        "transpose": "transpose",
        "transpose_flip_y": "transpose_flip_x",
        "transpose_flip_x": "transpose_flip_y",
        "transpose_flip_xy": "transpose_flip_xy",
    }
    if transform not in inverse:
        raise ValueError(f"unknown raw-to-geographic transform {transform!r}")
    return orient_raw_field(field_yx, inverse[transform])

def _column_index(column: str, n_columns: int) -> int:
    column = str(column).upper()
    if n_columns >= 6:
        if column not in REFERENCE_COLUMNS:
            raise ValueError(f"reference column must be one of {sorted(REFERENCE_COLUMNS)}")
        return REFERENCE_COLUMNS[column]
    if n_columns == 4 and column == "K_P50":
        return 3
    raise ValueError(
        f"reference table has {n_columns} columns; expected six columns X Y Z "
        "K_P10 K_P50 K_P90, or four columns for the P50-only table"
    )


def load_reference_permeability_surface(
    path: str | Path,
    *,
    column: str = "K_P50",
    z_mode: str = "top",
    z_value_m: float | None = None,
) -> ReferencePermeabilitySurface:
    """Load one horizontal surface from the headerless Munich 3-D reference table.

    ``z_mode`` may be ``top``, ``bottom``, ``first``, ``log_geomean`` or
    ``nearest``. ``nearest`` requires ``z_value_m``. The output is indexed
    [y, x] with both projected coordinates increasing.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    mode = str(z_mode).strip().lower()
    if mode not in {"top", "bottom", "first", "log_geomean", "nearest"}:
        raise ValueError("z_mode must be top, bottom, first, log_geomean or nearest")
    if mode == "nearest" and z_value_m is None:
        raise ValueError("z_value_m is required for z_mode='nearest'")

    state: dict[tuple[float, float], tuple[float, float, int] | tuple[float, float]] = {}
    detected_columns: int | None = None
    value_index: int | None = None
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            if detected_columns is None:
                detected_columns = len(parts)
                value_index = _column_index(column, detected_columns)
            if len(parts) != detected_columns:
                raise ValueError(
                    f"inconsistent column count at line {line_number}: "
                    f"{len(parts)} vs {detected_columns}"
                )
            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                value = float(parts[value_index])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid numeric reference row {line_number}") from exc
            if not np.isfinite(value) or value <= 0.0:
                continue
            key = (x, y)
            if mode == "first":
                state.setdefault(key, (z, value))
            elif mode == "top":
                prior = state.get(key)
                if prior is None or z > prior[0]:
                    state[key] = (z, value)
            elif mode == "bottom":
                prior = state.get(key)
                if prior is None or z < prior[0]:
                    state[key] = (z, value)
            elif mode == "nearest":
                distance = abs(z - float(z_value_m))
                prior = state.get(key)
                if prior is None or distance < prior[0]:
                    state[key] = (distance, value)
            else:
                log_value = float(np.log10(value))
                prior = state.get(key)
                if prior is None:
                    state[key] = (log_value, 1, 0)
                else:
                    state[key] = (prior[0] + log_value, prior[1] + 1, 0)

    if not state:
        raise ValueError("reference table produced no finite positive values")
    x_values = np.asarray(sorted({key[0] for key in state}), dtype=np.float64)
    y_values = np.asarray(sorted({key[1] for key in state}), dtype=np.float64)
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("reference surface requires at least two X and Y coordinates")
    values = np.full((y_values.size, x_values.size), np.nan, dtype=np.float64)
    x_to_i = {value: index for index, value in enumerate(x_values.tolist())}
    y_to_i = {value: index for index, value in enumerate(y_values.tolist())}
    for (x, y), payload in state.items():
        if mode == "log_geomean":
            value = 10.0 ** (payload[0] / payload[1])
        else:
            value = payload[1]
        values[y_to_i[y], x_to_i[x]] = float(value)
    active = np.isfinite(values) & (values > 0.0)
    return ReferencePermeabilitySurface(
        x_m=x_values,
        y_m=y_values,
        values=values,
        active_mask=active,
        value_name=str(column).upper(),
        z_mode=mode,
        z_value_m=None if z_value_m is None else float(z_value_m),
    )


def _block_log_median(field: Array, ratio_y: int, ratio_x: int) -> Array:
    log_values = np.log10(np.asarray(field, dtype=np.float64))
    h = (log_values.shape[0] // ratio_y) * ratio_y
    w = (log_values.shape[1] // ratio_x) * ratio_x
    if h == 0 or w == 0:
        raise ValueError("raw field is smaller than one reference-grid block")
    trimmed = log_values[:h, :w]
    return np.median(
        trimmed.reshape(h // ratio_y, ratio_y, w // ratio_x, ratio_x),
        axis=(1, 3),
    )


def _centered_score(raw_log: Array, reference_log: Array) -> tuple[float, float, float, int]:
    raw = np.asarray(raw_log, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference_log, dtype=np.float64).reshape(-1)
    valid = np.isfinite(raw) & np.isfinite(ref)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return np.inf, np.nan, np.nan, count
    raw = raw[valid]
    ref = ref[valid]
    shift = float(np.mean(raw - ref))
    residual = raw - (ref + shift)
    rmse = float(np.sqrt(np.mean(residual**2)))
    if np.std(raw) <= 0.0 or np.std(ref) <= 0.0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(raw, ref)[0, 1])
    return rmse, correlation, shift, count


def _coarse_candidates(
    raw_geo: Array,
    reference: ReferencePermeabilitySurface,
    *,
    raw_cell_size_m: float,
    anchor_stride: int,
    min_coverage: float,
    keep: int,
) -> list[tuple[float, int, int, float, float, int]]:
    ratio_x = int(round(reference.spacing_x_m / raw_cell_size_m))
    ratio_y = int(round(reference.spacing_y_m / raw_cell_size_m))
    if not np.isclose(ratio_x * raw_cell_size_m, reference.spacing_x_m, atol=1e-6):
        raise ValueError("reference X spacing is not an integer multiple of raw cell size")
    if not np.isclose(ratio_y * raw_cell_size_m, reference.spacing_y_m, atol=1e-6):
        raise ValueError("reference Y spacing is not an integer multiple of raw cell size")
    coarse = _block_log_median(raw_geo, ratio_y, ratio_x)
    ref_log = np.where(reference.active_mask, np.log10(reference.values), np.nan)
    if coarse.shape[0] > ref_log.shape[0] or coarse.shape[1] > ref_log.shape[1]:
        raise ValueError("raw domain is larger than the reference surface")

    rows = np.arange(0, coarse.shape[0], max(1, int(anchor_stride)), dtype=np.int64)
    cols = np.arange(0, coarse.shape[1], max(1, int(anchor_stride)), dtype=np.int64)
    ay, ax = np.meshgrid(rows, cols, indexing="ij")
    ay = ay.reshape(-1)
    ax = ax.reshape(-1)
    raw_anchor = coarse[ay, ax]
    required = max(3, int(np.ceil(min_coverage * raw_anchor.size)))
    candidates: list[tuple[float, int, int, float, float, int]] = []
    max_iy = ref_log.shape[0] - coarse.shape[0]
    max_ix = ref_log.shape[1] - coarse.shape[1]
    for iy in range(max_iy + 1):
        for ix in range(max_ix + 1):
            ref_anchor = ref_log[iy + ay, ix + ax]
            rmse, corr, shift, count = _centered_score(raw_anchor, ref_anchor)
            if count < required or not np.isfinite(rmse):
                continue
            candidates.append((rmse, iy, ix, corr, shift, count))
    candidates.sort(key=lambda item: item[0])
    return candidates[: max(1, int(keep))]


def _interpolator(reference: ReferencePermeabilitySurface) -> RegularGridInterpolator:
    log_values = np.where(reference.active_mask, np.log10(reference.values), np.nan)
    return RegularGridInterpolator(
        (reference.y_m, reference.x_m),
        log_values,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )


def _unit_shift_interpretation(
    shift: float,
    *,
    dynamic_viscosity_pa_s: float,
    density_kg_m3: float,
    gravity_m_s2: float,
) -> str:
    hydraulic_to_intrinsic = float(
        np.log10(dynamic_viscosity_pa_s / (density_kg_m3 * gravity_m_s2))
    )
    same_error = abs(shift)
    conversion_error = abs(shift - hydraulic_to_intrinsic)
    if conversion_error + 0.15 < same_error:
        return (
            "reference_values_behave_like_hydraulic_conductivity_m_per_s; "
            f"expected_log10_shift={hydraulic_to_intrinsic:.6f}"
        )
    if same_error + 0.15 < conversion_error:
        return "reference_values_behave_like_intrinsic_permeability_m2"
    return (
        "unit_relation_ambiguous; "
        f"hydraulic_to_intrinsic_expected_log10_shift={hydraulic_to_intrinsic:.6f}"
    )


def infer_lgcnn_domain_georeference(
    raw_field: Array,
    reference: ReferencePermeabilitySurface,
    *,
    run_name: str,
    raw_cell_size_m: float = 5.0,
    transforms: Iterable[str] = RAW_TO_GEO_TRANSFORMS,
    coarse_anchor_stride: int = 8,
    coarse_keep_per_transform: int = 3,
    refine_radius_m: float = 100.0,
    refine_step_m: float | None = None,
    refine_sample_stride_cells: int = 128,
    min_reference_coverage: float = 0.80,
    min_correlation: float = 0.90,
    max_centered_rmse_log10: float = 0.15,
    crs_assumption: str = "DHDN / Gauss-Kruger zone 4 (EPSG:31468), pending confirmation",
    dynamic_viscosity_pa_s: float = 1.002e-3,
    density_kg_m3: float = 998.2,
    gravity_m_s2: float = 9.80665,
) -> LGCNNDomainGeoreference:
    """Infer crop origin/orientation by matching a DaRUS field to the Munich map.

    The comparison is performed in log space and fits one additive offset, so a
    constant hydraulic-conductivity-to-intrinsic-permeability unit conversion
    does not affect the spatial match. A coarse 100 m fingerprint identifies
    candidate crop windows; a 5 m refinement then uses bilinear interpolation
    of the reference surface.
    """

    raw = np.asarray(raw_field, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError("raw_field must be a finite positive 2-D array")
    raw_cell_size_m = float(raw_cell_size_m)
    if raw_cell_size_m <= 0.0:
        raise ValueError("raw_cell_size_m must be positive")
    refine_step = raw_cell_size_m if refine_step_m is None else float(refine_step_m)
    if refine_step <= 0.0:
        raise ValueError("refine_step_m must be positive")

    coarse_all: list[tuple[float, str, int, int, float, float, int]] = []
    for transform in transforms:
        geo = orient_raw_field(raw, transform)
        for rmse, iy, ix, corr, shift, count in _coarse_candidates(
            geo,
            reference,
            raw_cell_size_m=raw_cell_size_m,
            anchor_stride=coarse_anchor_stride,
            min_coverage=min_reference_coverage,
            keep=coarse_keep_per_transform,
        ):
            coarse_all.append((rmse, transform, iy, ix, corr, shift, count))
    if not coarse_all:
        raise RuntimeError("no coarse geospatial match satisfies the coverage requirement")
    coarse_all.sort(key=lambda item: item[0])

    interp = _interpolator(reference)
    best: tuple[float, float, float, int, str, float, float, int, int] | None = None
    for _, transform, coarse_iy, coarse_ix, _, _, _ in coarse_all:
        geo = orient_raw_field(raw, transform)
        sample_rows = np.arange(
            0, geo.shape[0], max(1, int(refine_sample_stride_cells)), dtype=np.int64
        )
        sample_cols = np.arange(
            0, geo.shape[1], max(1, int(refine_sample_stride_cells)), dtype=np.int64
        )
        sy, sx = np.meshgrid(sample_rows, sample_cols, indexing="ij")
        sy = sy.reshape(-1)
        sx = sx.reshape(-1)
        raw_sample = np.log10(geo[sy, sx])

        ratio_x = reference.spacing_x_m / raw_cell_size_m
        ratio_y = reference.spacing_y_m / raw_cell_size_m
        initial_x_center = float(
            reference.x_m[coarse_ix] - (0.5 * ratio_x - 0.5) * raw_cell_size_m
        )
        initial_y_center = float(
            reference.y_m[coarse_iy] - (0.5 * ratio_y - 0.5) * raw_cell_size_m
        )
        offsets = np.arange(
            -float(refine_radius_m),
            float(refine_radius_m) + 0.5 * refine_step,
            refine_step,
        )
        for dy in offsets:
            y0 = initial_y_center + float(dy)
            y_coords = y0 + sy.astype(np.float64) * raw_cell_size_m
            for dx in offsets:
                x0 = initial_x_center + float(dx)
                x_coords = x0 + sx.astype(np.float64) * raw_cell_size_m
                reference_sample = interp(np.column_stack((y_coords, x_coords)))
                rmse, corr, shift, count = _centered_score(raw_sample, reference_sample)
                coverage = count / raw_sample.size
                if coverage < min_reference_coverage or not np.isfinite(rmse):
                    continue
                candidate = (
                    rmse,
                    corr,
                    shift,
                    count,
                    transform,
                    x0,
                    y0,
                    coarse_ix,
                    coarse_iy,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        raise RuntimeError("no refined geospatial match satisfies the coverage requirement")

    rmse, correlation, shift, count, transform, x0, y0, coarse_ix, coarse_iy = best
    oriented = orient_raw_field(raw, transform)
    coverage = count / (
        np.arange(0, oriented.shape[0], max(1, int(refine_sample_stride_cells))).size
        * np.arange(0, oriented.shape[1], max(1, int(refine_sample_stride_cells))).size
    )
    validated = bool(
        np.isfinite(correlation)
        and correlation >= float(min_correlation)
        and rmse <= float(max_centered_rmse_log10)
        and coverage >= float(min_reference_coverage)
    )
    return LGCNNDomainGeoreference(
        run_name=str(run_name),
        transform=transform,
        crs_assumption=str(crs_assumption),
        first_cell_center_x_m=float(x0),
        first_cell_center_y_m=float(y0),
        cell_size_m=raw_cell_size_m,
        nx=int(oriented.shape[1]),
        ny=int(oriented.shape[0]),
        reference_column=reference.value_name,
        reference_z_mode=reference.z_mode,
        reference_z_m=reference.z_value_m,
        log10_unit_shift_raw_minus_reference=float(shift),
        unit_shift_interpretation=_unit_shift_interpretation(
            shift,
            dynamic_viscosity_pa_s=dynamic_viscosity_pa_s,
            density_kg_m3=density_kg_m3,
            gravity_m_s2=gravity_m_s2,
        ),
        centered_rmse_log10=float(rmse),
        correlation=float(correlation),
        reference_coverage_fraction=float(coverage),
        matched_points=int(count),
        coarse_origin_ix=int(coarse_ix),
        coarse_origin_iy=int(coarse_iy),
        validated=validated,
    )
