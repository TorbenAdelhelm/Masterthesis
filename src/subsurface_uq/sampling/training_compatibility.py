from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

Array = np.ndarray

RELEASE25_REALK_MODEL_DOI = "10.18419/DARUS-5082"
RELEASE25_REALK_RAW_DATA_DOI = "10.18419/DARUS-5065"
RELEASE25_REALK_TRAINING_FIELDS = 3
RELEASE25_REALK_VALIDATION_FIELDS = 1
RELEASE25_REALK_SCALING_FIELDS = 1
RELEASE25_REALK_PATCH_BOX_SIZE = 1280
RELEASE25_REALK_PATCH_SKIP = 8

FEATURE_NAMES = (
    "mean_log10_k",
    "std_log10_k",
    "q05_log10_k",
    "q50_log10_k",
    "q95_log10_k",
    "gradient_rms_y_log10_per_m",
    "gradient_rms_x_log10_per_m",
    "lag1_correlation_y",
    "lag1_correlation_x",
)


def _positive_fields(fields: Array) -> Array:
    values = np.asarray(fields)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError("permeability fields must have shape [N,H,W] with N>0")
    if values.shape[1] < 2 or values.shape[2] < 2:
        raise ValueError("permeability fields must contain at least 2x2 cells")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("permeability fields must be finite and strictly positive")
    return values


def _lag1_correlation(values: Array, *, axis: int) -> float:
    if axis == 0:
        first = values[:-1, :].reshape(-1)
        second = values[1:, :].reshape(-1)
    elif axis == 1:
        first = values[:, :-1].reshape(-1)
        second = values[:, 1:].reshape(-1)
    else:
        raise ValueError("axis must be 0 or 1")
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = float(
        np.sqrt(
            np.sum(first * first, dtype=np.float64)
            * np.sum(second * second, dtype=np.float64)
        )
    )
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(first * second, dtype=np.float64) / denominator)


def permeability_field_features(
    field: Array,
    *,
    cell_size_m: float,
    spatial_stride: int = 1,
) -> dict[str, float]:
    """Return marginal and short-range descriptors of one field/patch in log10(K)."""

    raw = np.asarray(field)
    if raw.ndim != 2:
        raise ValueError("field must be two-dimensional")
    cell_size_m = float(cell_size_m)
    spatial_stride = int(spatial_stride)
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")

    sampled_raw = raw[::spatial_stride, ::spatial_stride]
    if sampled_raw.shape[0] < 2 or sampled_raw.shape[1] < 2:
        raise ValueError("spatial_stride leaves fewer than 2x2 diagnostic cells")
    if not np.all(np.isfinite(sampled_raw)) or np.any(sampled_raw <= 0.0):
        raise ValueError("permeability field must be finite and strictly positive")
    sampled = np.log10(np.asarray(sampled_raw, dtype=np.float64))
    spacing = cell_size_m * spatial_stride
    quantiles = np.quantile(sampled, (0.05, 0.50, 0.95))
    gradient_y = np.diff(sampled, axis=0) / spacing
    gradient_x = np.diff(sampled, axis=1) / spacing

    return {
        "mean_log10_k": float(np.mean(sampled)),
        "std_log10_k": float(np.std(sampled, ddof=1)),
        "q05_log10_k": float(quantiles[0]),
        "q50_log10_k": float(quantiles[1]),
        "q95_log10_k": float(quantiles[2]),
        "gradient_rms_y_log10_per_m": float(
            np.sqrt(np.mean(gradient_y * gradient_y))
        ),
        "gradient_rms_x_log10_per_m": float(
            np.sqrt(np.mean(gradient_x * gradient_x))
        ),
        "lag1_correlation_y": _lag1_correlation(sampled, axis=0),
        "lag1_correlation_x": _lag1_correlation(sampled, axis=1),
    }


def _summarize_feature_rows(
    rows: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    if not rows:
        raise ValueError("at least one feature row is required")
    reference: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        metric = np.asarray([row[name] for row in rows], dtype=np.float64)
        reference[name] = {
            "mean": float(np.mean(metric)),
            "std": float(np.std(metric, ddof=1)) if metric.size > 1 else 0.0,
            "min": float(np.min(metric)),
            "max": float(np.max(metric)),
        }
    return reference


def _comparison_payload(
    rows: list[dict[str, float]],
    reference: dict[str, dict[str, float]],
    *,
    unit_label: str,
) -> dict[str, dict[str, float | None]]:
    metrics: dict[str, dict[str, float | None]] = {}
    for name in FEATURE_NAMES:
        generated = np.asarray([row[name] for row in rows], dtype=np.float64)
        ref = reference[name]
        ref_std = float(ref["std"])
        mean_z = (
            None
            if ref_std <= np.finfo(float).eps
            else float((np.mean(generated) - float(ref["mean"])) / ref_std)
        )
        outside = (generated < float(ref["min"])) | (
            generated > float(ref["max"])
        )
        metrics[name] = {
            "training_mean": float(ref["mean"]),
            "training_std": ref_std,
            "training_min": float(ref["min"]),
            "training_max": float(ref["max"]),
            "generated_mean": float(np.mean(generated)),
            "generated_std": (
                float(np.std(generated, ddof=1)) if generated.size > 1 else 0.0
            ),
            "generated_min": float(np.min(generated)),
            "generated_max": float(np.max(generated)),
            "generated_mean_z_score": mean_z,
            f"fraction_generated_{unit_label}_outside_training_envelope": float(
                np.mean(outside)
            ),
        }
    return metrics


@dataclass(frozen=True)
class TrainingDistributionProfile:
    """Secondary full-field descriptors of LGCNN training permeability inputs."""

    field_count: int
    field_shape: tuple[int, int]
    cell_size_m: float
    spatial_stride: int
    minimum_k: float
    maximum_k: float
    metric_reference: dict[str, dict[str, float]]
    field_names: tuple[str, ...] = ()

    @property
    def training_k_range(self) -> tuple[float, float]:
        return (self.minimum_k, self.maximum_k)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_level": "full_field_secondary",
            "field_count": int(self.field_count),
            "field_names": list(self.field_names),
            "field_shape": [int(v) for v in self.field_shape],
            "cell_size_m": float(self.cell_size_m),
            "spatial_stride": int(self.spatial_stride),
            "minimum_k_m2": float(self.minimum_k),
            "maximum_k_m2": float(self.maximum_k),
            "metrics": {
                name: {key: float(value) for key, value in summary.items()}
                for name, summary in self.metric_reference.items()
            },
            "interpretation": (
                "Full-field descriptors are secondary geostatistical/support diagnostics "
                "and are not used to reject generated fields. The pretrained real-K LGCNN "
                "was optimized on overlapping cutouts, so patch-level compatibility is the "
                "primary surrogate-support diagnostic."
            ),
        }


def characterize_training_distribution(
    fields: Array,
    *,
    cell_size_m: float,
    spatial_stride: int = 1,
    field_names: tuple[str, ...] | list[str] | None = None,
) -> TrainingDistributionProfile:
    """Characterize complete permeability fields without treating them as patches."""

    values = _positive_fields(fields)
    names = tuple(field_names or ())
    if names and len(names) != values.shape[0]:
        raise ValueError("field_names must contain one entry per field")
    rows = [
        permeability_field_features(
            field,
            cell_size_m=cell_size_m,
            spatial_stride=spatial_stride,
        )
        for field in values
    ]
    return TrainingDistributionProfile(
        field_count=int(values.shape[0]),
        field_shape=(int(values.shape[1]), int(values.shape[2])),
        cell_size_m=float(cell_size_m),
        spatial_stride=int(spatial_stride),
        minimum_k=float(np.min(values)),
        maximum_k=float(np.max(values)),
        metric_reference=_summarize_feature_rows(rows),
        field_names=names,
    )


def release25_patch_positions(
    field_shape: tuple[int, int],
    *,
    box_size: int,
    skip_per_dir: int,
) -> Array:
    """Reproduce SimulationDatasetCuts.idx_to_pos for one full field.

    release25 defines n = (H-B)*(W-B)//skip^2 patches per full field and maps
    consecutive patch indices to the upper-left positions below. This deliberately
    excludes a patch whose upper-left corner is exactly H-B or W-B.
    """

    h, w = (int(field_shape[0]), int(field_shape[1]))
    box_size = int(box_size)
    skip_per_dir = int(skip_per_dir)
    if box_size <= 0 or skip_per_dir <= 0:
        raise ValueError("box_size and skip_per_dir must be positive")
    if box_size >= h or box_size >= w:
        raise ValueError("box_size must be smaller than both field dimensions")
    span_y = h - box_size
    span_x = w - box_size
    count = span_y * span_x // (skip_per_dir**2)
    if count <= 0:
        raise ValueError("patch specification produces no patches")
    patch_index = np.arange(count, dtype=np.int64)
    scaled = patch_index * skip_per_dir
    rows = (scaled // span_x) * skip_per_dir
    cols = scaled % span_x
    positions = np.column_stack((rows, cols))
    if np.any(rows + box_size > h) or np.any(cols + box_size > w):
        raise RuntimeError(
            "release25 patch indexing produced an out-of-domain patch; "
            "the requested geometry is incompatible with SimulationDatasetCuts"
        )
    return positions


def _evenly_spaced_indices(total: int, count: int) -> Array:
    total = int(total)
    count = int(count)
    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    if count >= total:
        return np.arange(total, dtype=np.int64)
    # Midpoint sampling avoids systematically preferring the first/last patch.
    edges = np.linspace(0.0, float(total), count + 1)
    indices = np.floor(0.5 * (edges[:-1] + edges[1:])).astype(np.int64)
    return np.clip(indices, 0, total - 1)


@dataclass(frozen=True)
class TrainingPatchDistributionProfile:
    """Patch-level support actually presented to the pretrained real-K LGCNN."""

    field_count: int
    field_names: tuple[str, ...]
    field_shape: tuple[int, int]
    cell_size_m: float
    box_size: int
    skip_per_dir: int
    feature_stride: int
    patch_population_per_field: int
    patch_population_total: int
    sampled_patch_count: int
    sampled_patch_count_per_field: tuple[int, ...]
    minimum_k: float
    maximum_k: float
    metric_reference: dict[str, dict[str, float]]

    @property
    def training_k_range(self) -> tuple[float, float]:
        return (self.minimum_k, self.maximum_k)

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_level": "release25_training_patch_primary",
            "model_dataset_doi": RELEASE25_REALK_MODEL_DOI,
            "raw_dataset_doi": RELEASE25_REALK_RAW_DATA_DOI,
            "documented_split": {
                "training_full_fields": RELEASE25_REALK_TRAINING_FIELDS,
                "validation_full_fields": RELEASE25_REALK_VALIDATION_FIELDS,
                "scaling_full_fields": RELEASE25_REALK_SCALING_FIELDS,
            },
            "field_count": int(self.field_count),
            "field_names": list(self.field_names),
            "field_shape": [int(v) for v in self.field_shape],
            "cell_size_m": float(self.cell_size_m),
            "patch_extraction": {
                "implementation": "release25 SimulationDatasetCuts",
                "box_size_cells": int(self.box_size),
                "skip_per_dir_cells": int(self.skip_per_dir),
                "patch_population_per_training_field": int(
                    self.patch_population_per_field
                ),
                "patch_population_total": int(self.patch_population_total),
                "sampled_patch_count_for_diagnostics": int(self.sampled_patch_count),
                "sampled_patch_count_per_field": [
                    int(v) for v in self.sampled_patch_count_per_field
                ],
                "feature_spatial_stride_cells": int(self.feature_stride),
            },
            "minimum_k_m2": float(self.minimum_k),
            "maximum_k_m2": float(self.maximum_k),
            "metrics": {
                name: {key: float(value) for key, value in summary.items()}
                for name, summary in self.metric_reference.items()
            },
            "interpretation": (
                "Primary surrogate-support reference. The original real-K LGCNN was "
                "trained on overlapping 1280-cell cutouts with skip 8. The diagnostic "
                "subsamples that exact patch lattice for tractable comparison; patch "
                "samples are correlated and are not treated as independent geological "
                "realizations."
            ),
        }


def characterize_training_patch_distribution(
    fields: Array,
    *,
    cell_size_m: float,
    box_size: int = RELEASE25_REALK_PATCH_BOX_SIZE,
    skip_per_dir: int = RELEASE25_REALK_PATCH_SKIP,
    max_patches: int = 192,
    feature_stride: int = 8,
    field_names: tuple[str, ...] | list[str] | None = None,
) -> TrainingPatchDistributionProfile:
    """Characterize a deterministic subsample of the exact release25 patch lattice."""

    values = _positive_fields(fields)
    names = tuple(field_names or ())
    if names and len(names) != values.shape[0]:
        raise ValueError("field_names must contain one entry per field")
    positions = release25_patch_positions(
        (int(values.shape[1]), int(values.shape[2])),
        box_size=box_size,
        skip_per_dir=skip_per_dir,
    )
    max_patches = int(max_patches)
    feature_stride = int(feature_stride)
    if max_patches <= 0:
        raise ValueError("max_patches must be positive")
    if feature_stride <= 0:
        raise ValueError("feature_stride must be positive")

    n_fields = int(values.shape[0])
    base = max_patches // n_fields
    remainder = max_patches % n_fields
    rows: list[dict[str, float]] = []
    counts: list[int] = []
    for field_index, field in enumerate(values):
        requested = base + (1 if field_index < remainder else 0)
        requested = max(requested, 1)
        selected_indices = _evenly_spaced_indices(
            len(positions), min(requested, len(positions))
        )
        counts.append(int(selected_indices.size))
        for patch_index in selected_indices:
            row, col = positions[int(patch_index)]
            patch = field[
                row : row + int(box_size),
                col : col + int(box_size),
            ]
            rows.append(
                permeability_field_features(
                    patch,
                    cell_size_m=cell_size_m,
                    spatial_stride=feature_stride,
                )
            )

    return TrainingPatchDistributionProfile(
        field_count=n_fields,
        field_names=names,
        field_shape=(int(values.shape[1]), int(values.shape[2])),
        cell_size_m=float(cell_size_m),
        box_size=int(box_size),
        skip_per_dir=int(skip_per_dir),
        feature_stride=feature_stride,
        patch_population_per_field=int(len(positions)),
        patch_population_total=int(len(positions) * n_fields),
        sampled_patch_count=len(rows),
        sampled_patch_count_per_field=tuple(counts),
        minimum_k=float(np.min(values)),
        maximum_k=float(np.max(values)),
        metric_reference=_summarize_feature_rows(rows),
    )


@dataclass
class TrainingCompatibilityDiagnostics:
    """Secondary complete-field comparison without filtering generated samples."""

    profile: TrainingDistributionProfile
    _rows: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)

    def update(self, permeability_batch: Array) -> None:
        values = _positive_fields(permeability_batch)
        if tuple(values.shape[1:]) != self.profile.field_shape:
            raise ValueError(
                "generated field shape differs from the LGCNN training-input shape: "
                f"{tuple(values.shape[1:])} != {self.profile.field_shape}"
            )
        for field in values:
            self._rows.append(
                permeability_field_features(
                    field,
                    cell_size_m=self.profile.cell_size_m,
                    spatial_stride=self.profile.spatial_stride,
                )
            )

    @property
    def count(self) -> int:
        return len(self._rows)

    def finalize(self) -> dict[str, object]:
        if not self._rows:
            raise RuntimeError(
                "training compatibility diagnostics received no generated fields"
            )
        return {
            "reference_level": "full_field_secondary",
            "generated_field_count": len(self._rows),
            "training_reference": self.profile.to_dict(),
            "metrics": _comparison_payload(
                self._rows,
                self.profile.metric_reference,
                unit_label="fields",
            ),
            "decision_rule": None,
            "interpretation": (
                "Secondary full-field comparison only. No generated realization is "
                "rejected; the Gaussian posterior-coordinate law is unchanged."
            ),
        }


@dataclass
class TrainingPatchCompatibilityDiagnostics:
    """Primary patch-level LGCNN support diagnostic with no sample rejection."""

    profile: TrainingPatchDistributionProfile
    generated_patches_per_field: int = 32
    _rows: list[dict[str, float]] = field(default_factory=list, init=False, repr=False)
    _field_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.generated_patches_per_field = int(self.generated_patches_per_field)
        if self.generated_patches_per_field <= 0:
            raise ValueError("generated_patches_per_field must be positive")

    def update(self, permeability_batch: Array) -> None:
        values = _positive_fields(permeability_batch)
        if tuple(values.shape[1:]) != self.profile.field_shape:
            raise ValueError(
                "generated field shape differs from the LGCNN training-input shape: "
                f"{tuple(values.shape[1:])} != {self.profile.field_shape}"
            )
        positions = release25_patch_positions(
            self.profile.field_shape,
            box_size=self.profile.box_size,
            skip_per_dir=self.profile.skip_per_dir,
        )
        selected = _evenly_spaced_indices(
            len(positions),
            min(self.generated_patches_per_field, len(positions)),
        )
        for field in values:
            self._field_count += 1
            for patch_index in selected:
                row, col = positions[int(patch_index)]
                patch = field[
                    row : row + self.profile.box_size,
                    col : col + self.profile.box_size,
                ]
                self._rows.append(
                    permeability_field_features(
                        patch,
                        cell_size_m=self.profile.cell_size_m,
                        spatial_stride=self.profile.feature_stride,
                    )
                )

    @property
    def count(self) -> int:
        return len(self._rows)

    def finalize(self) -> dict[str, object]:
        if not self._rows:
            raise RuntimeError(
                "training patch compatibility diagnostics received no generated patches"
            )
        return {
            "reference_level": "release25_training_patch_primary",
            "generated_field_count": int(self._field_count),
            "generated_patch_count": len(self._rows),
            "generated_patches_per_field": int(self.generated_patches_per_field),
            "training_reference": self.profile.to_dict(),
            "metrics": _comparison_payload(
                self._rows,
                self.profile.metric_reference,
                unit_label="patches",
            ),
            "decision_rule": None,
            "interpretation": (
                "Primary surrogate-support diagnostic because the pretrained real-K "
                "LGCNN learned from overlapping spatial cutouts. No generated field or "
                "patch is rejected, preserving eta~N(0,I) for MC/RQMC/Hermite PCE."
            ),
        }


def load_release25_permeability_normalization(
    info_yaml: str | Path,
) -> dict[str, object]:
    """Read the permeability input normalization recorded by release25 info.yaml."""

    source = Path(info_yaml).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    try:
        stats = payload["Inputs"]["Permeability X [m^2]"]
    except (TypeError, KeyError) as exc:
        raise ValueError(
            "release25 info.yaml does not contain Inputs -> 'Permeability X [m^2]'"
        ) from exc
    result = {
        "source": str(source),
        "variable": "Permeability X [m^2]",
        "index": int(stats["index"]),
        "norm": stats.get("norm"),
        "min": float(stats["min"]),
        "max": float(stats["max"]),
        "mean": float(stats["mean"]),
        "std": float(stats["std"]),
        "note": (
            "release25 normalizes prepared inputs before SimulationDatasetCuts extracts "
            "training patches. Physical/log10 patch diagnostics remain primary here; "
            "these stats record the exact network-input normalization context."
        ),
    }
    if not (
        np.isfinite(result["min"])
        and np.isfinite(result["max"])
        and result["min"] < result["max"]
    ):
        raise ValueError("invalid permeability normalization range in info.yaml")
    return result
