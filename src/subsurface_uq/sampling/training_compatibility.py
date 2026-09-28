from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray

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
    values = np.asarray(fields, dtype=np.float64)
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
    """Return marginal and short-range spatial descriptors in log10(K).

    These descriptors are diagnostics, not an acceptance/rejection rule. The
    same stride and physical spacing are used for the training reference and
    generated fields so comparisons remain like-for-like.
    """

    values = _positive_fields(field)[0]
    cell_size_m = float(cell_size_m)
    spatial_stride = int(spatial_stride)
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")

    sampled = np.log10(values[::spatial_stride, ::spatial_stride])
    if sampled.shape[0] < 2 or sampled.shape[1] < 2:
        raise ValueError("spatial_stride leaves fewer than 2x2 diagnostic cells")
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


@dataclass(frozen=True)
class TrainingDistributionProfile:
    """Empirical descriptors of permeability fields seen during LGCNN training."""

    field_count: int
    field_shape: tuple[int, int]
    cell_size_m: float
    spatial_stride: int
    minimum_k: float
    maximum_k: float
    metric_reference: dict[str, dict[str, float]]

    @property
    def training_k_range(self) -> tuple[float, float]:
        return (self.minimum_k, self.maximum_k)

    def to_dict(self) -> dict[str, object]:
        return {
            "field_count": int(self.field_count),
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
                "Empirical training-input descriptors are a compatibility reference only. "
                "They are not used to reject posterior samples and therefore do not alter "
                "the iid Gaussian latent-coordinate law."
            ),
        }


def characterize_training_distribution(
    fields: Array,
    *,
    cell_size_m: float,
    spatial_stride: int = 1,
) -> TrainingDistributionProfile:
    """Characterize the actual permeability inputs used to train the LGCNN."""

    values = _positive_fields(fields)
    rows = [
        permeability_field_features(
            field,
            cell_size_m=cell_size_m,
            spatial_stride=spatial_stride,
        )
        for field in values
    ]
    reference: dict[str, dict[str, float]] = {}
    for name in FEATURE_NAMES:
        metric = np.asarray([row[name] for row in rows], dtype=np.float64)
        reference[name] = {
            "mean": float(np.mean(metric)),
            "std": float(np.std(metric, ddof=1)) if metric.size > 1 else 0.0,
            "min": float(np.min(metric)),
            "max": float(np.max(metric)),
        }
    return TrainingDistributionProfile(
        field_count=int(values.shape[0]),
        field_shape=(int(values.shape[1]), int(values.shape[2])),
        cell_size_m=float(cell_size_m),
        spatial_stride=int(spatial_stride),
        minimum_k=float(np.min(values)),
        maximum_k=float(np.max(values)),
        metric_reference=reference,
    )


@dataclass
class TrainingCompatibilityDiagnostics:
    """Compare generated fields with training inputs without filtering samples."""

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
        metrics: dict[str, dict[str, float | None]] = {}
        for name in FEATURE_NAMES:
            generated = np.asarray([row[name] for row in self._rows], dtype=np.float64)
            reference = self.profile.metric_reference[name]
            ref_std = float(reference["std"])
            mean_z = (
                None
                if ref_std <= np.finfo(float).eps
                else float(
                    (np.mean(generated) - float(reference["mean"])) / ref_std
                )
            )
            outside = (generated < float(reference["min"])) | (
                generated > float(reference["max"])
            )
            metrics[name] = {
                "training_mean": float(reference["mean"]),
                "training_std": ref_std,
                "training_min": float(reference["min"]),
                "training_max": float(reference["max"]),
                "generated_mean": float(np.mean(generated)),
                "generated_std": (
                    float(np.std(generated, ddof=1))
                    if generated.size > 1
                    else 0.0
                ),
                "generated_min": float(np.min(generated)),
                "generated_max": float(np.max(generated)),
                "generated_mean_z_score": mean_z,
                "fraction_generated_fields_outside_training_envelope": float(
                    np.mean(outside)
                ),
            }
        return {
            "generated_field_count": len(self._rows),
            "training_reference": self.profile.to_dict(),
            "metrics": metrics,
            "decision_rule": None,
            "interpretation": (
                "Compatibility is reported descriptively. No generated field is rejected "
                "on these diagnostics, preserving the Gaussian posterior coordinate law "
                "required by MC/RQMC and Hermite PCE."
            ),
        }
