from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .base import PermeabilitySampler

Array = np.ndarray


@dataclass(frozen=True)
class PermeabilityDiagnosticsResult:
    """Streaming summary of the permeability realizations seen by propagation."""

    count: int
    mean_log10_k: Array
    variance_log10_k: Array
    std_log10_k: Array
    preview_log10_k: tuple[Array, ...]
    minimum_k: float | None = None
    maximum_k: float | None = None
    training_k_min: float | None = None
    training_k_max: float | None = None
    outside_training_fraction: float | None = None
    outside_training_fraction_by_sample: tuple[float, ...] = ()
    conditioning_max_abs_log10_residual: float | None = None
    conditioning_rms_log10_residual: float | None = None
    conditioning_max_abs_log10_residual_by_sample: tuple[float, ...] = ()


@dataclass
class PermeabilityDiagnostics:
    """Accumulate log10(K) statistics without retaining the full ensemble.

    Optional training bounds and conditioning observations add the input-law
    compatibility diagnostics required by RQ1 while preserving the existing
    lightweight plotting use case.
    """

    preview_count: int = 3
    ddof: int = 1
    training_k_range: tuple[float, float] | None = None
    observation_indices: Array | None = None
    observation_log10_k: Array | None = None
    _count: int = field(default=0, init=False, repr=False)
    _mean: Array | None = field(default=None, init=False, repr=False)
    _m2: Array | None = field(default=None, init=False, repr=False)
    _previews: list[Array] = field(default_factory=list, init=False, repr=False)
    _minimum_k: float = field(default=np.inf, init=False, repr=False)
    _maximum_k: float = field(default=-np.inf, init=False, repr=False)
    _outside_count: int = field(default=0, init=False, repr=False)
    _total_cell_count: int = field(default=0, init=False, repr=False)
    _outside_by_sample: list[float] = field(default_factory=list, init=False, repr=False)
    _conditioning_max_by_sample: list[float] = field(default_factory=list, init=False, repr=False)
    _conditioning_sumsq: float = field(default=0.0, init=False, repr=False)
    _conditioning_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.preview_count = int(self.preview_count)
        self.ddof = int(self.ddof)
        if self.preview_count < 0:
            raise ValueError("preview_count must be non-negative")
        if self.ddof < 0:
            raise ValueError("ddof must be non-negative")

        if self.training_k_range is not None:
            low, high = (float(self.training_k_range[0]), float(self.training_k_range[1]))
            if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
                raise ValueError("training_k_range must satisfy 0 < low < high")
            self.training_k_range = (low, high)

        if (self.observation_indices is None) != (self.observation_log10_k is None):
            raise ValueError(
                "observation_indices and observation_log10_k must be supplied together"
            )
        if self.observation_indices is not None:
            indices = np.asarray(self.observation_indices)
            values = np.asarray(self.observation_log10_k, dtype=np.float64)
            if indices.ndim != 2 or indices.shape[1] != 2:
                raise ValueError("observation_indices must have shape [n,2]")
            if values.ndim != 1 or values.shape[0] != indices.shape[0]:
                raise ValueError("observation_log10_k must have one value per observation")
            if not np.all(np.isfinite(indices)) or not np.all(np.isfinite(values)):
                raise ValueError("conditioning diagnostics must be finite")
            rounded = np.rint(indices)
            if not np.array_equal(indices, rounded):
                raise ValueError("observation_indices must contain integer grid cells")
            self.observation_indices = rounded.astype(np.int64)
            self.observation_log10_k = values

    @property
    def count(self) -> int:
        return self._count

    @property
    def preview_count_saved(self) -> int:
        return len(self._previews)

    def _conditioning_residuals(self, field_values: Array) -> Array | None:
        if self.observation_indices is None or self.observation_log10_k is None:
            return None
        h, w = field_values.shape
        rows = self.observation_indices[:, 0]
        cols = self.observation_indices[:, 1]
        if np.any(rows < 0) or np.any(rows >= h) or np.any(cols < 0) or np.any(cols >= w):
            raise ValueError("conditioning observation index lies outside permeability field")
        return field_values[rows, cols] - self.observation_log10_k

    def update(self, permeability_batch: Array) -> None:
        batch = np.asarray(permeability_batch)
        if batch.ndim != 3:
            raise ValueError(f"permeability batch must have shape [B,H,W], got {batch.shape}")
        if not np.all(np.isfinite(batch)):
            raise ValueError("permeability batch contains non-finite values")
        if np.any(batch <= 0.0):
            raise ValueError("permeability must be strictly positive before log10 diagnostics")

        self._minimum_k = min(self._minimum_k, float(np.min(batch)))
        self._maximum_k = max(self._maximum_k, float(np.max(batch)))
        log_batch = np.log10(batch.astype(np.float64, copy=False))

        for physical_values, field_values in zip(batch, log_batch):
            if self._mean is None:
                self._mean = np.zeros_like(field_values, dtype=np.float64)
                self._m2 = np.zeros_like(field_values, dtype=np.float64)
            elif field_values.shape != self._mean.shape:
                raise ValueError(
                    "permeability field shape changed during diagnostics: "
                    f"expected {self._mean.shape}, got {field_values.shape}"
                )

            self._count += 1
            delta = field_values - self._mean
            self._mean += delta / self._count
            delta2 = field_values - self._mean
            self._m2 += delta * delta2

            if len(self._previews) < self.preview_count:
                self._previews.append(field_values.astype(np.float32, copy=True))

            if self.training_k_range is not None:
                low, high = self.training_k_range
                outside = (physical_values < low) | (physical_values > high)
                outside_count = int(np.count_nonzero(outside))
                total = int(physical_values.size)
                self._outside_count += outside_count
                self._total_cell_count += total
                self._outside_by_sample.append(outside_count / total)

            residuals = self._conditioning_residuals(field_values)
            if residuals is not None:
                max_abs = float(np.max(np.abs(residuals))) if residuals.size else 0.0
                self._conditioning_max_by_sample.append(max_abs)
                self._conditioning_sumsq += float(np.sum(residuals * residuals))
                self._conditioning_count += int(residuals.size)

    def finalize(self) -> PermeabilityDiagnosticsResult:
        if self._count == 0 or self._mean is None or self._m2 is None:
            raise RuntimeError("permeability diagnostics received no samples")
        denominator = self._count - self.ddof
        if denominator <= 0:
            raise ValueError(
                f"cannot compute variance with count={self._count} and ddof={self.ddof}"
            )
        variance = self._m2 / denominator
        variance = np.maximum(variance, 0.0)

        outside_fraction = None
        training_min = None
        training_max = None
        if self.training_k_range is not None:
            training_min, training_max = self.training_k_range
            outside_fraction = self._outside_count / self._total_cell_count

        conditioning_max = None
        conditioning_rms = None
        if self._conditioning_count:
            conditioning_max = max(self._conditioning_max_by_sample, default=0.0)
            conditioning_rms = float(
                np.sqrt(self._conditioning_sumsq / self._conditioning_count)
            )

        return PermeabilityDiagnosticsResult(
            count=self._count,
            mean_log10_k=self._mean.astype(np.float32, copy=True),
            variance_log10_k=variance.astype(np.float32, copy=True),
            std_log10_k=np.sqrt(variance).astype(np.float32, copy=False),
            preview_log10_k=tuple(field.copy() for field in self._previews),
            minimum_k=float(self._minimum_k),
            maximum_k=float(self._maximum_k),
            training_k_min=training_min,
            training_k_max=training_max,
            outside_training_fraction=outside_fraction,
            outside_training_fraction_by_sample=tuple(self._outside_by_sample),
            conditioning_max_abs_log10_residual=conditioning_max,
            conditioning_rms_log10_residual=conditioning_rms,
            conditioning_max_abs_log10_residual_by_sample=tuple(
                self._conditioning_max_by_sample
            ),
        )


@dataclass
class DiagnosticPermeabilitySampler:
    """Observe permeability batches and yield the same batches unchanged."""

    sampler: PermeabilitySampler
    diagnostics: PermeabilityDiagnostics

    def __len__(self) -> int:
        return len(self.sampler)

    def __iter__(self) -> Iterator[Array]:
        for batch in self.sampler:
            self.diagnostics.update(batch)
            yield batch
