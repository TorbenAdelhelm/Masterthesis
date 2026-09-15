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


@dataclass
class PermeabilityDiagnostics:
    """Accumulate ``log10(K)`` statistics without retaining the full ensemble."""

    preview_count: int = 3
    ddof: int = 1
    _count: int = field(default=0, init=False, repr=False)
    _mean: Array | None = field(default=None, init=False, repr=False)
    _m2: Array | None = field(default=None, init=False, repr=False)
    _previews: list[Array] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.preview_count = int(self.preview_count)
        self.ddof = int(self.ddof)
        if self.preview_count < 0:
            raise ValueError("preview_count must be non-negative")
        if self.ddof < 0:
            raise ValueError("ddof must be non-negative")

    @property
    def count(self) -> int:
        return self._count

    @property
    def preview_count_saved(self) -> int:
        return len(self._previews)

    def update(self, permeability_batch: Array) -> None:
        batch = np.asarray(permeability_batch)
        if batch.ndim != 3:
            raise ValueError(f"permeability batch must have shape [B,H,W], got {batch.shape}")
        if not np.all(np.isfinite(batch)):
            raise ValueError("permeability batch contains non-finite values")
        if np.any(batch <= 0.0):
            raise ValueError("permeability must be strictly positive before log10 diagnostics")

        log_batch = np.log10(batch.astype(np.float64, copy=False))
        for field_values in log_batch:
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
        return PermeabilityDiagnosticsResult(
            count=self._count,
            mean_log10_k=self._mean.astype(np.float32, copy=True),
            variance_log10_k=variance.astype(np.float32, copy=True),
            std_log10_k=np.sqrt(variance).astype(np.float32, copy=False),
            preview_log10_k=tuple(field.copy() for field in self._previews),
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
