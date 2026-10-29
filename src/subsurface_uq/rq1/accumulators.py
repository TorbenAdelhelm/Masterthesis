from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.lib.format import open_memmap

from ..statistics import FieldStatistics, OnlineFieldStatistics

Array = np.ndarray


@dataclass(frozen=True)
class RQ1QoISamples:
    mean_anomaly: Array | None
    receptor_values: Array
    receptor_indices: tuple[tuple[int, int], ...]


@dataclass
class RQ1QoIAccumulator:
    """Collect the continuous scalar QoIs retained by the finalized RQ1 design."""

    background_temperature: float
    receptors: Sequence[tuple[int, int]]
    mean_anomaly_roi: tuple[int, int, int, int] | None
    name: str = "rq1_qoi"
    _mean_anomaly: list[float] = field(default_factory=list, init=False, repr=False)
    _receptors: list[Array] = field(default_factory=list, init=False, repr=False)
    _shape: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)

    def _validate_shape(self, shape: tuple[int, int]) -> None:
        if self._shape is not None:
            if self._shape != shape:
                raise ValueError(f"temperature shape changed from {self._shape} to {shape}")
            return
        h, w = shape
        if self.mean_anomaly_roi is not None:
            r0, r1, c0, c1 = self.mean_anomaly_roi
            if r0 < 0 or c0 < 0 or r1 > h or c1 > w or r1 <= r0 or c1 <= c0:
                raise ValueError(
                    f"mean_anomaly_roi {self.mean_anomaly_roi} lies outside temperature shape {shape}"
                )
        for row, col in self.receptors:
            if row < 0 or row >= h or col < 0 or col >= w:
                raise ValueError(f"receptor {(row, col)} lies outside temperature shape {shape}")
        self._shape = shape

    def update(self, temperatures: Array) -> None:
        batch = np.asarray(temperatures, dtype=np.float64)
        if batch.ndim != 3:
            raise ValueError(f"temperature batch must have shape [B,H,W], got {batch.shape}")
        if not np.all(np.isfinite(batch)):
            raise ValueError("temperature batch contains non-finite values")
        self._validate_shape((int(batch.shape[1]), int(batch.shape[2])))

        if self.mean_anomaly_roi is not None:
            r0, r1, c0, c1 = self.mean_anomaly_roi
            anomaly = batch[:, r0:r1, c0:c1] - float(self.background_temperature)
            self._mean_anomaly.extend(np.mean(anomaly, axis=(1, 2)).tolist())

        if self.receptors:
            values = np.stack(
                [batch[:, row, col] for row, col in self.receptors],
                axis=1,
            )
        else:
            values = np.empty((batch.shape[0], 0), dtype=np.float64)
        self._receptors.extend(values)
        self._count += int(batch.shape[0])

    def finalize(self) -> RQ1QoISamples:
        if self._count == 0:
            raise RuntimeError("RQ1 QoI accumulator received no temperature samples")
        receptor_values = np.asarray(self._receptors, dtype=np.float64)
        if receptor_values.ndim == 1:
            receptor_values = receptor_values.reshape(self._count, 0)
        mean_anomaly = None
        if self.mean_anomaly_roi is not None:
            if len(self._mean_anomaly) != self._count:
                raise RuntimeError("mean-anomaly QoI sample count is inconsistent")
            mean_anomaly = np.asarray(self._mean_anomaly, dtype=np.float64)
        return RQ1QoISamples(
            mean_anomaly=mean_anomaly,
            receptor_values=receptor_values,
            receptor_indices=tuple((int(r), int(c)) for r, c in self.receptors),
        )


@dataclass(frozen=True)
class DiskTemperatureStoreResult:
    path: Path
    count: int
    field_shape: tuple[int, int]


@dataclass
class DiskTemperatureStoreAccumulator:
    """Write exact temperature realizations to a disk-backed NumPy array."""

    path: str | Path
    expected_samples: int
    name: str = "rq1_temperature_store"
    _store: Array | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False, repr=False)
    _shape: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        self.expected_samples = int(self.expected_samples)
        if self.expected_samples <= 0:
            raise ValueError("expected_samples must be positive")

    def update(self, temperatures: Array) -> None:
        batch = np.asarray(temperatures, dtype=np.float32)
        if batch.ndim != 3:
            raise ValueError(f"temperature batch must have shape [B,H,W], got {batch.shape}")
        if not np.all(np.isfinite(batch)):
            raise ValueError("temperature batch contains non-finite values")
        shape = (int(batch.shape[1]), int(batch.shape[2]))
        if self._store is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._shape = shape
            self._store = open_memmap(
                self.path,
                mode="w+",
                dtype=np.float32,
                shape=(self.expected_samples, *shape),
            )
        elif self._shape != shape:
            raise ValueError(f"temperature field shape changed from {self._shape} to {shape}")
        stop = self._count + int(batch.shape[0])
        if stop > self.expected_samples:
            raise ValueError("temperature store received more samples than expected")
        self._store[self._count:stop] = batch
        self._count = stop

    def finalize(self) -> DiskTemperatureStoreResult:
        if self._store is None or self._shape is None or self._count == 0:
            raise RuntimeError("temperature store received no samples")
        if self._count != self.expected_samples:
            raise RuntimeError(
                f"temperature store expected {self.expected_samples} samples, got {self._count}"
            )
        self._store.flush()
        mmap_handle = getattr(self._store, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        self._store = None
        return DiskTemperatureStoreResult(
            path=Path(self.path),
            count=self._count,
            field_shape=self._shape,
        )


@dataclass(frozen=True)
class ReferenceConvergencePoint:
    budget: int
    e_mu: float
    e_sigma: float
    u_rms: float
    u95: float
    mean_std: float
    max_std: float


@dataclass
class ReferenceConvergenceFieldStatisticsAccumulator:
    """Field statistics plus checkpoint errors against an empirical MC reference.

    This object can replace the default field-statistics accumulator in
    MonteCarloRunner. It stores only the running moments and scalar checkpoint
    diagnostics, avoiding disk-backed full-field storage for repeated MC/RQMC
    convergence runs.
    """

    checkpoints: Sequence[int]
    reference_mean: Array
    reference_std: Array
    ddof: int = 1
    name: str = "field_statistics"
    _statistics: OnlineFieldStatistics = field(
        default_factory=OnlineFieldStatistics, init=False, repr=False
    )
    _points: list[ReferenceConvergencePoint] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        values = tuple(int(v) for v in self.checkpoints)
        if not values or sorted(set(values)) != list(values):
            raise ValueError("checkpoints must be unique and strictly increasing")
        if values[0] <= self.ddof:
            raise ValueError("every checkpoint must be greater than ddof")
        self.checkpoints = values
        self.reference_mean = np.asarray(self.reference_mean, dtype=np.float64)
        self.reference_std = np.asarray(self.reference_std, dtype=np.float64)
        if self.reference_mean.shape != self.reference_std.shape:
            raise ValueError("reference mean/std shapes do not match")
        if self.reference_mean.ndim != 2:
            raise ValueError("reference mean/std must be 2-D fields")
        if not np.all(np.isfinite(self.reference_mean)) or not np.all(
            np.isfinite(self.reference_std)
        ):
            raise ValueError("reference fields must be finite")

    @property
    def points(self) -> tuple[ReferenceConvergencePoint, ...]:
        return tuple(self._points)

    @staticmethod
    def _global_metrics(summary: FieldStatistics) -> tuple[float, float, float, float]:
        variance = np.asarray(summary.variance, dtype=np.float64)
        std = np.asarray(summary.std, dtype=np.float64)
        return (
            float(np.sqrt(np.mean(variance))),
            float(np.quantile(std, 0.95)),
            float(np.mean(std)),
            float(np.max(std)),
        )

    def _snapshot(self) -> None:
        summary = self._statistics.finalize(ddof=self.ddof)
        if summary.mean.shape != self.reference_mean.shape:
            raise ValueError(
                "reference temperature shape does not match repeated-run temperature shape"
            )
        e_mu = float(
            np.sqrt(
                np.mean(
                    (np.asarray(summary.mean, dtype=np.float64) - self.reference_mean) ** 2
                )
            )
        )
        e_sigma = float(
            np.sqrt(
                np.mean(
                    (np.asarray(summary.std, dtype=np.float64) - self.reference_std) ** 2
                )
            )
        )
        u_rms, u95, mean_std, max_std = self._global_metrics(summary)
        self._points.append(
            ReferenceConvergencePoint(
                budget=int(summary.count),
                e_mu=e_mu,
                e_sigma=e_sigma,
                u_rms=u_rms,
                u95=u95,
                mean_std=mean_std,
                max_std=max_std,
            )
        )

    def update(self, temperatures: Array) -> None:
        batch = np.asarray(temperatures)
        if batch.ndim != 3:
            raise ValueError(f"temperature batch must have shape [B,H,W], got {batch.shape}")
        wanted = set(self.checkpoints)
        for field_values in batch:
            self._statistics.update(field_values)
            if self._statistics.count in wanted:
                self._snapshot()

    def finalize(self) -> FieldStatistics:
        if self._statistics.count < self.checkpoints[-1]:
            raise RuntimeError(
                f"only {self._statistics.count} samples reached; expected at least "
                f"{self.checkpoints[-1]}"
            )
        return self._statistics.finalize(ddof=self.ddof)
