from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class TemperatureFunctional(Protocol):
    """Map a batch of physical temperature fields ``[B,H,W]`` to scalar QoIs."""

    @property
    def name(self) -> str: ...

    def evaluate_batch(self, temperatures: Array) -> Array: ...


@dataclass(frozen=True)
class MeanTemperatureAnomaly:
    """Spatial mean of ``T - T_background`` for each sample.

    This is intentionally continuous and is therefore a suitable first scalar
    target for global polynomial chaos. Thresholded plume areas and exceedance
    indicators are deferred because their discontinuities can degrade PCE
    convergence.
    """

    background_temperature: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.background_temperature):
            raise ValueError("background_temperature must be finite")

    @property
    def name(self) -> str:
        return "mean_temperature_anomaly"

    def evaluate_batch(self, temperatures: Array) -> Array:
        temperatures = np.asarray(temperatures, dtype=np.float64)
        if temperatures.ndim != 3:
            raise ValueError(
                "temperatures must have shape [B,H,W], "
                f"got {temperatures.shape}"
            )
        if not np.all(np.isfinite(temperatures)):
            raise ValueError("temperatures must be finite")
        anomaly = temperatures - float(self.background_temperature)
        return np.mean(anomaly, axis=(1, 2), dtype=np.float64)
