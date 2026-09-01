from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class TemperatureSurrogate(Protocol):
    """Model interface used by the UQ propagation layer."""

    def predict_temperature(self, permeability: Array) -> Array: ...

    def predict_temperature_batch(self, permeability_batch: Array) -> Array: ...


class BaseTemperatureSurrogate:
    """Base class with a safe sample-wise default batch implementation."""

    def predict_temperature(self, permeability: Array) -> Array:
        raise NotImplementedError

    def predict_temperature_batch(self, permeability_batch: Array) -> Array:
        batch = np.asarray(permeability_batch)
        if batch.ndim != 3:
            raise ValueError(
                f"permeability_batch must have shape [B,H,W], got {batch.shape}"
            )
        return np.stack([self.predict_temperature(field) for field in batch], axis=0)


@dataclass
class CallableTemperatureSurrogate(BaseTemperatureSurrogate):
    """Small adapter for deterministic ``K -> T`` callables."""

    fn: Callable[[Array], Array]

    def predict_temperature(self, permeability: Array) -> Array:
        result = np.asarray(self.fn(np.asarray(permeability)))
        if result.ndim != 2:
            raise ValueError(
                f"temperature callable must return [H,W], got {result.shape}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError("temperature prediction contains non-finite values")
        return result.astype(np.float32, copy=False)
