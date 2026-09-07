from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ExceedanceStatistics:
    """Per-cell empirical exceedance probabilities for Delta T thresholds."""

    count: int
    thresholds: tuple[float, ...]
    probabilities: Array


class OnlineExceedanceStatistics:
    """Streaming exceedance counts without retaining Monte Carlo samples.

    For each threshold ``tau`` this estimates

    ``P(T - background_temperature >= tau)``

    independently at every grid cell. ``update`` accepts either one ``[H,W]``
    temperature field or a batch ``[B,H,W]``.
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        *,
        background_temperature: float,
    ) -> None:
        values = tuple(float(value) for value in thresholds)
        if not values:
            raise ValueError("at least one exceedance threshold is required")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("exceedance thresholds must be finite")
        if len(set(values)) != len(values):
            raise ValueError("exceedance thresholds must be unique")
        if not np.isfinite(background_temperature):
            raise ValueError("background_temperature must be finite")

        self.thresholds = values
        self.background_temperature = float(background_temperature)
        self.count = 0
        self._counts: Array | None = None

    @property
    def field_shape(self) -> tuple[int, int] | None:
        if self._counts is None:
            return None
        return tuple(int(value) for value in self._counts.shape[1:])

    def update(self, temperatures: Array) -> None:
        batch = np.asarray(temperatures)
        if batch.ndim == 2:
            batch = batch[None, ...]
        if batch.ndim != 3:
            raise ValueError(f"expected [H,W] or [B,H,W], got {batch.shape}")
        if batch.shape[0] == 0:
            return
        if not np.all(np.isfinite(batch)):
            raise ValueError("temperature input contains non-finite values")

        shape = tuple(int(value) for value in batch.shape[1:])
        if self._counts is None:
            self._counts = np.zeros((len(self.thresholds), *shape), dtype=np.uint64)
        elif self._counts.shape[1:] != shape:
            raise ValueError(
                f"field shape changed from {self._counts.shape[1:]} to {shape}"
            )

        assert self._counts is not None
        background = np.asarray(self.background_temperature, dtype=batch.dtype)
        delta = batch - background
        for index, threshold in enumerate(self.thresholds):
            # Cast the threshold to the field dtype before comparison. This
            # avoids treating a nominal float32 boundary such as 0.1 as being
            # spuriously above its intended threshold because of representation.
            threshold_value = np.asarray(threshold, dtype=batch.dtype)
            self._counts[index] += np.sum(
                delta >= threshold_value,
                axis=0,
                dtype=np.uint64,
            )
        self.count += int(batch.shape[0])

    def finalize(self) -> ExceedanceStatistics:
        if self.count == 0 or self._counts is None:
            raise RuntimeError("no temperature fields have been accumulated")
        probabilities = self._counts.astype(np.float64) / float(self.count)
        return ExceedanceStatistics(
            count=self.count,
            thresholds=self.thresholds,
            probabilities=probabilities.astype(np.float32),
        )
