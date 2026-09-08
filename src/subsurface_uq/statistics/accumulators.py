from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .exceedance import ExceedanceStatistics, OnlineExceedanceStatistics
from .online import FieldStatistics, OnlineFieldStatistics

Array = np.ndarray


@runtime_checkable
class TemperatureAccumulator(Protocol):
    """Streaming statistic/QoI evaluated on temperature batches.

    Accumulators receive physical temperature fields with shape ``[B,H,W]``.
    They are deliberately independent of the permeability sampler and surrogate,
    which allows new quantities of interest to be added without changing the
    Monte Carlo propagation loop.
    """

    @property
    def name(self) -> str: ...

    def update(self, temperatures: Array) -> None: ...

    def finalize(self) -> object: ...


@dataclass
class FieldStatisticsAccumulator:
    """Accumulator for per-cell mean, variance, extrema and standard deviation."""

    ddof: int = 1
    name: str = "field_statistics"
    _statistics: OnlineFieldStatistics = field(
        default_factory=OnlineFieldStatistics, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.ddof < 0:
            raise ValueError("ddof must be non-negative")

    def update(self, temperatures: Array) -> None:
        self._statistics.update(temperatures)

    def finalize(self) -> FieldStatistics:
        return self._statistics.finalize(ddof=self.ddof)


@dataclass
class ExceedanceProbabilityAccumulator:
    """Accumulator for empirical spatial probabilities ``P(T-T_bg >= tau)``."""

    thresholds: Sequence[float]
    background_temperature: float
    name: str = "exceedance"
    _statistics: OnlineExceedanceStatistics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._statistics = OnlineExceedanceStatistics(
            self.thresholds,
            background_temperature=self.background_temperature,
        )

    def update(self, temperatures: Array) -> None:
        self._statistics.update(temperatures)

    def finalize(self) -> ExceedanceStatistics:
        return self._statistics.finalize()
