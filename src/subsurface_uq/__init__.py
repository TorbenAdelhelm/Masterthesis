"""Modular uncertainty quantification for subsurface heat-plume surrogates."""

from .propagation import MonteCarloResult, MonteCarloRunner
from .sampling import (
    EmpiricalPermeabilitySampler,
    PermeabilitySampler,
    Release25PerlinPermeabilitySampler,
    load_empirical_fields,
)
from .statistics import (
    ExceedanceProbabilityAccumulator,
    FieldStatistics,
    FieldStatisticsAccumulator,
    OnlineFieldStatistics,
    TemperatureAccumulator,
)
from .surrogates import (
    BaseTemperatureSurrogate,
    CallableTemperatureSurrogate,
    Release25Surrogate,
    TemperatureSurrogate,
)

__all__ = [
    "BaseTemperatureSurrogate",
    "CallableTemperatureSurrogate",
    "EmpiricalPermeabilitySampler",
    "ExceedanceProbabilityAccumulator",
    "FieldStatistics",
    "FieldStatisticsAccumulator",
    "MonteCarloResult",
    "MonteCarloRunner",
    "OnlineFieldStatistics",
    "PermeabilitySampler",
    "Release25PerlinPermeabilitySampler",
    "Release25Surrogate",
    "TemperatureAccumulator",
    "TemperatureSurrogate",
    "load_empirical_fields",
]
