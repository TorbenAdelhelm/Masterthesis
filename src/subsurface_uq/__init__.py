"""Modular uncertainty quantification for subsurface heat-plume surrogates."""

from .propagation import MonteCarloResult, MonteCarloRunner
from .sampling import (
    EmpiricalPermeabilitySampler,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
    PermeabilitySampler,
    Release25PerlinPermeabilitySampler,
    StochasticPermeabilityMap,
    load_empirical_fields,
    matern32_correlation_matrix,
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
    "GaussianCoordinatePermeabilitySampler",
    "KLLogGaussianPermeabilityMap",
    "MonteCarloResult",
    "MonteCarloRunner",
    "OnlineFieldStatistics",
    "PermeabilitySampler",
    "Release25PerlinPermeabilitySampler",
    "Release25Surrogate",
    "StochasticPermeabilityMap",
    "TemperatureAccumulator",
    "TemperatureSurrogate",
    "load_empirical_fields",
    "matern32_correlation_matrix",
]
