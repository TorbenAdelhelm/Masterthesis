from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..sampling.base import PermeabilitySampler
from ..statistics import (
    ExceedanceProbabilityAccumulator,
    ExceedanceStatistics,
    FieldStatistics,
    FieldStatisticsAccumulator,
    TemperatureAccumulator,
)
from ..surrogates.base import TemperatureSurrogate

Array = np.ndarray


@dataclass(frozen=True)
class MonteCarloResult:
    count: int
    mean: Array
    variance: Array
    std: Array
    minimum: Array
    maximum: Array
    samples: Array | None = None
    exceedance_thresholds: tuple[float, ...] = ()
    exceedance_probabilities: Array | None = None
    background_temperature: float | None = None
    accumulator_results: Mapping[str, object] = field(default_factory=dict)


@dataclass
class MonteCarloRunner:
    """Propagate permeability realizations through a deterministic surrogate.

    The propagation loop is intentionally model- and QoI-agnostic. For samples
    ``K^(m)`` from the configured permeability sampler it evaluates

    ``T^(m) = F(K^(m))``

    with the deterministic temperature surrogate ``F`` and forwards every
    temperature batch to streaming ``TemperatureAccumulator`` objects. The
    built-in field-statistics accumulator estimates the spatial Monte Carlo mean
    and variance without retaining all realizations. Additional QoIs can be
    attached through ``accumulators`` without modifying this loop.

    ``background_temperature`` and ``exceedance_thresholds`` remain as a
    backwards-compatible convenience interface; internally they construct an
    ``ExceedanceProbabilityAccumulator``.
    """

    sampler: PermeabilitySampler
    surrogate: TemperatureSurrogate

    def run(
        self,
        *,
        n_samples: int | None = None,
        store_all: bool = False,
        ddof: int = 1,
        background_temperature: float | None = None,
        exceedance_thresholds: Sequence[float] = (),
        accumulators: Sequence[TemperatureAccumulator] = (),
    ) -> MonteCarloResult:
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be positive or None")

        thresholds = tuple(float(value) for value in exceedance_thresholds)
        if thresholds and background_temperature is None:
            raise ValueError(
                "background_temperature is required when exceedance thresholds are requested"
            )

        configured: list[TemperatureAccumulator] = [FieldStatisticsAccumulator(ddof=ddof)]
        configured.extend(accumulators)
        if thresholds:
            configured.append(
                ExceedanceProbabilityAccumulator(
                    thresholds,
                    background_temperature=float(background_temperature),
                )
            )

        names = [str(accumulator.name) for accumulator in configured]
        if any(not name for name in names):
            raise ValueError("temperature accumulator names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"temperature accumulator names must be unique, got {names}")

        stored: list[Array] | None = [] if store_all else None
        seen = 0

        for permeability_batch in self.sampler:
            batch = np.asarray(permeability_batch)
            if batch.ndim != 3:
                raise ValueError(
                    f"sampler must yield [B,H,W], got {batch.shape}"
                )
            if n_samples is not None:
                remaining = n_samples - seen
                if remaining <= 0:
                    break
                batch = batch[:remaining]
            if batch.shape[0] == 0:
                continue

            batch_predict = getattr(self.surrogate, "predict_temperature_batch", None)
            if callable(batch_predict):
                temperatures = np.asarray(batch_predict(batch))
            else:
                temperatures = np.stack(
                    [self.surrogate.predict_temperature(field) for field in batch],
                    axis=0,
                )
            if temperatures.ndim != 3 or temperatures.shape[0] != batch.shape[0]:
                raise ValueError(
                    "surrogate batch output must have shape [B,H,W]; "
                    f"got {temperatures.shape} for input {batch.shape}"
                )

            for accumulator in configured:
                accumulator.update(temperatures)
            if stored is not None:
                stored.append(temperatures.astype(np.float32, copy=True))
            seen += int(batch.shape[0])

        if seen == 0:
            raise RuntimeError("Monte Carlo propagation produced no samples")

        finalized = {accumulator.name: accumulator.finalize() for accumulator in configured}
        summary = finalized.pop("field_statistics")
        if not isinstance(summary, FieldStatistics):
            raise TypeError("field_statistics accumulator returned an unexpected result type")

        exceedance_summary = finalized.get("exceedance")
        if exceedance_summary is not None and not isinstance(
            exceedance_summary, ExceedanceStatistics
        ):
            raise TypeError("exceedance accumulator returned an unexpected result type")

        samples = None if stored is None else np.concatenate(stored, axis=0)
        return MonteCarloResult(
            count=summary.count,
            mean=summary.mean,
            variance=summary.variance,
            std=summary.std,
            minimum=summary.minimum,
            maximum=summary.maximum,
            samples=samples,
            exceedance_thresholds=(
                () if exceedance_summary is None else exceedance_summary.thresholds
            ),
            exceedance_probabilities=(
                None
                if exceedance_summary is None
                else exceedance_summary.probabilities
            ),
            background_temperature=(
                None if exceedance_summary is None else float(background_temperature)
            ),
            accumulator_results=finalized,
        )
