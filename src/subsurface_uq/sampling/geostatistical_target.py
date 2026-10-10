"""Explicit geostatistical target assumptions and descriptive validation metrics.

The target is expressed in hydraulic conductivity ``K_h [m/s]`` while the
LGCNN consumes intrinsic permeability ``k [m²]``.  Validation is descriptive:
realizations are never clipped, rejected, or resampled to satisfy the target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import cramervonmises, norm, wasserstein_distance

Array = np.ndarray


@dataclass(frozen=True)
class Base10LognormalTarget:
    """Pointwise/marginal target in base-10 log space.

    ``log10(K_h) ~ Normal(log10_mean, log10_std²)``.

    ``interval_m_s`` is a high-probability/validation interval, not hard support.
    A genuine lognormal distribution is unbounded on ``(0, +inf)``.
    """

    log10_mean: float = -3.0
    log10_std: float = 0.5
    interval_m_s: tuple[float, float] = (1.0e-4, 5.0e-2)
    minimum_interval_fraction: float = 0.95

    def __post_init__(self) -> None:
        lo, hi = (float(v) for v in self.interval_m_s)
        if not np.isfinite(self.log10_mean):
            raise ValueError("log10_mean must be finite")
        if not np.isfinite(self.log10_std) or self.log10_std <= 0.0:
            raise ValueError("log10_std must be finite and positive")
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= lo:
            raise ValueError("interval_m_s must contain finite positive ordered bounds")
        if not 0.0 < self.minimum_interval_fraction < 1.0:
            raise ValueError("minimum_interval_fraction must lie in (0,1)")
        object.__setattr__(self, "interval_m_s", (lo, hi))

    @property
    def log10_variance(self) -> float:
        return float(self.log10_std**2)

    @property
    def theoretical_interval_fraction(self) -> float:
        lo, hi = self.interval_m_s
        z_lo = (np.log10(lo) - self.log10_mean) / self.log10_std
        z_hi = (np.log10(hi) - self.log10_mean) / self.log10_std
        return float(norm.cdf(z_hi) - norm.cdf(z_lo))

    @property
    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "distribution": "base10_lognormal_hydraulic_conductivity",
            "definition": "log10(K_h [m/s]) ~ Normal(log10_mean, log10_std^2)",
            "log10_variance": self.log10_variance,
            "interval_semantics": "validation/high-probability target; not hard support",
            "theoretical_interval_fraction": self.theoretical_interval_fraction,
        }


def validate_hydraulic_conductivity_marginal(
    hydraulic_conductivity_m_s: Array,
    target: Base10LognormalTarget,
) -> dict[str, object]:
    """Return descriptive marginal diagnostics for an ensemble or one field.

    Pixelwise goodness-of-fit p-values are intentionally not reported because
    spatial correlation violates the iid assumptions used to interpret them.
    The Cramér-von Mises statistic is retained only as an effect-size-like
    descriptive discrepancy measure.
    """

    values = np.asarray(hydraulic_conductivity_m_s, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("hydraulic conductivity must contain finite positive values")
    logs = np.log10(values)
    standardized = (logs - target.log10_mean) / target.log10_std
    lo, hi = target.interval_m_s
    inside = (values >= lo) & (values <= hi)
    lower = values < lo
    upper = values > hi
    expected = np.random.default_rng(0).normal(
        target.log10_mean, target.log10_std, size=min(values.size, 200_000)
    )
    if values.size > expected.size:
        idx = np.linspace(0, values.size - 1, expected.size, dtype=np.int64)
        observed_for_wasserstein = logs[idx]
    else:
        observed_for_wasserstein = logs
    cvm = cramervonmises(standardized, "norm")
    fraction = float(np.mean(inside))
    return {
        "n_values": int(values.size),
        "log10_mean": float(np.mean(logs)),
        "log10_std": float(np.std(logs, ddof=1)) if logs.size > 1 else 0.0,
        "log10_variance": float(np.var(logs, ddof=1)) if logs.size > 1 else 0.0,
        "hydraulic_conductivity_quantiles_m_s": {
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        },
        "fraction_inside_interval": fraction,
        "lower_tail_fraction": float(np.mean(lower)),
        "upper_tail_fraction": float(np.mean(upper)),
        "minimum_interval_fraction": target.minimum_interval_fraction,
        "interval_fraction_status": "pass" if fraction >= target.minimum_interval_fraction else "warn",
        "cvm_normality_statistic_log10": float(cvm.statistic),
        "wasserstein_to_target_normal_log10": float(
            wasserstein_distance(observed_for_wasserstein, expected)
        ),
        "iid_caveat": (
            "Pixels are spatially correlated; goodness-of-fit statistics are descriptive and "
            "must not be interpreted using iid p-values."
        ),
        "no_filtering": True,
        "target": target.metadata,
    }
