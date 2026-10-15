"""Explicit geostatistical target assumptions and descriptive validation metrics.

The target is expressed in hydraulic conductivity ``K_h [m/s]`` while the
LGCNN consumes intrinsic permeability ``k [m²]``. Validation is descriptive:
realizations are never clipped, rejected, or resampled to satisfy the target.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import cramervonmises, kurtosis, kstest, norm, skew, wasserstein_distance

Array = np.ndarray


@dataclass(frozen=True)
class Base10LognormalTarget:
    """Optional global histogram diagnostic target in base-10 log space.

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

    @classmethod
    def from_mapping(cls, payload: dict[str, object] | None) -> "Base10LognormalTarget":
        """Build a target while making std-vs-variance semantics explicit.

        A caller may provide exactly one of ``log10_std`` and ``log10_variance``.
        The teaching material supplied for this thesis states ``sigma_log10=0.5``;
        therefore the documented default is a standard deviation of 0.5 and a
        corresponding base-10 log variance of 0.25. A true variance target of 0.5
        remains possible by specifying ``log10_variance: 0.5`` explicitly.
        """

        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("distribution_target must be a mapping")
        allowed = {
            "distribution", "log10_mean", "log10_std", "log10_variance",
            "interval_m_s", "minimum_interval_fraction",
        }
        if not set(payload) <= allowed:
            raise ValueError("unknown distribution_target keys")
        distribution = payload.get("distribution", "base10_lognormal_hydraulic_conductivity")
        if distribution != "base10_lognormal_hydraulic_conductivity":
            raise ValueError("only base10_lognormal_hydraulic_conductivity is supported")
        has_std = "log10_std" in payload
        has_var = "log10_variance" in payload
        if has_std and has_var:
            raise ValueError("choose log10_std OR log10_variance, not both")
        if has_var:
            variance = float(payload["log10_variance"])
            if not np.isfinite(variance) or variance <= 0.0:
                raise ValueError("log10_variance must be finite and positive")
            log10_std = float(np.sqrt(variance))
        else:
            log10_std = float(payload.get("log10_std", 0.5))
        interval = tuple(payload.get("interval_m_s", (1.0e-4, 5.0e-2)))
        if len(interval) != 2:
            raise ValueError("interval_m_s must contain exactly two bounds")
        return cls(
            log10_mean=float(payload.get("log10_mean", -3.0)),
            log10_std=log10_std,
            interval_m_s=(float(interval[0]), float(interval[1])),
            minimum_interval_fraction=float(payload.get("minimum_interval_fraction", 0.95)),
        )

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
            "source_semantics": (
                "The documented geostatistical example uses sigma_log10=0.5; "
                "this is a standard deviation, not a variance."
            ),
        }


def _positive_values(values: Array) -> Array:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("hydraulic conductivity must contain finite positive values")
    return values


def _bounded_logs(logs: Array, max_values: int = 200_000) -> Array:
    logs = np.asarray(logs, dtype=np.float64).reshape(-1)
    if logs.size <= max_values:
        return logs
    index = np.linspace(0, logs.size - 1, max_values, dtype=np.int64)
    return logs[index]


def _qq_rmse_standard_normal(standardized: Array) -> float:
    z = np.sort(_bounded_logs(standardized))
    if z.size < 2:
        return 0.0
    probabilities = (np.arange(z.size, dtype=np.float64) + 0.5) / z.size
    expected = norm.ppf(probabilities)
    return float(np.sqrt(np.mean((z - expected) ** 2)))


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _normality_metrics(logs: Array, target: Base10LognormalTarget) -> dict[str, object]:
    sampled = _bounded_logs(logs)
    standardized = (sampled - target.log10_mean) / target.log10_std
    cvm = cramervonmises(standardized, "norm")
    ks = kstest(standardized, "norm")
    probabilities = (np.arange(sampled.size, dtype=np.float64) + 0.5) / sampled.size
    expected_logs = target.log10_mean + target.log10_std * norm.ppf(probabilities)
    observed_sorted = np.sort(sampled)
    raw_skew = skew(sampled, bias=False) if sampled.size > 2 else 0.0
    raw_kurtosis = kurtosis(sampled, fisher=True, bias=False) if sampled.size > 3 else 0.0
    return {
        "skewness_log10": _finite_or_none(raw_skew),
        "excess_kurtosis_log10": _finite_or_none(raw_kurtosis),
        "qq_rmse_standardized": _qq_rmse_standard_normal(standardized),
        "cvm_normality_statistic_log10": float(cvm.statistic),
        "ks_normality_statistic_log10": float(ks.statistic),
        "wasserstein_to_target_normal_log10": float(wasserstein_distance(observed_sorted, expected_logs)),
        "undefined_shape_moments": bool(not np.isfinite(raw_skew) or not np.isfinite(raw_kurtosis)),
    }


def _validated_tolerances(tolerances):
    if tolerances is None:
        return None
    allowed = {"max_abs_log10_mean_error", "max_abs_log10_std_error", "max_qq_rmse_standardized"}
    if not isinstance(tolerances, dict) or not set(tolerances) <= allowed:
        raise ValueError("unknown distribution validation tolerance")
    limits = {key: None for key in sorted(allowed)}
    for key, value in tolerances.items():
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError("distribution validation tolerances must be finite and non-negative")
            limits[key] = value
    return limits


def _heuristic_assessment(metrics: dict[str, object], tolerances: dict[str, float] | None) -> dict[str, object]:
    if metrics["target"] is None:
        return {"status": "not_assessed", "interpretation": "No global distribution target was configured."}
    if tolerances is None:
        return {
            "status": "descriptive_only",
            "interpretation": "No acceptance thresholds were supplied; diagnostics do not filter samples.",
        }
    limits = _validated_tolerances(tolerances)
    def check(key, observed):
        return None if limits[key] is None else bool(abs(observed) <= limits[key])
    checks = {
        "interval_fraction": bool(metrics["fraction_inside_interval"] >= metrics["minimum_interval_fraction"]),
        "log10_mean": check("max_abs_log10_mean_error", metrics["log10_mean_error"]),
        "log10_std": check("max_abs_log10_std_error", metrics["log10_std_error"]),
        "qq_shape": check("max_qq_rmse_standardized", metrics["normality_deviation"]["qq_rmse_standardized"]),
    }
    return {
        "status": "heuristically_compatible" if all(v is not False for v in checks.values()) else "heuristic_deviation",
        "checks": checks,
        "tolerances": limits,
        "interpretation": "Heuristic thresholds are user assumptions, not a statistical proof or sample-selection rule.",
    }


def validate_hydraulic_conductivity_marginal(
    hydraulic_conductivity_m_s: Array,
    target: Base10LognormalTarget | None = None,
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return descriptive marginal diagnostics for an ensemble or one field.

    Pixelwise goodness-of-fit p-values are intentionally not reported because
    spatial correlation violates the iid assumptions used to interpret them.
    Distribution discrepancies are effect-size diagnostics only. Without an
    explicit target, moments and physical quantiles remain available, while
    target discrepancies, interval counts and normality checks are unassessed.
    """

    tolerances = _validated_tolerances(tolerances)
    if target is None and tolerances is not None:
        raise ValueError("distribution validation tolerances require an explicit distribution target")
    values = _positive_values(hydraulic_conductivity_m_s)
    logs = np.log10(values)
    lo, hi = target.interval_m_s if target is not None else (None, None)
    fraction = float(np.mean((values >= lo) & (values <= hi))) if target is not None else None
    observed_mean = float(np.mean(logs))
    observed_std = float(np.std(logs, ddof=1)) if logs.size > 1 else 0.0
    observed_variance = float(np.var(logs, ddof=1)) if logs.size > 1 else 0.0
    metrics: dict[str, object] = {
        "n_values": int(values.size),
        "log10_mean": observed_mean,
        "log10_std": observed_std,
        "log10_variance": observed_variance,
        "log10_mean_error": observed_mean - target.log10_mean if target is not None else None,
        "log10_std_error": observed_std - target.log10_std if target is not None else None,
        "log10_variance_error": observed_variance - target.log10_variance if target is not None else None,
        "hydraulic_conductivity_quantiles_m_s": {
            "q01": float(np.quantile(values, 0.01)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
            "q99": float(np.quantile(values, 0.99)),
        },
        "fraction_inside_interval": fraction,
        "lower_tail_fraction": float(np.mean(values < lo)) if target is not None else None,
        "upper_tail_fraction": float(np.mean(values > hi)) if target is not None else None,
        "minimum_interval_fraction": target.minimum_interval_fraction if target is not None else None,
        "interval_fraction_status": ("pass" if fraction >= target.minimum_interval_fraction else "warn") if target is not None else "not_assessed",
        "normality_deviation": _normality_metrics(logs, target) if target is not None else None,
        "iid_caveat": (
            "Pixels are spatially correlated; goodness-of-fit statistics are descriptive and "
            "must not be interpreted using iid p-values."
        ),
        "no_filtering": True,
        "target": target.metadata if target is not None else None,
    }
    metrics["heuristic_assessment"] = _heuristic_assessment(metrics, tolerances)
    return metrics


class HydraulicConductivityValidationAccumulator:
    """Bounded-memory, descriptive pixel summaries within one scenario.

    Global target comparisons require an explicit target. The diagnostic sample
    uses its own seeded RNG and cannot alter the generator's Gaussian coordinates.
    """

    def __init__(
        self,
        target: Base10LognormalTarget | None = None,
        *,
        tolerances: dict[str, float] | None = None,
        max_pooled_sample_values: int = 200_000,
        sample_seed: int = 0,
    ) -> None:
        if max_pooled_sample_values < 100:
            raise ValueError("max_pooled_sample_values must be at least 100")
        self.target = target
        self.tolerances = _validated_tolerances(tolerances)
        if target is None and self.tolerances is not None:
            raise ValueError("distribution validation tolerances require an explicit distribution target")
        if isinstance(sample_seed, bool) or not isinstance(sample_seed, int) or sample_seed < 0:
            raise ValueError("sample_seed must be a non-negative integer")
        self.sample_seed = sample_seed
        self._rng = np.random.default_rng(sample_seed)
        self.max_pooled_sample_values = int(max_pooled_sample_values)
        self._n = 0
        self._sum = 0.0
        self._sum2 = 0.0
        self._inside = 0
        self._lower = 0
        self._upper = 0
        self._sample_logs = np.empty(0, dtype=np.float64)
        self._per_field: list[dict[str, object]] = []

    def _update_log_sample(self, logs: Array) -> None:
        """Merge uniform pixel samples, preserving population rather than batch weights.

        Keep all values until capacity is reached. Thereafter each buffer entry
        is an independent uniform draw with replacement from all pixels seen.
        Replace an entry with probability new_count / total_count. This avoids
        repeatedly thinning old fields and has memory cost O(capacity), even
        when the total population exceeds NumPy's hypergeometric size limits.
        """
        total = self._n + logs.size
        capacity = self.max_pooled_sample_values
        if total <= capacity:
            self._sample_logs = np.concatenate((self._sample_logs, logs))
            return
        if self._n <= capacity:
            # Transition from exact stored values to independent reservoir draws.
            choose_new = self._rng.random(capacity) < logs.size / total
            sample = np.empty(capacity, dtype=np.float64)
            old_count = int(np.count_nonzero(~choose_new))
            if old_count:
                sample[~choose_new] = self._sample_logs[self._rng.integers(self._n, size=old_count)]
            new_count = int(np.count_nonzero(choose_new))
            sample[choose_new] = logs[self._rng.integers(logs.size, size=new_count)]
            self._sample_logs = sample
        else:
            replace = self._rng.random(capacity) < logs.size / total
            count = int(np.count_nonzero(replace))
            self._sample_logs[replace] = logs[self._rng.integers(logs.size, size=count)]

    def update(self, hydraulic_conductivity_m_s: Array) -> None:
        values = _positive_values(hydraulic_conductivity_m_s)
        logs = np.log10(values)
        self._update_log_sample(logs)
        self._n += int(logs.size)
        self._sum += float(np.sum(logs, dtype=np.float64))
        self._sum2 += float(np.sum(logs * logs, dtype=np.float64))
        if self.target is not None:
            lo, hi = self.target.interval_m_s
            self._inside += int(np.count_nonzero((values >= lo) & (values <= hi)))
            self._lower += int(np.count_nonzero(values < lo))
            self._upper += int(np.count_nonzero(values > hi))
        self._per_field.append(validate_hydraulic_conductivity_marginal(
            values, self.target, tolerances=self.tolerances
        ))

    def finalize(self) -> dict[str, object]:
        if self._n == 0:
            raise ValueError("no hydraulic-conductivity fields were accumulated")
        mean = self._sum / self._n
        variance = ((self._sum2 - self._n * mean * mean) / (self._n - 1)) if self._n > 1 else 0.0
        variance = max(float(variance), 0.0)
        std = float(np.sqrt(variance))
        sampled_values = np.power(10.0, self._sample_logs)
        sampled_metrics = validate_hydraulic_conductivity_marginal(
            sampled_values, self.target, tolerances=self.tolerances
        )
        pooled = dict(sampled_metrics)
        pooled.update({
            "n_values": self._n,
            "log10_mean": float(mean),
            "log10_std": std,
            "log10_variance": variance,
            "log10_mean_error": float(mean - self.target.log10_mean) if self.target is not None else None,
            "log10_std_error": float(std - self.target.log10_std) if self.target is not None else None,
            "log10_variance_error": float(variance - self.target.log10_variance) if self.target is not None else None,
            "fraction_inside_interval": float(self._inside / self._n) if self.target is not None else None,
            "lower_tail_fraction": float(self._lower / self._n) if self.target is not None else None,
            "upper_tail_fraction": float(self._upper / self._n) if self.target is not None else None,
            "quantile_normality_sample_size": int(self._sample_logs.size),
            "quantile_normality_sampling": {
                "method": "uniform_pixels_with_replacement" if self._n > self.max_pooled_sample_values else "all_values",
                "seed": self.sample_seed, "capacity": self.max_pooled_sample_values,
                "population_size": self._n,
                "interpretation": "Descriptive pixel pool; no geological scenario probabilities. Moments and interval counts use every pixel.",
            },
        })
        pooled["interval_fraction_status"] = (
            ("pass" if pooled["fraction_inside_interval"] >= self.target.minimum_interval_fraction else "warn")
            if self.target is not None else "not_assessed"
        )
        pooled["heuristic_assessment"] = _heuristic_assessment(pooled, self.tolerances)
        return {
            "pooled": pooled,
            "per_field": self._per_field,
            "field_count": len(self._per_field),
            "target": self.target.metadata if self.target is not None else None,
            "no_filtering": True,
        }
