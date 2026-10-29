"""Explicit reference scenarios, never inferred probabilities of RUN cutouts.

Conversions use natural logarithms; sigma_R is in log10 permeability units.
Expert scalar marginals refer to one declared reference permeability anchor,
not the histogram of a spatially heterogeneous field.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from itertools import product
import json
import math

import numpy as np
from scipy.special import ndtri_exp

from .reference_field import build_reference_field_maps


def physical_to_lognormal(mean, variance):
    if not (math.isfinite(mean) and mean > 0 and math.isfinite(variance) and variance >= 0):
        raise ValueError("physical mean must be positive and variance non-negative, both finite")
    log_mean = math.log(mean)
    if variance == 0:
        return log_mean, 0.0
    log_ratio = math.log(variance) - 2 * log_mean
    s2 = (log_ratio + math.log1p(math.exp(-log_ratio)) if log_ratio > 0
          else math.log1p(math.exp(log_ratio)))
    sigma = math.exp(log_ratio / 2) if log_ratio < -36 else math.sqrt(s2)
    if sigma == 0:
        raise ValueError("lognormal sigma is below the representable positive range")
    return log_mean - s2 / 2, sigma


def _validate_log_parameters(mu, sigma):
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma >= 0):
        raise ValueError("finite mu and non-negative finite sigma required")


def _positive_exp(log_value, name):
    """Reject unrepresentable positive outputs instead of returning zero/inf."""
    try:
        value = math.exp(log_value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the representable finite range") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} is outside the representable positive finite range")
    return value


def _equal_tail_z(alpha):
    if not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError("alpha must lie strictly between zero and one")
    return -float(ndtri_exp(math.log(alpha) - math.log(2)))


def lognormal_to_physical(mu, sigma):
    _validate_log_parameters(mu, sigma)
    s2 = sigma * sigma
    log_mean = mu + s2 / 2
    mean = _positive_exp(log_mean, "physical mean")
    if sigma == 0:
        return mean, 0.0
    if sigma < 1e-8:
        log_expm1_s2 = 2 * math.log(sigma)
    elif s2 < 50:
        log_expm1_s2 = math.log(math.expm1(s2))
    else:
        log_expm1_s2 = s2 + math.log1p(-math.exp(-s2))
    variance = _positive_exp(2 * log_mean + log_expm1_s2, "physical variance")
    return mean, variance


def bounds_to_lognormal(lb, ub, alpha):
    """Equal-tail interval [Q(alpha/2), Q(1-alpha/2)], never hard support."""
    if not (math.isfinite(lb) and math.isfinite(ub) and 0 < lb < ub
            and math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError("need finite 0 < lb < ub and 0 < alpha < 1")
    z = _equal_tail_z(alpha)
    return (math.log(lb) + math.log(ub)) / 2, (math.log(ub) - math.log(lb)) / (2 * z)


def lognormal_to_bounds(mu, sigma, alpha):
    _validate_log_parameters(mu, sigma)
    z = _equal_tail_z(alpha)
    return (_positive_exp(mu - z * sigma, "lower quantile bound"),
            _positive_exp(mu + z * sigma, "upper quantile bound"))


@dataclass(frozen=True)
class ReferenceScenario:
    reference_run: str
    sigma_R: float | None = None
    covariance_family: str = "matern32"
    ell_x: float = 800.
    ell_y: float = 500.
    n_modes: int | None = None
    energy_threshold: float | None = .95
    center: str = "median"
    marginal: str = "lognormal"
    physical_mean: float | None = None
    physical_variance: float | None = None
    lb: float | None = None
    ub: float | None = None
    alpha: float | None = None
    marginal_reference_k: float | None = None

    def __post_init__(self):
        if not self.reference_run or self.marginal != "lognormal":
            raise ValueError("reference_run and lognormal marginal required")
        if self.center not in {"median", "arithmetic-mean"}:
            raise ValueError("unknown centering convention")
        if self.covariance_family not in {"gaussian", "matern32", "exponential"}:
            raise ValueError("unsupported separable covariance family")
        if any(not math.isfinite(v) or v <= 0 for v in (self.ell_x, self.ell_y)):
            raise ValueError("correlation lengths must be finite and positive")
        if (self.n_modes is None) == (self.energy_threshold is None):
            raise ValueError("choose exactly one of n_modes or energy_threshold")
        if self.n_modes is not None and (isinstance(self.n_modes, bool)
                or not isinstance(self.n_modes, int) or self.n_modes <= 0):
            raise ValueError("n_modes must be a positive integer")
        if self.energy_threshold is not None and not (math.isfinite(self.energy_threshold)
                and 0 < self.energy_threshold <= 1):
            raise ValueError("energy_threshold must be in (0,1]")
        moments = (self.physical_mean, self.physical_variance)
        bounds = (self.lb, self.ub, self.alpha)
        has_moments, has_bounds = any(v is not None for v in moments), any(v is not None for v in bounds)
        if has_moments and has_bounds:
            raise ValueError("choose expert moments OR quantile bounds")
        if (has_moments and any(v is None for v in moments)
                or has_bounds and any(v is None for v in bounds)):
            raise ValueError("incomplete marginal specification")
        if has_moments or has_bounds:
            mu, sigma = (physical_to_lognormal(*moments) if has_moments
                         else bounds_to_lognormal(*bounds))
            anchor = self.marginal_reference_k
            if anchor is None or not math.isfinite(anchor) or anchor <= 0:
                raise ValueError("expert marginals require positive marginal_reference_k in m2")
            center_value = math.exp(mu + (sigma**2 / 2 if self.center == "arithmetic-mean" else 0))
            if not math.isclose(center_value, anchor, rel_tol=1e-8):
                raise ValueError("expert marginal conflicts with reference anchor and center")
            derived = sigma / math.log(10)
            if self.sigma_R is not None and not math.isclose(self.sigma_R, derived, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError("sigma_R conflicts with expert marginal dispersion")
            if self.sigma_R is None:
                object.__setattr__(self, "sigma_R", derived)
        elif self.marginal_reference_k is not None:
            raise ValueError("marginal_reference_k requires expert moments or bounds")
        if self.sigma_R is None or not math.isfinite(self.sigma_R) or self.sigma_R <= 0:
            raise ValueError("positive finite sigma_R required (log10 units)")

    @property
    def ell_row_m(self) -> float:
        """Correlation length along array rows (legacy field ``ell_y``)."""
        return float(self.ell_y)

    @property
    def ell_col_m(self) -> float:
        """Correlation length along array columns (legacy field ``ell_x``)."""
        return float(self.ell_x)

    @property
    def scenario_id(self):
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "ref-" + sha256(canonical.encode()).hexdigest()[:20]

    def manifest(self):
        result = {"scenario_id": self.scenario_id, "config": asdict(self),
                  "input_law": "reference-centered-lognormal",
                  "sigma_R_units": "log10_permeability", "coordinate_distribution": "iid_standard_normal",
                  "correlation_lengths_m": {
                      "row": self.ell_row_m,
                      "column": self.ell_col_m,
                      "legacy_aliases": {"ell_y": "row", "ell_x": "column"},
                      "axis_interpretation": "array axes; geographic x/y are not assumed",
                  },
                  "assumption_status": "model-form/sensitivity; not statistically identified",
                  "bounds_semantics": "equal-tail untruncated pointwise prior at reference anchor",
                  "truncation_semantics": "representation sensitivity; actual retained variance varies spatially",
                  "RUN_policy": "separate reference scenarios, not iid geological draws"}
        if self.physical_mean is not None or self.lb is not None:
            mu, sigma = (physical_to_lognormal(self.physical_mean, self.physical_variance)
                         if self.physical_mean is not None else bounds_to_lognormal(self.lb, self.ub, self.alpha))
            result["anchor_log_parameters"] = {"mu_ln": mu, "sigma_ln": sigma,
                                               "mu_log10": mu / math.log(10), "sigma_log10": sigma / math.log(10)}
        return result

    def build_maps(self, reference, *, cell_size_m, **conditioning):
        if self.n_modes is not None and self.n_modes > np.asarray(reference).size:
            raise ValueError("n_modes exceeds reference grid dimension")
        maps = build_reference_field_maps(
            reference, cell_size_m=cell_size_m, residual_std_log10_k=self.sigma_R,
            length_scale_m=(self.ell_row_m, self.ell_col_m), covariance_model=self.covariance_family,
            n_modes=self.n_modes, energy_threshold=self.energy_threshold or 1., center=self.center,
            **conditioning)
        for field_map in maps:
            field_map.scenario = self.manifest()
        return maps


def reference_scenario_from_config(config: dict[str, object]) -> ReferenceScenario:
    """Accept new row/column names while preserving legacy ell_y/ell_x artifacts."""
    if not isinstance(config, dict):
        raise ValueError("scenario config must be a mapping")
    params = dict(config)
    row = params.pop("ell_row_m", None)
    col = params.pop("ell_col_m", None)
    if row is not None:
        if "ell_y" in params and not math.isclose(float(params["ell_y"]), float(row)):
            raise ValueError("ell_row_m conflicts with legacy ell_y")
        params["ell_y"] = float(row)
    if col is not None:
        if "ell_x" in params and not math.isclose(float(params["ell_x"]), float(col)):
            raise ValueError("ell_col_m conflicts with legacy ell_x")
        params["ell_x"] = float(col)
    return ReferenceScenario(**params)


def correlation_range_to_length_scale(correlation_range_m, covariance_family):
    """Convert an axial rho=1/e distance to the implemented kernel length.

    The separable Gaussian kernel is exp(-h**2/(2*ell**2)); exponential is
    exp(-abs(h)/ell); Matern-3/2 is (1+sqrt(3)*abs(h)/ell)*exp(-sqrt(3)*abs(h)/ell).
    Ranges and lengths are model-form assumptions, not fitted quantities.
    """
    if (isinstance(correlation_range_m, bool)
            or not isinstance(correlation_range_m, (int, float))
            or not math.isfinite(correlation_range_m) or correlation_range_m <= 0):
        raise ValueError("effective correlation range must be positive and finite")
    if covariance_family == "gaussian":
        factor = math.sqrt(2.)
    elif covariance_family == "exponential":
        factor = 1.
    elif covariance_family == "matern32":
        # With u=sqrt(3)*h/ell, solve u-log1p(u)=1. A fixed bisection
        # gives deterministic conversion without an optimizer tolerance.
        lo, hi = 0., 4.
        for _ in range(64):
            mid = (lo + hi) / 2
            if mid - math.log1p(mid) < 1.:
                lo = mid
            else:
                hi = mid
        factor = ((lo + hi) / 2) / math.sqrt(3.)
    else:
        raise ValueError("unsupported separable covariance family")
    return float(correlation_range_m) / factor


def _matrix_parameters(params):
    """Resolve existing axis aliases and one explicit truncation choice."""
    params = dict(params)
    for alias, legacy in (("ell_row_m", "ell_y"), ("ell_col_m", "ell_x")):
        if alias in params:
            if legacy in params and not math.isclose(float(params[legacy]), float(params[alias])):
                raise ValueError(f"{alias} conflicts with {legacy} in matrix")
            params[legacy] = params.pop(alias)
    if "truncation" in params:
        truncation = params.pop("truncation")
        if not isinstance(truncation, dict) or set(truncation) not in ({"n_modes"}, {"energy_threshold"}):
            raise ValueError("truncation axis entries require exactly one truncation choice")
        params.update(n_modes=None, energy_threshold=None)
        params.update(truncation)
    return params


def _parameter_set_fields(entry):
    """Validate a coupled parameter entry and identify its physical axes."""
    allowed = {"label", "sigma_R", "ell_x", "ell_y", "ell_row_m", "ell_col_m",
               "correlation_range_row_m", "correlation_range_col_m", "truncation"}
    if not isinstance(entry, dict) or not set(entry) <= allowed:
        raise ValueError("parameter_sets entries must be mappings with supported parameters")
    if not isinstance(entry.get("label"), str) or not entry["label"].strip():
        raise ValueError("each parameter set requires a non-empty label")
    if len(entry) == 1:
        raise ValueError("parameter sets must declare at least one parameter")
    physical_axes = set()
    for key in entry:
        axis = {"ell_row_m": "ell_y", "correlation_range_row_m": "ell_y",
                "ell_col_m": "ell_x", "correlation_range_col_m": "ell_x"}.get(key, key)
        physical_axes.add(axis)
    for direction, legacy in (("row", "ell_y"), ("col", "ell_x")):
        if (f"correlation_range_{direction}_m" in entry
                and (legacy in entry or f"ell_{direction}_m" in entry)):
            raise ValueError("parameter set cannot declare both correlation range and kernel length for one axis")
    return physical_axes - {"label"}


def scenario_matrix_with_metadata(base, axes):
    """Expand independent axes and coupled parameter sets, retaining label provenance.

    Labels are outside the scientific scenario configuration, so changing a
    display label cannot alter a scenario ID. Independent axes remain Cartesian;
    every mapping in ``parameter_sets`` is one complete, coupled setting.
    """
    allowed = {"reference_run", "sigma_R", "covariance_family", "ell_x", "ell_y",
               "ell_row_m", "ell_col_m", "truncation", "parameter_sets"}
    if (not isinstance(axes, dict) or not set(axes) <= allowed
            or any(not isinstance(values, (list, tuple)) or not values for values in axes.values())):
        raise ValueError("unknown or empty scenario matrix axis")
    parameter_sets = axes.get("parameter_sets", [])
    labels = []
    independent = {key: value for key, value in axes.items() if key != "parameter_sets"}
    independent_fields = {"ell_row_m": "ell_y", "ell_col_m": "ell_x"}
    independent_fields = {independent_fields.get(key, key) for key in independent}
    for entry in parameter_sets:
        overlap = _parameter_set_fields(entry) & independent_fields
        if overlap:
            raise ValueError(f"parameter_sets conflict with independent matrix axes: {sorted(overlap)}")
        labels.append(entry["label"])
    if len(set(labels)) != len(labels):
        raise ValueError("parameter_sets labels must be unique")
    scenarios, metadata = [], {}
    for values in product(*axes.values()):
        params = dict(zip(axes, values))
        entry = params.pop("parameter_sets", None)
        if entry is not None:
            coupled = {key: value for key, value in entry.items() if key != "label"}
            family = params.get("covariance_family", base.covariance_family)
            for direction in ("row", "col"):
                range_key = f"correlation_range_{direction}_m"
                if range_key in coupled:
                    coupled[f"ell_{direction}_m"] = correlation_range_to_length_scale(
                        coupled.pop(range_key), family)
            params.update(coupled)
        scenario = replace(base, **_matrix_parameters(params))
        scenarios.append(scenario)
        if entry is not None:
            metadata[scenario.scenario_id] = {
                "label": entry["label"],
                "declared_parameters": {key: value for key, value in entry.items() if key != "label"},
                "resolved_parameters": {
                    "sigma_R": scenario.sigma_R, "covariance_family": scenario.covariance_family,
                    "ell_row_m": scenario.ell_row_m, "ell_col_m": scenario.ell_col_m,
                    "n_modes": scenario.n_modes, "energy_threshold": scenario.energy_threshold,
                },
                "assumed_effective_range_m": {
                    direction: entry[f"correlation_range_{direction}_m"]
                    for direction in ("row", "col") if f"correlation_range_{direction}_m" in entry
                },
                "effective_range_definition": "axial distance where prior kernel correlation rho = exp(-1)",
                "assumption_status": "explicit coupled model-form/sensitivity setting; not calibrated",
            }
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("duplicate scenarios in matrix")
    return scenarios, metadata


def scenario_matrix(base, axes):
    """Expand explicit axes and coupled settings; no weights or pooling by default."""
    return scenario_matrix_with_metadata(base, axes)[0]


def scenario_weights(ids, weights=None):
    """No implicit mixture. Explicit equal/custom weights are assumptions."""
    if weights is None:
        return None
    if weights == "equal":
        return {key: 1 / len(ids) for key in ids}
    if not isinstance(weights, dict) or set(weights) != set(ids):
        raise ValueError("custom scenario weights must cover exactly the scenario IDs")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError("scenario weights must be non-negative, finite and sum to one")
    return dict(weights)
