"""Descriptive reference fidelity and checks against the represented residual law.

Spatial summaries use pixels as a description of one heterogeneous field.
Envelopes summarize variation across generated fields, never confidence from
treating their correlated pixels (or Sobol vectors) as independent observations.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from .reference_field import ReferenceFieldLogGaussianPermeabilityMap
from .spatial_diagnostics import compare_reference_ensemble, log_field_spectrum
from .training_compatibility import permeability_field_features
from ..visualization.reference_fields import reference_field_agreement


_PROBABILITIES = np.asarray([.01, .05, .25, .5, .75, .95, .99])
_ENVELOPE_PROBABILITIES = [.05, .5, .95]
_DIRECTIONS = ("row", "column", "diagonal_positive", "diagonal_negative")


def _safe(value):
    """Represent undefined correlations/insufficient sample moments as JSON null."""
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _envelope(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("envelopes require a nonempty field-by-descriptor array")
    quantiles = np.full((3, values.shape[1]), np.nan)
    for index, column in enumerate(values.T):
        finite = column[np.isfinite(column)]
        if finite.size:
            quantiles[:, index] = np.quantile(finite, _ENVELOPE_PROBABILITIES)
    return {"probabilities": _ENVELOPE_PROBABILITIES, "values": quantiles,
            "defined_field_count": np.isfinite(values).sum(axis=0)}


def _paired_values(logs, direction, lag):
    if direction == "row":
        return logs[lag:, :], logs[:-lag, :]
    if direction == "column":
        return logs[:, lag:], logs[:, :-lag]
    if direction == "diagonal_positive":
        return logs[lag:, lag:], logs[:-lag, :-lag]
    return logs[lag:, :-lag], logs[:-lag, lag:]


def _directional_curves(logs, direction, lags, *, correlation=False):
    result = []
    for lag in lags:
        left, right = _paired_values(logs, direction, int(lag))
        if not correlation:
            result.append(float(.5 * np.mean((left - right) ** 2)))
            continue
        constant = any(np.ptp(value) <= 16 * np.finfo(float).eps
                       * max(1., float(np.max(np.abs(value)))) for value in (left, right))
        left, right = left - left.mean(), right - right.mean()
        denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
        result.append(float(np.clip(np.sum(left * right) / denominator, -1., 1.))
                      if denominator > 0 and not constant else np.nan)
    return np.asarray(result)


def _curve_payload(reference, generated, *, cell_size_m, lags, correlation=False):
    payload = {}
    for direction in _DIRECTIONS:
        curves = np.asarray([_directional_curves(field, direction, lags,
                                                correlation=correlation)
                             for field in generated])
        scale = np.sqrt(2.) if direction.startswith("diagonal") else 1.
        payload[direction] = {
            "distance_m": lags * cell_size_m * scale,
            "reference": (_directional_curves(reference, direction, lags,
                                              correlation=correlation)
                          if reference is not None else None),
            "generated_per_field": curves,
            "generated_envelope": _envelope(curves),
        }
    return payload


def reference_ensemble_coherence(reference, generated, *, cell_size_m, max_lag_cells=24):
    """Compare one fixed reference with its full, unfiltered matched-grid ensemble.

    Histogram/QQ distributions pool heterogeneous locations for description only.
    Correlation curves use Pearson correlation of the two lagged spatial arrays,
    distinguishing normalized spatial structure from variogram amplitude.
    """
    if any(np.iscomplexobj(value) for value in (reference, generated)):
        raise ValueError("reference and generated permeability must be real-valued")
    reference, generated = (np.asarray(value, dtype=float) for value in (reference, generated))
    if (reference.ndim != 2 or generated.ndim != 3 or not len(generated)
            or tuple(generated.shape[1:]) != reference.shape or min(reference.shape) < 3
            or any(not np.all(np.isfinite(value)) or np.any(value <= 0)
                   for value in (reference, generated))):
        raise ValueError("one positive finite 2-D reference and matching nonempty [N,H,W] fields required (H,W>=3)")
    if (not np.isfinite(cell_size_m) or cell_size_m <= 0
            or isinstance(max_lag_cells, bool) or int(max_lag_cells) != max_lag_cells
            or max_lag_cells < 1):
        raise ValueError("positive cell size and positive integer max_lag_cells required")
    fidelity = compare_reference_ensemble(reference, generated, cell_size_m=cell_size_m,
                                          max_lag_cells=int(max_lag_cells))
    ref_log, logs = np.log10(reference), np.log10(generated)
    residual = logs - ref_log[None]
    low, high = min(ref_log.min(), logs.min()), max(ref_log.max(), logs.max())
    if low == high:
        low, high = low - .05, high + .05
    edges = np.linspace(low, high, 61)
    ref_density = np.histogram(ref_log, bins=edges, density=True)[0]
    density_rows = np.asarray([np.histogram(field, bins=edges, density=True)[0]
                               for field in logs])
    cdf_coordinates = edges
    reference_cdf = np.searchsorted(np.sort(ref_log.ravel()), cdf_coordinates, side="right") / ref_log.size
    generated_cdf = np.searchsorted(np.sort(logs.ravel()), cdf_coordinates, side="right") / logs.size
    per_field = []
    for index, field in enumerate(generated):
        per_field.append({"sample_index": index,
                          **reference_field_agreement(reference, field),
                          "field_features": permeability_field_features(field, cell_size_m=cell_size_m)})
    lags = np.arange(1, min(int(max_lag_cells), min(reference.shape) - 1) + 1)
    reference_spectrum = fidelity["spectra"]["reference"]
    spectrum_rows = [log_field_spectrum(field, cell_size_m=cell_size_m) for field in generated]
    spectra = {}
    for name, coordinate_key in (("radial", "radial_edges_per_m"), ("angular", "angular_edges_rad")):
        curves = np.asarray([item[f"{name}_power"] for item in spectrum_rows])
        spectra[name] = {"coordinate_edges": reference_spectrum[coordinate_key],
                         "reference": reference_spectrum[f"{name}_power"],
                         "generated_per_field": curves,
                         "generated_envelope": _envelope(curves)}
    return _safe({
        "schema_version": 1,
        "assessment": "descriptive_reference_coherence",
        "reference_fidelity": fidelity,
        "reference_field_count": 1, "generated_field_count": len(generated),
        "diagnostic_shape": list(reference.shape), "cell_size_m": float(cell_size_m),
        "distribution": {
            "space": "log10_intrinsic_permeability_m2", "bin_edges_log10": edges,
            "reference_density": ref_density, "generated_density": density_rows.mean(axis=0),
            "generated_density_per_field": density_rows,
            "generated_density_envelope": _envelope(density_rows),
            "cdf_log10": cdf_coordinates, "reference_cdf": reference_cdf,
            "generated_cdf": generated_cdf, "probabilities": _PROBABILITIES,
            "reference_quantiles_log10": np.quantile(ref_log, _PROBABILITIES),
            "generated_quantiles_log10": np.quantile(logs, _PROBABILITIES),
            "generated_field_quantiles_log10": np.quantile(logs, _PROBABILITIES, axis=(1, 2)).T,
            "log_ratio_quantiles": np.quantile(residual, _PROBABILITIES, axis=(1, 2)).T,
            "interpretation": "spatial mixtures over heterogeneous locations, not one pointwise lognormal marginal",
        },
        "per_field": per_field,
        "variograms": _curve_payload(ref_log, logs, cell_size_m=cell_size_m, lags=lags),
        "autocorrelation": _curve_payload(ref_log, logs, cell_size_m=cell_size_m, lags=lags, correlation=True),
        "residual_variograms": _curve_payload(None, residual, cell_size_m=cell_size_m, lags=lags),
        "residual_autocorrelation": _curve_payload(None, residual, cell_size_m=cell_size_m, lags=lags, correlation=True),
        "spectra": spectra,
        "residual_definition": "log10(K / K_ref); not automatically zero-mean under arithmetic-mean centering or conditioning",
        "ensemble_envelope": {
            "probabilities": _ENVELOPE_PROBABILITIES,
            "status": "empirical_realization_spread" if len(generated) > 1 else "single_field_no_ensemble_spread",
            "interpretation": "empirical quantiles across fields; not a confidence interval; correlated pixels do not increase the ensemble count; Sobol vectors may be dependent",
        },
        "decision_rule": None, "sample_filtering": None,
        "training_support": "unassessed_by_reference_comparison",
        "interpretation": "Reference fidelity and residual texture describe consequences of the declared uncertainty scenario. Exact equality to the reference is not expected. They neither calibrate residual covariance nor certify LGCNN prediction accuracy.",
    })


def _correlation(covariance, *, variance_zero_tolerance=0.):
    variance = np.maximum(np.diag(covariance), 0.)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    active = variance > variance_zero_tolerance
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan),
                     where=(denominator > 0) & active[:, None] & active[None, :])


def _relative_error(empirical, expected):
    scale = float(np.linalg.norm(expected))
    return float(np.linalg.norm(empirical - expected) / scale) if scale > 0 else None


def _correlation_error(empirical_covariance, expected_correlation, *, variance_zero_tolerance=0.):
    empirical_correlation = _correlation(empirical_covariance, variance_zero_tolerance=variance_zero_tolerance)
    finite = np.isfinite(expected_correlation) & np.isfinite(empirical_correlation)
    return (float(np.sqrt(np.mean((expected_correlation[finite] - empirical_correlation[finite])**2)))
            if np.any(finite) else None)


def _iid_covariance_relative_rmse(covariance, count):
    """Known-covariance Gaussian iid expectation for unbiased sample covariance."""
    norm_squared = float(np.sum(covariance**2))
    return (float(np.sqrt((float(np.trace(covariance))**2 + norm_squared)
                          / ((count - 1) * norm_squared)))
            if count > 1 and norm_squared > 0 else None)


def _prefix_convergence(samples, log_mean, covariance, *, iid_mc, variance_zero_tolerance):
    count = len(samples)
    counts = [2**power for power in range(1, count.bit_length())]
    if count not in counts:
        counts.append(count)
    expected_correlation = _correlation(covariance, variance_zero_tolerance=variance_zero_tolerance)
    mean_error, covariance_error, correlation_error, covariance_rmse = [], [], [], []
    for size in counts:
        prefix = samples[:size]
        mean_error.append(float(np.sqrt(np.mean((prefix.mean(axis=0) - log_mean)**2))))
        empirical_covariance = np.atleast_2d(np.cov(prefix, rowvar=False, ddof=1)) if size > 1 else None
        covariance_error.append(_relative_error(empirical_covariance, covariance) if size > 1 else None)
        correlation_error.append(_correlation_error(empirical_covariance, expected_correlation,
                                                    variance_zero_tolerance=variance_zero_tolerance)
                                 if size > 1 else None)
        covariance_rmse.append(_iid_covariance_relative_rmse(covariance, size) if iid_mc else None)
    return {
        "sample_counts": counts, "log10_mean_rmse": mean_error,
        "covariance_relative_frobenius_error": covariance_error,
        "correlation_rmse": correlation_error,
        "iid_gaussian_expected_covariance_relative_frobenius_rmse": covariance_rmse,
        "interpretation": "nested prefixes of the same recorded design; descriptive stability, not independent replications or proof of convergence. Gaussian iid covariance RMSE is a known-law finite-sample expectation for MC only, not an acceptance threshold; RQMC needs independent scrambles.",
    }


def reference_probe_law_diagnostics(field_map, log10_samples, indices_yx, *, sampling_method="unspecified"):
    """Check predetermined native-grid probes against their actual retained law.

    ``log10_samples`` has shape [field_count, probe_count]. Probes are selected
    before inspecting fields, share the map's native grid and must be distinct.
    No Gaussian iid test is applied to pooled, spatially correlated pixels.
    """
    from .kriging import (ConditionalKLLogGaussianPermeabilityMap,
                          ContinuousPointConditionalKLLogGaussianPermeabilityMap)

    if not isinstance(field_map, ReferenceFieldLogGaussianPermeabilityMap):
        raise ValueError("reference-centered field map required")
    method = str(sampling_method).upper()
    if method not in {"MC", "RQMC", "UNSPECIFIED"}:
        raise ValueError("sampling_method must be MC, RQMC or unspecified")
    if np.iscomplexobj(log10_samples) or np.iscomplexobj(indices_yx):
        raise ValueError("probe samples and indices must be real-valued")
    indices = np.asarray(indices_yx)
    basis = field_map.prior.mode_matrix_at_indices(indices)
    indices = indices.astype(int)
    if len({tuple(index) for index in indices}) != len(indices):
        raise ValueError("probe cells must be distinct")
    samples = np.asarray(log10_samples, dtype=float)
    if (samples.ndim != 2 or not len(samples) or samples.shape[1] != len(indices)
            or not np.all(np.isfinite(samples))):
        raise ValueError("finite nonempty log10_samples with shape [N,probe_count] required")
    rows, columns = indices.T
    reference = field_map.reference_permeability[rows, columns]
    log_reference = np.log10(reference)
    prior_variance = np.sum(basis**2, axis=1)
    log_mean = log_reference.copy()
    if field_map.center == "arithmetic-mean":
        log_mean -= .5 * np.log(10.) * prior_variance
    gaussian = field_map.gaussian_map
    conditional = gaussian is not field_map.prior
    if not conditional:
        covariance = basis @ basis.T
    elif isinstance(gaussian, ContinuousPointConditionalKLLogGaussianPermeabilityMap):
        # This public map implements a symmetric low-rank square root of the
        # coordinate covariance. Apply it to probe loadings without an m*m matrix.
        coordinate_mean = gaussian.posterior_coordinate_mean
        transformed = gaussian.prior_coordinates(basis) - coordinate_mean
        covariance = transformed @ transformed.T
        log_mean += basis @ coordinate_mean
    elif isinstance(gaussian, ConditionalKLLogGaussianPermeabilityMap):
        covariance = basis @ gaussian.posterior_coordinate_covariance @ basis.T
        log_mean += basis @ gaussian.posterior_coordinate_mean
    else:
        raise ValueError("unsupported conditioned reference map; cannot assume prior covariance")
    covariance = .5 * (covariance + covariance.T)
    variance = np.maximum(np.diag(covariance), 0.)
    count = len(samples)
    empirical_mean = samples.mean(axis=0)
    empirical_covariance = (np.atleast_2d(np.cov(samples, rowvar=False, ddof=1))
                            if count > 1 else np.full_like(covariance, np.nan))
    empirical_variance = np.diag(empirical_covariance)
    zero_tolerance = 64 * np.finfo(float).eps * max(float(np.max(prior_variance)), np.finfo(float).tiny)
    expected_correlation = _correlation(covariance, variance_zero_tolerance=zero_tolerance)
    empirical_correlation = _correlation(empirical_covariance, variance_zero_tolerance=zero_tolerance)
    active = variance > zero_tolerance
    standardized = (samples[:, active] - log_mean[active]) / np.sqrt(variance[active])
    qq = np.quantile(standardized, _PROBABILITIES, axis=0).T if np.any(active) else np.empty((0, len(_PROBABILITIES)))
    normal_quantiles = ndtri(_PROBABILITIES)
    pooled_qq = np.quantile(standardized, _PROBABILITIES) if standardized.size else np.full(len(_PROBABILITIES), np.nan)
    ln10 = np.log(10.)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected_physical = np.exp(ln10 * log_mean + .5 * ln10**2 * variance)
        physical = np.power(10., samples)
        empirical_physical = physical.mean(axis=0)
        expected_ratio = np.exp(ln10 * (log_mean - log_reference) + .5 * ln10**2 * variance)
        empirical_ratio = np.power(10., samples - log_reference).mean(axis=0)
    iid_mc = method == "MC"
    correlation_error = _correlation_error(empirical_covariance, expected_correlation,
                                            variance_zero_tolerance=zero_tolerance)
    return _safe({
        "schema_version": 1, "assessment": "descriptive_retained_law_check",
        "indices_yx": indices, "field_count": count, "probe_count": len(indices),
        "grid_support": "native cells; no geometric-block aggregation",
        "sampling_method": sampling_method, "center": field_map.center,
        "conditioned": conditional, "reference_permeability_m2": reference,
        "nominal_sigma_R_log10": field_map.prior.std_log10_k,
        "prior_retained_variance_log10": prior_variance,
        "standardization_variance_zero_tolerance": zero_tolerance,
        "expected_log10_mean": log_mean, "expected_log10_variance": variance,
        "empirical_log10_mean": empirical_mean, "empirical_log10_variance": empirical_variance,
        "log10_mean_error": empirical_mean - log_mean,
        "log10_variance_ratio": np.divide(empirical_variance, variance,
                                           out=np.full_like(variance, np.nan), where=active),
        "expected_covariance_log10": covariance, "empirical_covariance_log10": empirical_covariance,
        "expected_correlation": expected_correlation, "empirical_correlation": empirical_correlation,
        "covariance_relative_frobenius_error": _relative_error(empirical_covariance, covariance) if count > 1 else None,
        "correlation_rmse": correlation_error,
        "expected_arithmetic_permeability_m2": expected_physical,
        "empirical_arithmetic_permeability_m2": empirical_physical,
        "expected_arithmetic_ratio_to_reference": expected_ratio,
        "empirical_arithmetic_ratio_to_reference": empirical_ratio,
        "standardized_residual_qq": {
            "probe_indices_in_qq": np.flatnonzero(active), "probabilities": _PROBABILITIES,
            "normal_quantiles": normal_quantiles, "pooled_empirical_quantiles": pooled_qq,
            "per_probe_empirical_quantiles": qq,
            "pooled_qq_rmse": float(np.sqrt(np.mean((pooled_qq - normal_quantiles)**2))) if standardized.size else None,
            "interpretation": "effect-size description only; probes share spatial dependence, pooled values are not iid and no normality p-value is computed",
        },
        "mean_sampling_error": {
            "status": "iid_mc_reference_standard_error" if iid_mc else "not_assessed_without_independent_replications",
            "expected_log10_mean_standard_error": np.sqrt(variance / count) if iid_mc else None,
            "iid_gaussian_expected_covariance_relative_frobenius_rmse": _iid_covariance_relative_rmse(covariance, count) if iid_mc else None,
            "interpretation": "MC standard errors use independent field draws, not pixel counts; RQMC integration error requires independent scrambles; no confidence interval or pass threshold is supplied",
        },
        "prefix_convergence": _prefix_convergence(samples, log_mean, covariance, iid_mc=iid_mc,
                                                    variance_zero_tolerance=zero_tolerance),
        "ensemble_status": "finite_ensemble_descriptive_check" if count > 1 else "insufficient_fields_for_empirical_covariance",
        "numeric_null_semantics": "undefined correlation/variance ratio, insufficient samples or a physical expectation outside finite floating-point range",
        "decision_rule": None, "sample_filtering": None,
        "interpretation": "Agreement checks implementation and finite-design behavior relative to the specified retained law. It does not identify the geological residual law or establish LGCNN training compatibility. Conditioning changes center and covariance; arithmetic-mean correction uses prior retained variance.",
    })
