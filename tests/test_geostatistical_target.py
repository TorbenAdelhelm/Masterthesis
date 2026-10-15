import json
import numpy as np
import pytest

from subsurface_uq.sampling.geostatistical_target import (
    Base10LognormalTarget,
    HydraulicConductivityValidationAccumulator,
    validate_hydraulic_conductivity_marginal,
)
from subsurface_uq.sampling.hydraulic_conductivity import (
    FluidProperties,
    HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
    hydraulic_conductivity_to_permeability,
    hydraulic_conversion_metadata,
    permeability_to_hydraulic_conductivity,
)


def test_hydraulic_conductivity_permeability_roundtrip():
    kh = np.array([1e-4, 1e-3, 5e-2])
    fluid = FluidProperties(998.2, 1.002e-3, 9.80665)
    k = hydraulic_conductivity_to_permeability(kh, fluid=fluid)
    np.testing.assert_allclose(permeability_to_hydraulic_conductivity(k, fluid=fluid), kh, rtol=1e-14)
    historical = hydraulic_conductivity_to_permeability(kh, convention="historical-training")
    np.testing.assert_allclose(
        historical, kh / HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR
    )
    np.testing.assert_allclose(
        permeability_to_hydraulic_conductivity(historical, convention="historical-training"), kh
    )
    assert hydraulic_conversion_metadata("historical-training")["divisor_s_inv"] == 7.5e6
    with pytest.raises(ValueError):
        permeability_to_hydraulic_conductivity(historical, convention="unknown")


def test_base10_lognormal_target_semantics_and_no_clipping():
    target = Base10LognormalTarget(-3.0, 0.5, (1e-4, 5e-2), 0.95)
    assert target.log10_variance == pytest.approx(0.25)
    assert 0.0 < target.theoretical_interval_fraction < 1.0
    rng = np.random.default_rng(9)
    values = 10.0 ** rng.normal(-3.0, 0.5, 200_000)
    result = validate_hydraulic_conductivity_marginal(values, target)
    assert result["log10_mean"] == pytest.approx(-3.0, abs=0.01)
    assert result["log10_std"] == pytest.approx(0.5, abs=0.01)
    assert result["no_filtering"] is True
    assert result["lower_tail_fraction"] > 0.0
    assert result["upper_tail_fraction"] > 0.0
    assert "qq_rmse_standardized" in result["normality_deviation"]
    expected = target.theoretical_interval_fraction
    assert result["fraction_inside_interval"] == pytest.approx(expected, abs=0.01)


def test_target_mapping_disambiguates_std_and_variance():
    default = Base10LognormalTarget.from_mapping({})
    assert default.log10_std == pytest.approx(0.5)
    assert default.log10_variance == pytest.approx(0.25)
    variance_target = Base10LognormalTarget.from_mapping({"log10_variance": 0.5})
    assert variance_target.log10_std == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError, match="OR"):
        Base10LognormalTarget.from_mapping({"log10_std": 0.5, "log10_variance": 0.25})


def test_streaming_validation_preserves_full_field_moments_without_filtering():
    rng = np.random.default_rng(17)
    target = Base10LognormalTarget()
    fields = 10.0 ** rng.normal(-3.0, 0.5, size=(4, 60, 70))
    accumulator = HydraulicConductivityValidationAccumulator(
        target,
        tolerances={"max_abs_log10_std_error": 0.05, "max_qq_rmse_standardized": 0.15},
        max_pooled_sample_values=5000,
    )
    for field in fields:
        accumulator.update(field)
    result = accumulator.finalize()
    expected_logs = np.log10(fields).reshape(-1)
    assert result["pooled"]["log10_mean"] == pytest.approx(expected_logs.mean())
    assert result["pooled"]["log10_std"] == pytest.approx(expected_logs.std(ddof=1))
    assert result["field_count"] == 4
    assert result["no_filtering"] is True
    assert result["pooled"]["heuristic_assessment"]["status"] in {
        "heuristically_compatible", "heuristic_deviation"
    }


@pytest.mark.parametrize("tolerances", [
    {"max_abs_log10_mean_error": .25},
    {"max_abs_log10_std_error": .15},
    {"max_qq_rmse_standardized": .5},
    {"max_abs_log10_mean_error": .25, "max_abs_log10_std_error": None},
    {},
])
def test_partial_thresholds_serialize_as_finite_limits_or_null(tolerances):
    values = np.logspace(-4, -2, 50)
    result = validate_hydraulic_conductivity_marginal(values, Base10LognormalTarget(), tolerances=tolerances)
    json.dumps(result, allow_nan=False)
    limits = result["heuristic_assessment"]["tolerances"]
    assert all(value is None or np.isfinite(value) for value in limits.values())
    for key in set(limits) - set(tolerances):
        assert limits[key] is None


@pytest.mark.parametrize("limit", [float("inf"), float("nan"), -.1])
def test_invalid_thresholds_fail_before_accumulation(limit):
    with pytest.raises(ValueError, match="finite and non-negative"):
        HydraulicConductivityValidationAccumulator(Base10LognormalTarget(),
            tolerances={"max_abs_log10_mean_error": limit})


def test_descriptive_marginals_do_not_imply_a_global_target():
    values = np.logspace(-5, -2, 100)
    result = validate_hydraulic_conductivity_marginal(values)
    assert result["target"] is None and result["normality_deviation"] is None
    assert result["log10_mean_error"] is None and result["fraction_inside_interval"] is None
    assert result["heuristic_assessment"]["status"] == "not_assessed"
    assert result["log10_std"] == pytest.approx(np.std(np.log10(values), ddof=1))
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="explicit distribution target"):
        HydraulicConductivityValidationAccumulator(tolerances={"max_abs_log10_mean_error": .25})


def _pooled(fields, *, capacity=800):
    accumulator = HydraulicConductivityValidationAccumulator(max_pooled_sample_values=capacity, sample_seed=7)
    for field in fields:
        accumulator.update(field)
    return accumulator.finalize()["pooled"]


def test_pooled_quantiles_do_not_systematically_favor_late_fields():
    fields = [10 ** np.linspace(c - .01, c + .01, 100).reshape(10, 10)
              for c in np.linspace(-4, -2, 32)]
    exact = np.quantile(np.asarray(fields), [.05, .5, .95])
    for ordered in (fields, fields[::-1]):
        result = _pooled(ordered)
        observed = result["hydraulic_conductivity_quantiles_m_s"]
        # Sampling uncertainty is allowed; the old repeated thinning gave
        # median errors of factors 4.7 and 0.21 on this same population.
        np.testing.assert_allclose([observed["q05"], observed["q50"], observed["q95"]], exact, rtol=.35)
        assert result["n_values"] == 3200
        assert result["log10_mean"] == pytest.approx(-3.)
        assert result["quantile_normality_sample_size"] == 800
        assert result["quantile_normality_sampling"]["method"] == "uniform_pixels_with_replacement"
    assert _pooled(fields) == _pooled(fields)


def test_pooled_sampling_weights_pixels_not_field_batches():
    small = np.full((10, 10), 1e-4)
    large = np.full((99, 100), 1e-2)
    for fields in ([small, large], [large, small]):
        result = _pooled(fields, capacity=4000)
        assert result["n_values"] == 10000
        assert result["log10_mean"] == pytest.approx(-2.02)
        assert result["hydraulic_conductivity_quantiles_m_s"]["q05"] == pytest.approx(1e-2)


def test_small_pooled_population_retains_exact_quantiles():
    values = np.logspace(-5, -2, 64).reshape(8, 8)
    result = _pooled([values], capacity=100)
    assert result["hydraulic_conductivity_quantiles_m_s"]["q50"] == pytest.approx(np.median(values))
    assert result["quantile_normality_sampling"]["method"] == "all_values"
