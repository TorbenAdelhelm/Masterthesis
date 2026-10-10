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
