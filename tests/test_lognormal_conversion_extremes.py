"""Conversions preserve valid extreme laws without overflowing intermediates."""
import math

import numpy as np
import pytest
import yaml

from subsurface_uq.sampling.scenarios import (
    ReferenceScenario,
    bounds_to_lognormal,
    lognormal_to_bounds,
    lognormal_to_physical,
    physical_to_lognormal,
    scenario_matrix,
    scenario_weights,
)
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.reference_field import save_reference_field_input_model


@pytest.mark.parametrize("mean,variance", [
    (1e-200, 1e-200),  # Squared mean underflows, but both moments are finite.
    (1e200, 1e200),    # Squared mean overflows, but relative variance is positive.
    (1e200, 1e-200),   # Log variance underflows, while log std is representable.
])
def test_extreme_physical_moments_roundtrip(mean, variance):
    mu, sigma = physical_to_lognormal(mean, variance)
    assert math.isfinite(mu) and math.isfinite(sigma) and sigma > 0
    restored = lognormal_to_physical(mu, sigma)
    np.testing.assert_allclose(restored, [mean, variance], rtol=2e-12, atol=0)


def test_finite_physical_variance_despite_overflowing_expm1_factor():
    mean, variance = lognormal_to_physical(-550.0, 30.0)
    assert mean == pytest.approx(math.exp(-100.0), rel=1e-14, abs=0)
    assert variance == pytest.approx(math.exp(700.0), rel=1e-14, abs=0)


def test_bounds_do_not_require_representable_physical_moments():
    with pytest.raises(ValueError, match="physical"):
        lognormal_to_physical(0.0, 30.0)
    lb, ub = lognormal_to_bounds(0.0, 30.0, .05)
    assert 0 < lb < ub < float("inf")
    np.testing.assert_allclose(bounds_to_lognormal(lb, ub, .05), [0.0, 30.0],
                               rtol=1e-12, atol=1e-12)


def test_smallest_positive_alpha_has_representable_equal_tail_bounds():
    alpha = np.nextafter(0.0, 1.0)
    lb, ub = lognormal_to_bounds(0.0, 1.0, alpha)
    assert 0 < lb < 1 < ub < float("inf")
    np.testing.assert_allclose(bounds_to_lognormal(lb, ub, alpha), [0.0, 1.0],
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mu,sigma", [(1000.0, 0.0), (-1000.0, 0.0),
                                      (0.0, 30.0), (0.0, 1e308)])
def test_unrepresentable_physical_moments_raise_value_error(mu, sigma):
    with pytest.raises(ValueError, match="representable"):
        lognormal_to_physical(mu, sigma)


@pytest.mark.parametrize("mu,sigma", [(0.0, 1000.0), (1000.0, 0.0),
                                      (-1000.0, 0.0)])
def test_unrepresentable_bounds_raise_value_error(mu, sigma):
    with pytest.raises(ValueError, match="representable"):
        lognormal_to_bounds(mu, sigma, .05)


def test_positive_variance_is_not_silently_converted_to_zero_sigma():
    with pytest.raises(ValueError, match="sigma.*representable"):
        physical_to_lognormal(1e300, 1e-300)


def test_zero_variance_remains_a_valid_conversion():
    mu, sigma = physical_to_lognormal(1e-200, 0.0)
    assert sigma == 0
    mean, variance = lognormal_to_physical(mu, sigma)
    assert mean == pytest.approx(1e-200, rel=1e-12, abs=0)
    assert variance == 0


def test_legacy_expert_scenario_preserves_serialized_amplitude_and_id(tmp_path):
    mean, variance = 2e-10, 3e-21
    # Conversion used before the log-domain implementation. Its last few bits
    # differ from the stable conversion, but it is scientifically consistent.
    old_s2 = math.log1p(variance / mean**2)
    old_sigma_R = math.sqrt(old_s2) / math.log(10)
    anchor = math.exp(math.log(mean) - old_s2 / 2)
    scenario = ReferenceScenario("RUN_1", physical_mean=mean, physical_variance=variance,
                                 marginal_reference_k=anchor, sigma_R=old_sigma_R,
                                 ell_x=5.0, ell_y=5.0, n_modes=2, energy_threshold=None)
    assert scenario.sigma_R == old_sigma_R
    field_map = scenario.build_maps(np.full((3, 4), anchor), cell_size_m=5.0)[0]
    path = tmp_path / "legacy_expert.yaml"
    payload = save_reference_field_input_model(path, field_map)
    payload["input_law"] = "reference-centered-lognormal-candidate"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_conditional_kl_input_model(path)
    assert loaded.payload["scenario"]["scenario_id"] == scenario.scenario_id
    assert loaded.unconditional.scenario["scenario_id"] == scenario.scenario_id
    assert loaded.prior.std_log10_k == old_sigma_R


def test_numeric_formatting_does_not_admit_duplicate_matrix_scenarios():
    base = ReferenceScenario("RUN_1", sigma_R=.1)
    with pytest.raises(ValueError, match="duplicate"):
        scenario_matrix(base, {"ell_x": [500, 500.0]})


def test_custom_scenario_weight_sum_uses_absolute_tolerance():
    with pytest.raises(ValueError, match="sum to one"):
        scenario_weights(["first", "second"], {"first": .5, "second": .5 + 5e-10})
