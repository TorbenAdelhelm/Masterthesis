import numpy as np
import pytest

from subsurface_uq.sampling.geostatistical_target import (
    Base10LognormalTarget,
    validate_hydraulic_conductivity_marginal,
)
from subsurface_uq.sampling.hydraulic_conductivity import (
    FluidProperties,
    hydraulic_conductivity_to_permeability,
    permeability_to_hydraulic_conductivity,
)


def test_hydraulic_conductivity_permeability_roundtrip():
    kh = np.array([1e-4, 1e-3, 5e-2])
    fluid = FluidProperties(998.2, 1.002e-3, 9.80665)
    k = hydraulic_conductivity_to_permeability(kh, fluid=fluid)
    np.testing.assert_allclose(permeability_to_hydraulic_conductivity(k, fluid=fluid), kh, rtol=1e-14)


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
    expected = target.theoretical_interval_fraction
    assert result["fraction_inside_interval"] == pytest.approx(expected, abs=0.01)
