import numpy as np
import pytest

pytest.importorskip("gstools")
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec, GSToolsReferenceFFTSampler


@pytest.mark.parametrize("family", ["gaussian", "matern32_radial", "exponential_radial"])
def test_radial_covariance_matches_gstools_and_native_crop(family):
    spec = GSToolsReferenceSpec(family, 2.5, 1.8)
    sampler = GSToolsReferenceFFTSampler(spec, (12, 10), cell_size_m=1., workers=1)
    assert sampler.metadata["requested_lag_max_absolute_error"] < 1.e-11
    assert max(sampler.metadata["native_increment_relative_representation_errors"].values()) < 1.e-8
    row, col = np.array([0., 1., 3.]), np.array([0., 2., 4.])
    np.testing.assert_allclose(spec.correlation(row, col), spec.gstools_model().cov_spatial([col, row]), atol=1.e-14)
    if family == "exponential_radial":
        assert not np.isclose(spec.correlation(2.5, 1.8), np.exp(-2.))


def test_spectral_sampler_distribution_and_seed_reproducibility():
    sampler = GSToolsReferenceFFTSampler(GSToolsReferenceSpec("exponential_radial", 2.5, 1.8),
                                        (12, 10), cell_size_m=1., workers=1)
    np.testing.assert_array_equal(sampler.sample_residual(9151), sampler.sample_residual(9151))
    assert not np.array_equal(sampler.sample_residual(9151), sampler.sample_residual(9152))
    probes = np.array([[0, 0], [0, 1], [1, 0], [11, 9]])
    values = np.array([sampler.sample_residual(seed)[tuple(probes.T)] for seed in range(1600)])
    offsets = (probes[:, None] - probes[None, :])
    expected = .05**2 * sampler.spec.correlation(offsets[..., 0], offsets[..., 1])
    np.testing.assert_allclose(values.mean(axis=0), 0, atol=.004)
    np.testing.assert_allclose(np.cov(values, rowvar=False), expected, atol=.00023)


def test_reference_composition_and_validation():
    reference = np.full((12, 10), 2.e-10)
    sampler = GSToolsReferenceFFTSampler(GSToolsReferenceSpec("gaussian", 2.5, 1.8),
                                        reference.shape, cell_size_m=1., workers=1)
    residual = sampler.sample_residual(8)
    np.testing.assert_allclose(sampler.sample_permeability(reference, 8), reference * 10**residual, rtol=1.e-7)
    mean_sampler = GSToolsReferenceFFTSampler(GSToolsReferenceSpec("gaussian", 2.5, 1.8, center="arithmetic-mean"),
                                             reference.shape, cell_size_m=1., workers=1)
    ratio = mean_sampler.sample_permeability(reference, 8) / sampler.sample_permeability(reference, 8)
    np.testing.assert_allclose(ratio, np.exp(-.5 * np.log(10.)**2 * .05**2 * sampler.represented_unit_variance), rtol=2.e-7)
    with pytest.raises(ValueError):
        sampler.sample_permeability(-reference, 8)
    with pytest.raises(ValueError):
        sampler.sample_residual(-1)
    with pytest.raises(ValueError):
        GSToolsReferenceSpec("exponential", 2., 3.)


def test_non_psd_embedding_is_not_silently_used():
    with pytest.raises(ValueError, match="no PSD embedding"):
        GSToolsReferenceFFTSampler(GSToolsReferenceSpec("exponential_radial", 200., 150.),
                                   (12, 10), cell_size_m=1., workers=1, max_embedding_points=2048)
