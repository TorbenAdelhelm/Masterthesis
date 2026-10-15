import json

import numpy as np
import pytest

from subsurface_uq.sampling.reference_coherence import (
    reference_ensemble_coherence,
    reference_probe_law_diagnostics,
)
from subsurface_uq.sampling.reference_field import (
    ReferenceFieldLogGaussianPermeabilityMap, build_reference_field_maps,
)
from subsurface_uq.sampling.kriging import ConditionalKLLogGaussianPermeabilityMap


def _reference(shape=(12, 16)):
    yy, xx = np.indices(shape)
    return 10. ** (-9.5 + .03 * yy + .1 * np.sin(xx / 3.))


def _map(center="median", **conditioning):
    return build_reference_field_maps(
        _reference((6, 8)), cell_size_m=5., residual_std_log10_k=.15,
        length_scale_m=(13., 20.), n_modes=12, center=center, **conditioning,
    )


def _exact_mean_covariance_design(dimension):
    # Deterministic diagnostic fixture: zero mean, sample covariance exactly I.
    # This fixture is not claimed to be a random Gaussian integration design.
    return np.concatenate((np.eye(dimension), -np.eye(dimension))) * np.sqrt((2 * dimension - 1) / 2)


def _probe_samples(field_map, indices, coordinates=None):
    if coordinates is None:
        coordinates = _exact_mean_covariance_design(field_map.dimension)
    fields = field_map.map_log10_coordinates(coordinates)
    indices = np.asarray(indices)
    return fields[:, indices[:, 0], indices[:, 1]]


def test_identity_ensemble_keeps_reference_statistics_and_zero_residuals():
    reference = _reference()
    report = reference_ensemble_coherence(reference, np.stack([reference, reference]),
                                          cell_size_m=5., max_lag_cells=5)
    assert report["reference_fidelity"]["marginal"]["wasserstein_distance_log10"] == 0
    for row in report["per_field"]:
        assert row["log10_rmse"] == 0
    for direction in report["variograms"]:
        curve = report["variograms"][direction]
        np.testing.assert_allclose(curve["reference"], curve["generated_envelope"]["values"][1])
        np.testing.assert_array_equal(report["residual_variograms"][direction]["generated_per_field"], 0.)
    np.testing.assert_array_equal(report["distribution"]["reference_density"],
                                  report["distribution"]["generated_density"])
    assert report["training_support"] == "unassessed_by_reference_comparison"
    assert report["decision_rule"] is None
    assert report["sample_filtering"] is None
    json.dumps(report, allow_nan=False)


def test_distribution_preserving_rearrangement_exposes_correlation_change():
    reference = _reference()
    rearranged = np.random.default_rng(21).permutation(reference.ravel()).reshape(reference.shape)
    report = reference_ensemble_coherence(reference, rearranged[None], cell_size_m=5., max_lag_cells=5)
    assert report["reference_fidelity"]["marginal"]["wasserstein_distance_log10"] == 0
    np.testing.assert_array_equal(report["distribution"]["reference_density"],
                                  report["distribution"]["generated_density"])
    row = report["autocorrelation"]["row"]
    assert np.max(np.abs(np.asarray(row["reference"]) - row["generated_per_field"][0])) > .3
    assert report["reference_fidelity"]["spectra"]["radial_power_relative_l2"] > .1


def test_multiplicative_shift_changes_bias_without_changing_centered_structure():
    reference = _reference()
    report = reference_ensemble_coherence(reference, (reference * 10. ** .2)[None],
                                          cell_size_m=5., max_lag_cells=4)
    assert report["per_field"][0]["log10_bias"] == pytest.approx(.2)
    for family in ("variograms", "autocorrelation"):
        for curve in report[family].values():
            np.testing.assert_allclose(curve["reference"], curve["generated_per_field"][0], atol=1e-12)
    assert report["reference_fidelity"]["spectra"]["radial_power_relative_l2"] < 1e-12


def test_field_envelopes_describe_spread_and_constants_are_json_safe():
    reference = np.full((5, 7), 1e-10)
    fields = np.stack((reference, reference * 2.))
    report = reference_ensemble_coherence(reference, fields, cell_size_m=5.)
    assert report["autocorrelation"]["row"]["reference"] == [None] * 4
    assert report["autocorrelation"]["row"]["generated_envelope"]["values"] == [[None] * 4] * 3
    assert "not a confidence interval" in report["ensemble_envelope"]["interpretation"]
    single = reference_ensemble_coherence(reference, reference[None], cell_size_m=5.)
    assert single["ensemble_envelope"]["status"] == "single_field_no_ensemble_spread"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("center", ["median", "arithmetic-mean"])
def test_probe_expected_moments_use_retained_variance_and_declared_center(center):
    field_map, _ = _map(center)
    indices = [[0, 0], [2, 3], [5, 7]]
    samples = _probe_samples(field_map, indices)
    report = reference_probe_law_diagnostics(field_map, samples, indices, sampling_method="MC")
    basis = field_map.prior.mode_matrix_at_indices(indices)
    covariance = basis @ basis.T
    expected_mean = np.log10(field_map.reference_permeability[tuple(np.asarray(indices).T)])
    if center == "arithmetic-mean":
        expected_mean -= .5 * np.log(10.) * np.diag(covariance)
        np.testing.assert_allclose(report["expected_arithmetic_ratio_to_reference"], 1., atol=1e-14)
    else:
        assert min(report["expected_arithmetic_ratio_to_reference"]) > 1.
    np.testing.assert_allclose(report["expected_log10_mean"], expected_mean)
    np.testing.assert_allclose(report["empirical_log10_mean"], expected_mean, atol=1e-14)
    np.testing.assert_allclose(report["expected_covariance_log10"], covariance, atol=1e-14)
    np.testing.assert_allclose(report["empirical_covariance_log10"], covariance, atol=1e-14)
    np.testing.assert_allclose(report["expected_log10_variance"],
                               field_map.prior.pointwise_log10_variance()[tuple(np.asarray(indices).T)])
    assert report["covariance_relative_frobenius_error"] < 1e-12
    assert report["mean_sampling_error"]["status"] == "iid_mc_reference_standard_error"
    json.dumps(report, allow_nan=False)


def test_probe_law_diagnostic_detects_wrong_amplitude_and_spatial_covariance():
    field_map, _ = _map()
    indices = [[1, 1], [1, 2], [3, 4]]
    samples = _probe_samples(field_map, indices)
    baseline = np.log10(field_map.reference_permeability[tuple(np.asarray(indices).T)])
    amplified = baseline + 2 * (samples - baseline)
    wrong_amplitude = reference_probe_law_diagnostics(field_map, amplified, indices)
    np.testing.assert_allclose(wrong_amplitude["log10_variance_ratio"], 4., atol=1e-12)
    assert wrong_amplitude["covariance_relative_frobenius_error"] == pytest.approx(3.)
    permuted = samples.copy()
    permuted[:, 1] = np.random.default_rng(12).permutation(permuted[:, 1])
    wrong_correlation = reference_probe_law_diagnostics(field_map, permuted, indices)
    np.testing.assert_allclose(wrong_correlation["log10_variance_ratio"], 1., atol=1e-12)
    assert wrong_correlation["correlation_rmse"] > .1


@pytest.mark.parametrize("center", ["median", "arithmetic-mean"])
def test_probe_conditioned_moments_use_posterior_center_and_covariance(center):
    unconditioned, _ = _map(center)
    points = np.asarray([[7.5, 12.5], [22.5, 27.5]])
    observations = unconditioned.mean_at_points(points) + [.1, -.15]
    _, field_map = _map(center, observation_coordinates_yx_m=points,
                        observation_log10_k=observations, observation_std_log10_k=.04)
    indices = [[0, 0], [1, 2], [5, 7]]
    samples = _probe_samples(field_map, indices)
    report = reference_probe_law_diagnostics(field_map, samples, indices, sampling_method="RQMC")
    basis = field_map.prior.mode_matrix_at_indices(indices)
    observation_basis = field_map.prior.mode_matrix_at_coordinates(points)
    innovation = observation_basis @ observation_basis.T + np.eye(len(points)) * .04**2
    cross = basis @ observation_basis.T
    expected_covariance = basis @ basis.T - cross @ np.linalg.solve(innovation, cross.T)
    np.testing.assert_allclose(report["expected_covariance_log10"], expected_covariance, atol=1e-13)
    np.testing.assert_allclose(report["empirical_covariance_log10"], expected_covariance, atol=1e-13)
    expected_mean = field_map.map_log10_coordinates(np.zeros(field_map.dimension))[tuple(np.asarray(indices).T)]
    np.testing.assert_allclose(report["expected_log10_mean"], expected_mean, atol=1e-13)
    assert report["conditioned"] is True
    assert report["mean_sampling_error"]["expected_log10_mean_standard_error"] is None
    assert max(abs(np.asarray(report["expected_arithmetic_ratio_to_reference"]) - 1.)) > .01
    json.dumps(report, allow_nan=False)


def test_probe_single_sample_does_not_invent_covariance_or_iid_rqmc_errors():
    field_map, _ = _map()
    indices = [[1, 2], [4, 5]]
    samples = _probe_samples(field_map, indices, np.zeros((1, field_map.dimension)))
    report = reference_probe_law_diagnostics(field_map, samples, indices, sampling_method="RQMC")
    assert report["empirical_log10_variance"] == [None, None]
    assert report["empirical_covariance_log10"] == [[None, None], [None, None]]
    assert report["covariance_relative_frobenius_error"] is None
    assert report["mean_sampling_error"]["expected_log10_mean_standard_error"] is None
    assert report["ensemble_status"] == "insufficient_fields_for_empirical_covariance"
    assert report["prefix_convergence"]["sample_counts"] == [1]
    assert report["prefix_convergence"]["covariance_relative_frobenius_error"] == [None]
    json.dumps(report, allow_nan=False)


def test_exact_cell_conditioning_marks_deterministic_probe_correlation_undefined():
    prior_map, _ = _map()
    gaussian = ConditionalKLLogGaussianPermeabilityMap(
        prior_map.prior, observation_indices=[[2, 3]],
        observation_log10_k=[.1], observation_std_log10_k=0.,
    )
    field_map = ReferenceFieldLogGaussianPermeabilityMap(prior_map.reference_permeability,
                                                        gaussian, center="median")
    indices = [[2, 3], [5, 7]]
    samples = _probe_samples(field_map, indices)
    report = reference_probe_law_diagnostics(field_map, samples, indices)
    np.testing.assert_allclose(report["expected_covariance_log10"],
                               report["empirical_covariance_log10"], atol=1e-14)
    assert report["expected_log10_variance"][0] < report["standardization_variance_zero_tolerance"]
    assert report["expected_correlation"][0] == [None, None]
    assert report["empirical_correlation"][0] == [None, None]
    assert report["standardized_residual_qq"]["probe_indices_in_qq"] == [1]
    json.dumps(report, allow_nan=False)


def test_probe_prefixes_and_iid_covariance_reference_use_field_count():
    field_map, _ = _map()
    indices = [[0, 0], [2, 3], [5, 7]]
    samples = _probe_samples(field_map, indices)
    report = reference_probe_law_diagnostics(field_map, samples, indices, sampling_method="MC")
    prefixes = report["prefix_convergence"]
    assert prefixes["sample_counts"] == [2, 4, 8, 16, 24]
    assert prefixes["log10_mean_rmse"][-1] < 1e-13
    assert prefixes["covariance_relative_frobenius_error"][-1] < 1e-12
    covariance = np.asarray(report["expected_covariance_log10"])
    norm_squared = np.sum(covariance**2)
    expected = np.sqrt((np.trace(covariance)**2 + norm_squared) / (23 * norm_squared))
    assert report["mean_sampling_error"]["iid_gaussian_expected_covariance_relative_frobenius_rmse"] == pytest.approx(expected)
    assert prefixes["iid_gaussian_expected_covariance_relative_frobenius_rmse"][-1] == pytest.approx(expected)
    np.testing.assert_allclose(report["mean_sampling_error"]["expected_log10_mean_standard_error"],
                               np.sqrt(np.diag(covariance) / len(samples)))
    rqmc = reference_probe_law_diagnostics(field_map, samples[:7], indices, sampling_method="RQMC")
    assert rqmc["prefix_convergence"]["sample_counts"] == [2, 4, 7]
    assert rqmc["prefix_convergence"]["iid_gaussian_expected_covariance_relative_frobenius_rmse"] == [None] * 3
    assert rqmc["mean_sampling_error"]["iid_gaussian_expected_covariance_relative_frobenius_rmse"] is None


def test_invalid_comparisons_are_rejected_before_reporting():
    reference = _reference()
    with pytest.raises(ValueError, match="positive finite"):
        reference_ensemble_coherence(reference, -reference[None], cell_size_m=5.)
    with pytest.raises(ValueError, match="positive finite"):
        reference_ensemble_coherence(reference, np.empty((0, *reference.shape)), cell_size_m=5.)
    field_map, _ = _map()
    with pytest.raises(ValueError, match="distinct"):
        reference_probe_law_diagnostics(field_map, np.ones((2, 2)), [[1, 2], [1, 2]])
    with pytest.raises(ValueError, match="integer"):
        reference_probe_law_diagnostics(field_map, np.ones((2, 1)), [[1.5, 2]])
    with pytest.raises(ValueError, match="probe_count"):
        reference_probe_law_diagnostics(field_map, np.ones((2, 2)), [[1, 2]])
    with pytest.raises(ValueError, match="sampling_method"):
        reference_probe_law_diagnostics(field_map, np.ones((2, 1)), [[1, 2]], sampling_method="LHS")
    with pytest.raises(ValueError, match="real-valued"):
        reference_ensemble_coherence(reference.astype(complex), reference[None], cell_size_m=5.)
    with pytest.raises(ValueError, match="real-valued"):
        reference_ensemble_coherence(reference, reference[None] + 1j, cell_size_m=5.)
    with pytest.raises(ValueError, match="real-valued"):
        reference_probe_law_diagnostics(field_map, np.ones((2, 1)) + 1j, [[1, 2]])
