"""Distributional checks for synthetic-reference measurement conditioning.

Dense Gaussian formulae and empirical draws independently verify the conditional
law, including the extra simulated observation noise required by Matheron's rule.
"""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gstools")

from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec
from subsurface_uq.sampling.reference_conditioning import (
    ConditionalReferenceFFTSampler,
    LocationPriorSpec,
    draw_reference_measurements,
)


def _reference(shape=(10, 9)):
    row, col = np.indices(shape)
    return 10. ** (-10 + .035 * row - .05 * col + .12 * np.sin(col))


def _observations(reference, spec, **overrides):
    settings = dict(covariance=spec, n_measurements=3, location_seed=31,
                    observation_seed=73, cell_size_m=1.,
                    location_prior=LocationPriorSpec())
    settings.update(overrides)
    return draw_reference_measurements(reference, **settings)


def _covariance(spec, query, observations):
    offsets = np.asarray(query)[:, None, :] - np.asarray(observations)[None, :, :]
    return spec.sigma_R ** 2 * spec.correlation(offsets[..., 0], offsets[..., 1])


@pytest.mark.parametrize("family,angle", [
    ("gaussian", 0.), ("matern32_radial", 0.), ("exponential_radial", 0.),
    ("gaussian", 37.),
])
@pytest.mark.parametrize("mean_mode", ["reference_median", "global_log_mean"])
def test_fixed_design_posterior_matches_dense_gaussian_formula(family, angle, mean_mode):
    reference = _reference()
    spec = GSToolsReferenceSpec(family, 2.5, 1.4, sigma_R=.2, angle_deg=angle)
    observations = _observations(reference, spec, measurement_mode="noisy",
                                 observation_std_log10_k=.12)
    sampler = ConditionalReferenceFFTSampler(reference, spec, observations,
        cell_size_m=1., mean_mode=mean_mode, workers=1, chunk_size=13)
    query = np.array([[0, 0], [0, 8], [4, 5], [9, 8], *observations.indices.tolist()])
    log_reference = np.log10(reference)
    if mean_mode == "reference_median":
        prior_query_mean = log_reference[tuple(query.T)]
        prior_observation_mean = log_reference[tuple(observations.indices.T)]
    else:
        prior_query_mean = np.full(len(query), log_reference.mean())
        prior_observation_mean = np.full(observations.indices.shape[0], log_reference.mean())
    cqd = _covariance(spec, query, observations.indices)
    cdd = _covariance(spec, observations.indices, observations.indices)
    cqq = _covariance(spec, query, query)
    innovation = cdd + np.diag(observations.observation_std_log10_k**2)
    expected_mean = prior_query_mean + cqd @ np.linalg.solve(
        innovation, observations.observed_log10_k - prior_observation_mean)
    expected_covariance = cqq - cqd @ np.linalg.solve(innovation, cqd.T)
    actual_mean, actual_covariance = sampler.posterior_at_indices(query)
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_covariance, expected_covariance, rtol=1e-9, atol=1e-10)
    grid_mean, grid_variance = sampler.posterior_moments()
    np.testing.assert_allclose(grid_mean[tuple(query.T)], expected_mean, atol=5e-7)
    np.testing.assert_allclose(grid_variance[tuple(query.T)], np.diag(expected_covariance), atol=1e-9)
    assert np.linalg.eigvalsh(actual_covariance).min() >= -1e-10


def test_exact_reference_measurements_pin_draws_and_leave_reference_center_unchanged():
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 2.3, 1.7, sigma_R=.2)
    observations = _observations(reference, spec)
    sampler = ConditionalReferenceFFTSampler(reference, spec, observations,
                                           cell_size_m=1., workers=1)
    mean, covariance = sampler.posterior_at_indices(observations.indices)
    np.testing.assert_allclose(mean, observations.reference_log10_k, atol=1e-10)
    np.testing.assert_allclose(covariance, 0, atol=1e-10)
    grid_mean, grid_variance = sampler.posterior_moments()
    np.testing.assert_allclose(grid_mean, np.log10(reference), atol=5e-7)
    assert np.any(grid_variance > .001)
    for seed in (0, 5, 803):
        field = sampler.sample_log10(seed)
        np.testing.assert_allclose(field[tuple(observations.indices.T)],
                                   observations.observed_log10_k, atol=1e-8)
        physical = sampler.sample_permeability(seed)
        assert np.isfinite(physical).all() and np.all(physical > 0)
        np.testing.assert_allclose(physical[tuple(observations.indices.T)],
                                   reference[tuple(observations.indices.T)], rtol=1e-6)


def test_constant_mean_reconstruction_updates_mean_from_sparse_reference_values():
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 2.3, 1.7, sigma_R=.2)
    observations = _observations(reference, spec)
    sampler = ConditionalReferenceFFTSampler(reference, spec, observations,
                         cell_size_m=1., mean_mode="global_log_mean", workers=1)
    mean, variance = sampler.posterior_moments()
    np.testing.assert_allclose(mean[tuple(observations.indices.T)],
                               observations.reference_log10_k, atol=5e-7)
    assert np.ptp(mean) > .05
    assert not np.allclose(mean, np.log10(reference), atol=1e-4)
    assert np.all(variance >= -1e-10)


def test_location_observation_and_field_seeds_control_separate_randomness():
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 2.3, 1.7, sigma_R=.2)
    first = _observations(reference, spec, measurement_mode="noisy", observation_std_log10_k=.15)
    repeat = _observations(reference, spec, measurement_mode="noisy", observation_std_log10_k=.15)
    other_noise = _observations(reference, spec, measurement_mode="noisy",
                                observation_std_log10_k=.15, observation_seed=74)
    np.testing.assert_array_equal(first.indices, repeat.indices)
    np.testing.assert_array_equal(first.observed_log10_k, repeat.observed_log10_k)
    np.testing.assert_array_equal(first.indices, other_noise.indices)
    assert not np.array_equal(first.observed_log10_k, other_noise.observed_log10_k)
    np.testing.assert_array_equal(first.reference_log10_k, np.log10(reference)[tuple(first.indices.T)])
    sampler = ConditionalReferenceFFTSampler(reference, spec, first, cell_size_m=1., workers=1)
    np.testing.assert_array_equal(sampler.sample_log10(99), sampler.sample_log10(99))
    assert not np.array_equal(sampler.sample_log10(99), sampler.sample_log10(100))


def test_noisy_conditional_draws_recover_analytic_mean_and_covariance():
    reference = _reference((8, 7))
    spec = GSToolsReferenceSpec("exponential_radial", 2.1, 1.5, sigma_R=.2)
    observations = _observations(reference, spec, measurement_mode="noisy",
                                 observation_std_log10_k=.2)
    sampler = ConditionalReferenceFFTSampler(reference, spec, observations,
        cell_size_m=1., mean_mode="global_log_mean", workers=1, chunk_size=19)
    query = np.array([*observations.indices.tolist(), [0, 0], [7, 6]])
    expected_mean, expected_covariance = sampler.posterior_at_indices(query)
    values = np.array([sampler.sample_log10(seed)[tuple(query.T)] for seed in range(1600)])
    np.testing.assert_allclose(values.mean(axis=0), expected_mean, atol=.018)
    # A correction omitting independent simulated measurement noise roughly
    # halves variance at noisy observation sites and fails this assertion.
    np.testing.assert_allclose(np.cov(values, rowvar=False), expected_covariance,
                               rtol=.12, atol=.0011)


def test_nonnegative_location_weights_define_allowed_support_and_distinct_cells():
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 2.3, 1.7)
    weights = np.zeros(reference.shape)
    allowed = np.array([[1, 2], [4, 5], [8, 7]])
    weights[tuple(allowed.T)] = [1., 3., 7.]
    selected = _observations(reference, spec, location_weights=weights)
    assert set(map(tuple, selected.indices)) == set(map(tuple, allowed))
    assert np.all(selected.selection_probabilities > 0)
    assert np.all(selected.selection_probabilities <= 1)
    with pytest.raises((ValueError, RuntimeError)):
        _observations(reference, spec, location_weights=weights, n_measurements=4)


def test_margin_stride_and_minimum_spacing_apply_in_physical_coordinates():
    reference = _reference((12, 11))
    spec = GSToolsReferenceSpec("exponential_radial", 4., 3.)
    prior = LocationPriorSpec(margin_cells=1, candidate_stride_cells=2, min_spacing_m=3.)
    selected = _observations(reference, spec, location_prior=prior, n_measurements=6)
    assert np.all(selected.indices >= 1)
    assert np.all(selected.indices[:, 0] < reference.shape[0]-1)
    assert np.all(selected.indices[:, 1] < reference.shape[1]-1)
    assert np.all((selected.indices-1) % 2 == 0)
    distances = np.linalg.norm(selected.indices[:, None, :]-selected.indices[None, :, :], axis=2)
    assert distances[np.triu_indices(len(selected.indices), 1)].min() >= 3.
    with pytest.raises((ValueError, RuntimeError)):
        _observations(reference, spec, location_prior=LocationPriorSpec(min_spacing_m=100.))


def test_variance_prior_starts_uniform_then_uses_conditional_covariance():
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 4., 2.)
    selected = _observations(reference, spec, n_measurements=6,
        location_prior=LocationPriorSpec(kind="variance", exploration_fraction=0.))
    assert selected.selection_probabilities[0] == pytest.approx(1/reference.size)
    uniform_without_replacement = 1/(reference.size-np.arange(6))
    assert np.any(np.abs(selected.selection_probabilities[1:]-uniform_without_replacement[1:]) > 1e-4)


def test_gradient_prior_prefers_reference_structural_changes_and_handles_flat_reference():
    reference = np.ones((10, 9))*1e-10
    reference[:, 4:] *= 10
    spec = GSToolsReferenceSpec("exponential_radial", 2., 1.)
    prior = LocationPriorSpec(kind="gradient", exploration_fraction=0.)
    picks = [_observations(reference, spec, n_measurements=1, location_seed=seed,
                           location_prior=prior).indices[0, 1] for seed in range(100)]
    assert np.count_nonzero(np.isin(picks, [3, 4])) >= 60
    flat = _observations(np.full((10, 9), 1e-10), spec, n_measurements=1, location_prior=prior)
    assert flat.selection_probabilities[0] == pytest.approx(1/reference.size)


@pytest.mark.parametrize("kwargs", [
    {"kind": "unknown"}, {"variance_power": -1}, {"gradient_power": -1},
    {"exploration_fraction": -0.01}, {"exploration_fraction": 1.01},
    {"candidate_stride_cells": 0}, {"margin_cells": -1}, {"min_spacing_m": -1},
    {"variance_power": float("nan")}, {"min_spacing_m": float("inf")},
])
def test_invalid_location_priors_are_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LocationPriorSpec(**kwargs)


@pytest.mark.parametrize("mode,std", [("exact", .1), ("noisy", 0.), ("noisy", -.1), ("other", .1)])
def test_measurement_generation_rejects_inconsistent_noise_assumptions(mode, std):
    with pytest.raises((ValueError, TypeError)):
        _observations(_reference(), GSToolsReferenceSpec("exponential_radial", 2., 1.),
                       measurement_mode=mode, observation_std_log10_k=std)


@pytest.mark.parametrize("bad_weights", [
    np.full((10, 9), -1.), np.full((10, 9), np.nan), np.zeros((10, 9)), np.ones((2, 2)),
])
def test_invalid_or_empty_weighted_location_support_is_rejected(bad_weights):
    with pytest.raises((ValueError, TypeError)):
        _observations(_reference(), GSToolsReferenceSpec("exponential_radial", 2., 1.),
                       location_weights=bad_weights)


@pytest.mark.parametrize("invalid_indices", [
    np.array([[1, 2], [1, 2], [3, 4]]),
    np.array([[1, 2], [10, 2], [3, 4]]),
    np.array([[1, 2], [-1, 2], [3, 4]]),
    np.array([[1, 2], [1.5, 2], [3, 4]]),
])
def test_duplicate_noninteger_and_outside_observation_cells_are_rejected(invalid_indices):
    reference = _reference()
    spec = GSToolsReferenceSpec("exponential_radial", 2., 1.)
    observations = _observations(reference, spec)
    with pytest.raises((ValueError, TypeError)):
        invalid = replace(observations, indices=invalid_indices)
        ConditionalReferenceFFTSampler(reference, spec, invalid, cell_size_m=1., workers=1)


def test_incompatible_exact_data_are_rejected_when_covariance_is_numerically_rank_one():
    reference = _reference()
    spec = GSToolsReferenceSpec("gaussian", 1e12, 1e12, sigma_R=.2)
    observations = _observations(reference, spec)
    # At these lengths C_DD has only a constant mode in floating point. A
    # constant prior mean cannot fit varying exact values through that mode.
    # This must fail before constructing an enormous FFT embedding, rather
    # than silently adding a nugget or claiming interpolation.
    assert np.ptp(observations.reference_log10_k) > .01
    with pytest.raises(ValueError):
        ConditionalReferenceFFTSampler(reference, spec, observations,
            cell_size_m=1., mean_mode="global_log_mean", workers=1)


def test_conditional_mean_semantics_reject_conflicting_arithmetic_mean_spec():
    reference = _reference()
    median_spec = GSToolsReferenceSpec("exponential_radial", 2., 1.)
    observations = _observations(reference, median_spec)
    conflicting = replace(median_spec, center="arithmetic-mean")
    with pytest.raises(ValueError):
        ConditionalReferenceFFTSampler(reference, conflicting, observations,
            cell_size_m=1., mean_mode="reference_median", workers=1)


def test_dense_smooth_exact_design_does_not_silently_lose_constraints():
    reference = np.full((12, 10), 1e-10)
    spec = GSToolsReferenceSpec("gaussian", 20., 20., sigma_R=.2)
    observations = _observations(reference, spec, n_measurements=reference.size)
    with pytest.raises(ValueError, match="numerical rank"):
        ConditionalReferenceFFTSampler(reference, spec, observations, cell_size_m=1., workers=1)
