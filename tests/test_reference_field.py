from types import SimpleNamespace

import numpy as np
import pytest

from subsurface_uq.sampling.reference_field import build_reference_field_maps, save_reference_field_input_model
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.coordinates import GaussianCoordinatePermeabilitySampler
from subsurface_uq.sampling.qmc import ScrambledSobolGaussianPermeabilitySampler
from subsurface_uq.rq1.workflow import _build_grf_maps


def _maps(center="median", **kwargs):
    reference = np.power(10., -9.5 + np.arange(20).reshape(4, 5) / 100)
    return build_reference_field_maps(
        reference, cell_size_m=20., residual_std_log10_k=.15,
        length_scale_m=(50., 60.), energy_threshold=.99, center=center, **kwargs,
    )


def test_reference_median_and_retained_covariance_are_distinct_components():
    model, _ = _maps()
    zero = np.zeros(model.dimension)
    np.testing.assert_allclose(model.map_coordinates(zero), model.reference_permeability, rtol=1e-6)
    basis = model.prior.map_log10_coordinates(np.eye(model.dimension)).reshape(model.dimension, -1).T
    np.testing.assert_allclose(model.prior.pointwise_log10_variance().ravel(), np.sum(basis ** 2, axis=1))
    np.testing.assert_allclose(model.map_log10_coordinates(np.eye(model.dimension))
                               - np.log10(model.reference_permeability), basis.T.reshape(model.dimension, 4, 5), atol=1e-14)


def test_arithmetic_mean_correction_uses_actual_truncated_variance():
    model, _ = _maps(center="arithmetic-mean")
    m = model.map_log10_coordinates(np.zeros(model.dimension))
    expectation = np.exp(np.log(10.) * m + .5 * np.log(10.) ** 2 * model.prior.pointwise_log10_variance())
    np.testing.assert_allclose(expectation, model.reference_permeability, rtol=1e-12)


def test_noisy_conditioning_subtracts_the_same_spatial_mean_and_matches_dense_posterior():
    unconditioned, _ = _maps()
    points = np.array([[10., 10.], [55., 72.]])
    data = unconditioned.mean_at_points(points) + [.2, -.1]
    unconditional, conditioned = _maps(
        observation_coordinates_yx_m=points, observation_log10_k=data, observation_std_log10_k=.07,
    )
    A = unconditional.prior.mode_matrix_at_coordinates(points)
    innovation = A @ A.T + np.eye(2) * .07 ** 2
    mean_xi = A.T @ np.linalg.solve(innovation, data - unconditional.mean_at_points(points))
    expected_mean = unconditional.mean_at_points(points) + A @ mean_xi
    np.testing.assert_allclose(conditioned.map_log10_coordinates_at_points(np.zeros(conditioned.dimension), points),
                               expected_mean, atol=1e-12)
    coordinates = np.eye(conditioned.dimension)
    centered = conditioned.map_log10_coordinates_at_points(coordinates, points) - expected_mean
    posterior_covariance = A @ (np.eye(A.shape[1]) - A.T @ np.linalg.solve(innovation, A)) @ A.T
    np.testing.assert_allclose(centered.T @ centered, posterior_covariance, atol=1e-12)
    assert np.max(np.abs(expected_mean - data)) > 1e-4
    assert np.all(np.diag(posterior_covariance) > 0)


def test_reference_artifact_roundtrip_drives_rq1_and_checks_corruption(tmp_path):
    model, _ = _maps()
    path = tmp_path / "input.yaml"
    save_reference_field_input_model(path, model)
    loaded = load_conditional_kl_input_model(path)
    xi = np.random.default_rng(7).normal(size=(2, model.dimension))
    np.testing.assert_array_equal(loaded.unconditional.map_coordinates(xi), model.map_coordinates(xi))
    prior, conditional, _ = _build_grf_maps(SimpleNamespace(input_model=path, cell_size_m=20.), model.field_shape)
    np.testing.assert_array_equal(prior.map_coordinates(xi), conditional.map_coordinates(xi))
    (tmp_path / "input.reference.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_conditional_kl_input_model(path)


def test_reference_field_reuses_mc_and_rqmc_without_rejection():
    model, _ = _maps()
    for sampler_type in (GaussianCoordinatePermeabilitySampler, ScrambledSobolGaussianPermeabilitySampler):
        first = np.concatenate(list(sampler_type(field_map=model, n_samples=8, batch_size=2, seed=19)))
        second = np.concatenate(list(sampler_type(field_map=model, n_samples=8, batch_size=2, seed=19)))
        np.testing.assert_array_equal(first, second)
        assert first.shape == (8, 4, 5)
        assert np.all(first > 0)


def test_conditioned_reference_artifact_roundtrip_preserves_measurement_update(tmp_path):
    model, _ = _maps(center="arithmetic-mean")
    points = [[10., 10.], [50., 65.]]
    observations = dict(observation_coordinates_yx_m=points,
                        observation_log10_k=model.mean_at_points(points) + [.1, -.2],
                        observation_std_log10_k=[.03, .08])
    unconditional, conditional = _maps(center="arithmetic-mean", **observations)
    path = tmp_path / "conditioned.yaml"
    save_reference_field_input_model(path, unconditional, **observations)
    loaded = load_conditional_kl_input_model(path)
    xi = np.random.default_rng(20).normal(size=(3, model.dimension))
    np.testing.assert_array_equal(loaded.conditional.map_coordinates(xi), conditional.map_coordinates(xi))


def test_outside_reference_domain_and_partial_observations_are_rejected():
    model, _ = _maps()
    with pytest.raises(ValueError, match="inside"):
        model.mean_at_points([[90., 20.]])
    with pytest.raises(ValueError, match="conditioning requires"):
        _maps(observation_coordinates_yx_m=[[10., 10.]])
