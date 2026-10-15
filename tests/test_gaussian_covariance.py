"""Regression coverage for the Gaussian/RBF residual covariance family."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments.reference_field_permeability import main
from subsurface_uq.pce import PolynomialChaosRegressor, iid_gaussian_design
from subsurface_uq.rq1.config import load_rq1_config
from subsurface_uq.sampling.coordinates import GaussianCoordinatePermeabilitySampler
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.kl import (
    KLLogGaussianPermeabilityMap,
    gaussian_correlation_matrix,
)
from subsurface_uq.sampling.kriging import ContinuousPointConditionalKLLogGaussianPermeabilityMap
from subsurface_uq.sampling.qmc import ScrambledSobolGaussianPermeabilitySampler
from subsurface_uq.sampling.reference_field import build_reference_field_maps, save_reference_field_input_model


def _prior(**kwargs):
    return KLLogGaussianPermeabilityMap(
        shape=(3, 4), domain_size_m=(300., 400.), mean_log10_k=0.,
        std_log10_k=.37, length_scale_m=(90., 140.),
        covariance_model="gaussian", **kwargs,
    )


def test_gaussian_matrix_has_documented_convention_and_is_positive_definite():
    matrix = gaussian_correlation_matrix(7, 700., 125.)
    locations = (np.arange(7) + .5) * 100.
    expected = np.exp(-.5 * ((locations[:, None] - locations[None, :]) / 125.) ** 2)
    np.testing.assert_array_equal(matrix, expected)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), np.ones(7))
    assert np.linalg.eigvalsh(matrix).min() > 0


@pytest.mark.parametrize("args", [(0, 100., 25.), (3, 0., 25.), (3, np.inf, 25.), (3, 100., -1.)])
def test_gaussian_matrix_rejects_invalid_geometry(args):
    with pytest.raises(ValueError):
        gaussian_correlation_matrix(*args)


def test_gaussian_full_kl_reconstructs_covariance_and_continuous_extension():
    prior = _prior(n_modes=12)
    cells = np.indices(prior.field_shape).reshape(2, -1).T
    centers = (cells + .5) * 100.
    basis = prior.mode_matrix_at_indices(cells)
    expected = .37**2 * np.kron(
        gaussian_correlation_matrix(3, 300., 90.),
        gaussian_correlation_matrix(4, 400., 140.),
    )
    np.testing.assert_allclose(basis @ basis.T, expected, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(prior.mode_matrix_at_coordinates(centers), basis, rtol=2e-11, atol=1e-12)
    # With every mode retained, the Nystrom cross-covariance from an off-grid
    # query to all grid centers must be exactly the declared Gaussian kernel.
    query = np.array([[40., 70.], [165., 245.], [290., 390.]])
    separation = (query[:, None, :] - centers[None, :, :]) / [90., 140.]
    expected_cross = .37**2 * np.exp(-.5 * np.sum(separation**2, axis=-1))
    np.testing.assert_allclose(prior.mode_matrix_at_coordinates(query) @ basis.T,
                               expected_cross, rtol=2e-11, atol=1e-12)
    assert prior.metadata["covariance"] == "separable_gaussian"
    assert prior.retained_energy_fraction == pytest.approx(1.)


def test_gaussian_off_grid_conditioning_matches_dense_posterior():
    prior = _prior(n_modes=8)
    points = np.array([[40., 70.], [165., 245.]])
    observations = np.array([.1, -.2])
    noise = .08
    posterior = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
        prior, points, observations, noise,
    )
    a = prior.mode_matrix_at_coordinates(points)
    innovation = a @ a.T + noise**2 * np.eye(2)
    mean = a.T @ np.linalg.solve(innovation, observations)
    covariance = np.eye(prior.dimension) - a.T @ np.linalg.solve(innovation, a)
    query = np.array([[90., 120.], [280., 370.]])
    b = prior.mode_matrix_at_coordinates(query)
    actual_mean, actual_std = posterior.posterior_moments_at_points(query)
    np.testing.assert_allclose(actual_mean, b @ mean, atol=1e-12)
    np.testing.assert_allclose(actual_std**2, np.einsum("im,mn,in->i", b, covariance, b), atol=1e-12)


@pytest.mark.parametrize("center", ["median", "arithmetic-mean"])
@pytest.mark.parametrize("conditioned", [False, True])
def test_gaussian_reference_artifact_roundtrip(tmp_path, center, conditioned):
    reference = 10. ** np.linspace(-10., -9., 12).reshape(3, 4)
    kwargs = dict(cell_size_m=100., residual_std_log10_k=.1,
                  length_scale_m=(90., 140.), covariance_model="gaussian", n_modes=5, center=center)
    observations = {}
    if conditioned:
        observations = dict(observation_coordinates_yx_m=[[40., 70.], [165., 245.]],
                            observation_log10_k=[-9.9, -9.4], observation_std_log10_k=.07)
    unconditional, conditional = build_reference_field_maps(reference, **kwargs, **observations)
    artifact = tmp_path / "input.yaml"
    save_reference_field_input_model(artifact, unconditional, **observations)
    loaded = load_conditional_kl_input_model(artifact)
    coordinates = np.random.default_rng(3).normal(size=(3, conditional.dimension))
    np.testing.assert_array_equal(loaded.unconditional.map_coordinates(coordinates), unconditional.map_coordinates(coordinates))
    np.testing.assert_array_equal(loaded.conditional.map_coordinates(coordinates), conditional.map_coordinates(coordinates))
    assert loaded.payload["coordinate_distribution"] == "iid_standard_normal"
    assert loaded.payload["prior"]["covariance"] == "separable_gaussian"


@pytest.mark.parametrize("sampler_type", [GaussianCoordinatePermeabilitySampler, ScrambledSobolGaussianPermeabilitySampler])
def test_gaussian_kernel_retains_reproducible_normal_coordinate_samplers(sampler_type):
    prior = _prior(n_modes=5)
    first = np.concatenate(list(sampler_type(prior, n_samples=8, batch_size=2, seed=9)))
    second = np.concatenate(list(sampler_type(prior, n_samples=8, batch_size=4, seed=9)))
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first)) and np.all(first > 0)


def test_gaussian_kernel_log_response_is_exactly_linear_in_hermite_coordinates():
    prior = _prior(n_modes=5)
    design = iid_gaussian_design(40, prior.dimension, seed=8)
    # A log-field QoI has a known exact linear Gaussian response. Recovering
    # its moments validates the same coordinate contract used by response PCE.
    response = prior.map_log10_coordinates(design).mean(axis=(1, 2))
    regressor = PolynomialChaosRegressor(prior.dimension, 1, basis_family="hermite").fit(design, response)
    validation = iid_gaussian_design(16, prior.dimension, seed=9)
    np.testing.assert_allclose(regressor.predict(validation), prior.map_log10_coordinates(validation).mean(axis=(1, 2)), atol=1e-13)
    weights = prior.map_log10_coordinates(np.eye(prior.dimension)).mean(axis=(1, 2))
    assert regressor.mean == pytest.approx(0., abs=1e-13)
    assert regressor.variance == pytest.approx(float(weights @ weights))


def test_gaussian_generator_cli_writes_reconstructible_artifact(tmp_path):
    source = tmp_path / "reference.npy"
    reference = 10. ** np.linspace(-10., -9.8, 64).reshape(8, 8)
    np.save(source, reference)
    output = tmp_path / "output"
    assert main(["--reference-npy", str(source), "--reference-run", "RUN_2",
                 "--permeability-convention", "historical-training", "--covariance-model", "gaussian",
                 "--residual-std-log10-k", ".05", "--length-scale-row-m", "20", "--length-scale-col-m", "30",
                 "--n-modes", "3", "--n-samples", "2", "--output-dir", str(output)]) == 0
    loaded = load_conditional_kl_input_model(output / "modes_3/stochastic_input_model.yaml")
    assert loaded.prior.metadata["covariance"] == "separable_gaussian"
    assert loaded.payload["scenario"]["config"]["covariance_family"] == "gaussian"
    assert np.load(output / "modes_3/generated_fields.npy").shape == (2, 8, 8)
    assert json.loads((output / "summary.json").read_text())["distribution_target"] is None


def test_rq1_accepts_gaussian_covariance_and_serializes_name(tmp_path):
    template = Path(__file__).resolve().parents[1] / "configs/rq1.example.yaml"
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    payload["grf"]["covariance_model"] = "gaussian"
    source = tmp_path / "rq1.yaml"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_rq1_config(source)
    assert config.covariance_model == "gaussian"
    assert config.to_dict()["grf"]["covariance_model"] == "gaussian"


def test_unknown_covariance_name_remains_rejected():
    with pytest.raises(ValueError, match="covariance_model"):
        KLLogGaussianPermeabilityMap((3, 4), (300., 400.), 0., .1, (90., 140.), covariance_model="rbf_typo")
