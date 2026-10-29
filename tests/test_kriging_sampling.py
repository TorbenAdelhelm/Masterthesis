import numpy as np
import pytest

from subsurface_uq.sampling import (
    ConditionalKLLogGaussianPermeabilityMap,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
    StochasticPermeabilityMap,
)


def _prior(*, n_modes: int = 8) -> KLLogGaussianPermeabilityMap:
    return KLLogGaussianPermeabilityMap(
        shape=(4, 5),
        domain_size_m=(400.0, 500.0),
        mean_log10_k=-9.4,
        std_log10_k=0.35,
        length_scale_m=(140.0, 180.0),
        n_modes=n_modes,
    )


def test_mode_matrix_matches_prior_field_at_selected_cells():
    prior = _prior(n_modes=6)
    cells = np.asarray([[0, 0], [1, 3], [3, 4]])
    xi = np.asarray([0.2, -0.4, 0.1, 0.5, -0.2, 0.3])

    field = prior.map_log10_coordinates(xi)
    A = prior.mode_matrix_at_indices(cells)
    reconstructed = prior.mean_log10_k + A @ xi

    np.testing.assert_allclose(
        reconstructed,
        field[cells[:, 0], cells[:, 1]],
        rtol=1e-12,
        atol=1e-12,
    )


def test_exact_conditioning_honours_borehole_cells_for_every_realization():
    prior = _prior(n_modes=8)
    cells = np.asarray([[0, 1], [2, 3]])
    xi_reference = np.asarray([0.4, -0.3, 0.1, 0.2, -0.1, 0.5, -0.2, 0.15])
    reference = prior.map_log10_coordinates(xi_reference)
    observed = reference[cells[:, 0], cells[:, 1]]

    conditioned = ConditionalKLLogGaussianPermeabilityMap(
        prior=prior,
        observation_indices=cells,
        observation_log10_k=observed,
        observation_std_log10_k=0.0,
    )
    eta = np.random.default_rng(42).standard_normal((12, conditioned.dimension))
    fields = conditioned.map_log10_coordinates(eta)

    assert conditioned.dimension < prior.dimension
    assert isinstance(conditioned, StochasticPermeabilityMap)
    for row, col, value in zip(cells[:, 0], cells[:, 1], observed):
        np.testing.assert_allclose(fields[:, row, col], value, rtol=0.0, atol=1e-10)


def test_gaussian_sampler_reuses_independent_posterior_coordinates_reproducibly():
    prior = _prior(n_modes=7)
    cells = np.asarray([[1, 1], [3, 2]])
    reference = prior.map_log10_coordinates(np.linspace(-0.3, 0.3, prior.dimension))
    conditioned = ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
        prior=prior,
        observation_indices=cells,
        observation_k=10.0 ** reference[cells[:, 0], cells[:, 1]],
    )
    sampler = GaussianCoordinatePermeabilitySampler(
        field_map=conditioned,
        n_samples=5,
        batch_size=2,
        seed=1234,
    )

    first = np.concatenate(list(sampler), axis=0)
    second = np.concatenate(list(sampler), axis=0)

    assert first.shape == (5, *prior.field_shape)
    assert np.all(np.isfinite(first))
    assert np.all(first > 0.0)
    np.testing.assert_array_equal(first, second)
    assert sampler.metadata["coordinate_distribution"] == "iid_standard_normal"
    assert sampler.metadata["coordinate_dimension"] == conditioned.dimension


def test_noisy_conditioning_keeps_full_coordinate_rank_and_reduces_uncertainty():
    prior = _prior(n_modes=6)
    cells = np.asarray([[1, 2]])
    observed = np.asarray([prior.mean_log10_k + 0.2])
    conditioned = ConditionalKLLogGaussianPermeabilityMap(
        prior=prior,
        observation_indices=cells,
        observation_log10_k=observed,
        observation_std_log10_k=0.1,
    )

    assert conditioned.dimension == prior.dimension
    posterior_cov = conditioned.posterior_coordinate_covariance
    assert np.trace(posterior_cov) < prior.dimension
    assert np.all(np.linalg.eigvalsh(posterior_cov) >= -1e-10)


def test_incompatible_exact_data_is_rejected_for_too_small_kl_subspace():
    prior = _prior(n_modes=1)
    cells = np.asarray([[0, 0], [3, 4]])
    observed = np.asarray([prior.mean_log10_k + 1.0, prior.mean_log10_k - 1.0])

    with pytest.raises(ValueError, match="incompatible with the retained KL subspace"):
        ConditionalKLLogGaussianPermeabilityMap(
            prior=prior,
            observation_indices=cells,
            observation_log10_k=observed,
            observation_std_log10_k=0.0,
        )


def test_duplicate_conditioning_cells_are_rejected():
    prior = _prior(n_modes=4)

    with pytest.raises(ValueError, match="duplicate conditioning cells"):
        ConditionalKLLogGaussianPermeabilityMap(
            prior=prior,
            observation_indices=np.asarray([[1, 1], [1, 1]]),
            observation_log10_k=np.asarray([-9.3, -9.2]),
        )
