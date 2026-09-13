import numpy as np

from subsurface_uq.sampling import PermeabilitySampler
from subsurface_uq.sampling.coordinates import (
    GaussianCoordinatePermeabilitySampler,
    StochasticPermeabilityMap,
)
from subsurface_uq.sampling.kl import (
    KLLogGaussianPermeabilityMap,
    matern32_correlation_matrix,
)


def test_matern32_correlation_matrix_is_symmetric_with_unit_diagonal():
    matrix = matern32_correlation_matrix(7, 700.0, 125.0)

    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.diag(matrix), np.ones(7), rtol=0.0, atol=1e-14)
    assert np.all(matrix > 0.0)


def test_factorized_full_kl_reconstructs_discrete_covariance():
    shape = (3, 4)
    domain = (300.0, 400.0)
    length_scale = (90.0, 140.0)
    sigma = 0.37
    field_map = KLLogGaussianPermeabilityMap(
        shape=shape,
        domain_size_m=domain,
        mean_log10_k=0.0,
        std_log10_k=sigma,
        length_scale_m=length_scale,
        n_modes=shape[0] * shape[1],
    )

    coordinates = np.eye(field_map.dimension)
    basis_columns = field_map.map_log10_coordinates(coordinates).reshape(
        field_map.dimension, -1
    ).T
    reconstructed = basis_columns @ basis_columns.T

    corr_y = matern32_correlation_matrix(shape[0], domain[0], length_scale[0])
    corr_x = matern32_correlation_matrix(shape[1], domain[1], length_scale[1])
    expected = sigma**2 * np.kron(corr_y, corr_x)

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(field_map.retained_energy_fraction, 1.0, atol=1e-12)


def test_zero_coordinates_map_to_geometric_mean_permeability():
    field_map = KLLogGaussianPermeabilityMap(
        shape=(5, 6),
        domain_size_m=(500.0, 600.0),
        mean_log10_k=-9.5,
        std_log10_k=0.4,
        length_scale_m=(160.0, 160.0),
        energy_threshold=0.9,
    )

    permeability = field_map.map_coordinates(np.zeros(field_map.dimension))

    assert permeability.shape == field_map.field_shape
    np.testing.assert_allclose(
        permeability,
        np.full(field_map.field_shape, 10.0**-9.5, dtype=np.float32),
        rtol=2e-6,
    )


def test_energy_threshold_selects_leading_modes():
    threshold = 0.95
    field_map = KLLogGaussianPermeabilityMap(
        shape=(8, 9),
        domain_size_m=(800.0, 900.0),
        mean_log10_k=-9.5,
        std_log10_k=0.3,
        length_scale_m=(220.0, 180.0),
        energy_threshold=threshold,
    )

    assert 1 <= field_map.dimension <= 8 * 9
    assert field_map.retained_energy_fraction >= threshold
    assert np.all(np.diff(field_map.eigenvalues) <= 0.0)
    assert np.all(field_map.eigenvalues > 0.0)


def test_coordinate_map_supports_single_and_batched_inputs():
    field_map = KLLogGaussianPermeabilityMap(
        shape=(4, 5),
        domain_size_m=(400.0, 500.0),
        mean_log10_k=-9.0,
        std_log10_k=0.25,
        length_scale_m=(100.0, 120.0),
        n_modes=4,
    )

    single = field_map.map_coordinates(np.zeros(4))
    batch = field_map.map_coordinates(np.zeros((3, 4)))

    assert single.shape == (4, 5)
    assert batch.shape == (3, 4, 5)
    np.testing.assert_allclose(batch[0], single)
    assert isinstance(field_map, StochasticPermeabilityMap)


def test_gaussian_coordinate_sampler_is_reproducible_and_protocol_compatible():
    field_map = KLLogGaussianPermeabilityMap(
        shape=(4, 4),
        domain_size_m=(400.0, 400.0),
        mean_log10_k=-9.4,
        std_log10_k=0.2,
        length_scale_m=(120.0, 120.0),
        n_modes=5,
    )
    sampler = GaussianCoordinatePermeabilitySampler(
        field_map=field_map,
        n_samples=5,
        batch_size=2,
        seed=123,
    )

    first = np.concatenate(list(sampler), axis=0)
    second = np.concatenate(list(sampler), axis=0)

    assert isinstance(sampler, PermeabilitySampler)
    assert first.shape == (5, 4, 4)
    assert np.all(first > 0.0)
    np.testing.assert_array_equal(first, second)


def test_invalid_coordinate_dimension_is_rejected():
    field_map = KLLogGaussianPermeabilityMap(
        shape=(3, 3),
        domain_size_m=(300.0, 300.0),
        mean_log10_k=-9.0,
        std_log10_k=0.2,
        length_scale_m=(100.0, 100.0),
        n_modes=3,
    )

    try:
        field_map.map_coordinates(np.zeros(2))
    except ValueError as exc:
        assert "expected stochastic dimension 3" in str(exc)
    else:
        raise AssertionError("dimension mismatch should raise ValueError")
