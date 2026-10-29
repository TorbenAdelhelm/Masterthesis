import numpy as np

from subsurface_uq.sampling import PermeabilitySampler
from subsurface_uq.sampling.coordinates import (
    StochasticPermeabilityMap,
    UniformCoordinatePermeabilitySampler,
)
from subsurface_uq.sampling.perlin import historical_perlin_v2_field
from subsurface_uq.sampling.perlin_coordinates import (
    RELEASE25_PERLIN_OFFSET_SPAN,
    PerlinCoordinatePermeabilityMap,
)


def _small_map(**overrides):
    kwargs = {
        "shape": (12, 13),
        "domain_size_m": (120.0, 130.0),
        "frequency": (5.0, 6.0),
        "k_min": 1.0e-11,
        "k_max": 5.0e-9,
        "offset_bounds": ((1.0, 3.0), (2.0, 4.0)),
    }
    kwargs.update(overrides)
    return PerlinCoordinatePermeabilityMap(**kwargs)


def test_release25_default_coordinates_are_two_dimensional_uniform_offsets():
    field_map = PerlinCoordinatePermeabilityMap(
        shape=(4, 5),
        domain_size_m=(40.0, 50.0),
        frequency=(2.0, 2.0),
    )

    offsets = field_map.coordinates_to_offsets(
        np.asarray(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )
    )

    np.testing.assert_allclose(offsets[0], [0.0, 0.0])
    np.testing.assert_allclose(
        offsets[1],
        [RELEASE25_PERLIN_OFFSET_SPAN / 2.0] * 2,
    )
    np.testing.assert_allclose(
        offsets[2],
        [RELEASE25_PERLIN_OFFSET_SPAN] * 2,
    )
    assert field_map.dimension == 2
    assert field_map.metadata["natural_pce_basis"] == "Legendre"


def test_deterministic_x_base_shift_is_separate_from_random_coordinates():
    field_map = _small_map(x_base_shift=7.0)

    offsets = field_map.coordinates_to_offsets(np.asarray([0.0, 0.0]))

    np.testing.assert_allclose(offsets, [9.0, 3.0])


def test_perlin_coordinate_map_matches_historical_formula_for_same_offset():
    field_map = _small_map(x_base_shift=0.5)
    coordinates = np.asarray([0.0, 0.0])
    offset = field_map.coordinates_to_offsets(coordinates)

    actual = field_map.map_coordinates(coordinates)
    expected = historical_perlin_v2_field(
        shape=field_map.shape,
        domain_size_m=field_map.domain_size_m,
        k_min=field_map.k_min,
        k_max=field_map.k_max,
        frequency=field_map.frequency,
        offset=offset,
    ).astype(np.float32)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == field_map.field_shape
    assert np.all(actual > 0.0)


def test_perlin_coordinate_map_supports_batched_inputs_and_protocol():
    field_map = _small_map()
    coordinates = np.asarray(
        [
            [-0.5, 0.2],
            [0.1, -0.7],
            [0.8, 0.6],
        ]
    )

    fields = field_map.map_coordinates(coordinates)

    assert fields.shape == (3, 12, 13)
    assert np.all(np.isfinite(fields))
    assert np.all(fields > 0.0)
    assert isinstance(field_map, StochasticPermeabilityMap)


def test_uniform_coordinate_sampler_is_reproducible_and_protocol_compatible():
    field_map = _small_map()
    sampler = UniformCoordinatePermeabilitySampler(
        field_map=field_map,
        n_samples=5,
        batch_size=2,
        seed=31415,
    )

    first = np.concatenate(list(sampler), axis=0)
    second = np.concatenate(list(sampler), axis=0)

    assert isinstance(sampler, PermeabilitySampler)
    assert first.shape == (5, 12, 13)
    assert sampler.metadata["coordinate_distribution"] == "iid_uniform_minus1_1"
    np.testing.assert_array_equal(first, second)


def test_perlin_coordinates_outside_standard_domain_are_rejected():
    field_map = _small_map()

    try:
        field_map.map_coordinates(np.asarray([1.1, 0.0]))
    except ValueError as exc:
        assert "must lie in [-1, 1]" in str(exc)
    else:
        raise AssertionError("coordinates outside [-1,1] should raise ValueError")


def test_perlin_coordinate_dimension_mismatch_is_rejected():
    field_map = _small_map()

    try:
        field_map.map_coordinates(np.zeros(3))
    except ValueError as exc:
        assert "expected stochastic dimension 2" in str(exc)
    else:
        raise AssertionError("dimension mismatch should raise ValueError")
