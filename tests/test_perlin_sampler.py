from __future__ import annotations

import noise
import numpy as np

from subsurface_uq.sampling import (
    RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
    Release25PerlinPermeabilitySampler,
)


def _historical_reference(
    *,
    shape: tuple[int, int],
    domain_size_m: tuple[float, float],
    k_min: float,
    k_max: float,
    frequency: tuple[float, float],
    offset: tuple[float, float, float],
) -> np.ndarray:
    """Independent transcription of the historical 2-D perlin_v2 branch."""

    h, w = shape
    simulation_area_max = max(domain_size_m)
    scale_x = domain_size_m[0] / simulation_area_max
    scale_y = domain_size_m[1] / simulation_area_max

    values = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            x = (i / h * scale_x + offset[0]) * frequency[0]
            y = (j / w * scale_y + offset[1]) * frequency[1]
            values[i, j] = noise.pnoise2(x, y)

    current_min = np.min(values)
    current_max = np.max(values)
    values = (values - current_min) / (current_max - current_min)

    aimed_min = np.log10(k_min)
    aimed_max = np.log10(k_max)
    values = values * (aimed_max - aimed_min) + aimed_min
    return 10**values


def test_sampler_matches_historical_perlin_v2_formula() -> None:
    shape = (9, 7)
    domain_size_m = (90.0, 70.0)
    frequency = (2.5, 3.0)
    k_min = 1.0193679918450561e-11
    k_max = 5.09683995922528e-09
    base_offset = (12.25, 4.5, 99.0)

    sampler = Release25PerlinPermeabilitySampler(
        n_samples=1,
        batch_size=1,
        seed=2907,
        shape=shape,
        domain_size_m=domain_size_m,
        frequency=frequency,
        k_min=k_min,
        k_max=k_max,
        base_offset=base_offset,
    )
    actual = next(iter(sampler))[0]
    expected = _historical_reference(
        shape=shape,
        domain_size_m=domain_size_m,
        k_min=k_min,
        k_max=k_max,
        frequency=frequency,
        offset=base_offset,
    )

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=1e-6, atol=0.0)
    assert np.isclose(actual.min(), k_min, rtol=1e-6)
    assert np.isclose(actual.max(), k_max, rtol=1e-6)


def test_sampler_is_reproducible_and_batches_without_changing_order() -> None:
    kwargs = dict(
        n_samples=3,
        seed=2907,
        shape=(8, 6),
        domain_size_m=(80.0, 60.0),
        frequency=(2.0, 3.0),
        k_min=1.0e-11,
        k_max=5.0e-9,
    )
    one_by_one = Release25PerlinPermeabilitySampler(batch_size=1, **kwargs)
    batched = Release25PerlinPermeabilitySampler(batch_size=2, **kwargs)

    fields_one = np.concatenate(list(one_by_one), axis=0)
    fields_batched = np.concatenate(list(batched), axis=0)

    np.testing.assert_array_equal(fields_one, fields_batched)
    assert len(one_by_one) == 3
    assert one_by_one.field_shape == (8, 6)


def test_sampler_metadata_records_required_reproducibility_parameters() -> None:
    sampler = Release25PerlinPermeabilitySampler(
        n_samples=4,
        batch_size=2,
        seed=123,
        shape=(5, 4),
        domain_size_m=(25.0, 20.0),
        frequency=(7.0, 9.0),
        k_min=2.0e-11,
        k_max=4.0e-9,
    )

    metadata = sampler.metadata
    assert metadata["seed"] == 123
    assert metadata["frequency"] == [7.0, 9.0]
    assert metadata["k_min"] == 2.0e-11
    assert metadata["k_max"] == 4.0e-9
    assert metadata["sample_count"] == 4
    assert len(metadata["base_offset"]) == 3


def test_release25_synthetic_background_temperature_is_exact_dataset_value() -> None:
    assert RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C == 10.6
