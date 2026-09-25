import numpy as np

from subsurface_uq.experiments.realistic_permeability import build_parser
from subsurface_uq.sampling import (
    RadialExponentialPermeabilitySampler,
    calibrate_covariance_candidates,
    correlation_for_offsets,
    sample_borehole_observations,
)


def test_radial_exponential_correlation_differs_from_separable_diagonal():
    kwargs = dict(
        delta_y_m=np.asarray([100.0]),
        delta_x_m=np.asarray([100.0]),
        length_scale_y_m=200.0,
        length_scale_x_m=400.0,
    )
    separable = correlation_for_offsets("exponential", **kwargs)
    radial = correlation_for_offsets("radial_exponential", **kwargs)

    assert radial.shape == (1,)
    assert radial[0] > separable[0]


def test_synthetic_boreholes_are_reproducible_and_respect_spacing():
    field = np.exp(np.arange(20 * 24, dtype=np.float64).reshape(20, 24) / 1000.0)
    first = sample_borehole_observations(
        field, n_boreholes=8, seed=123, margin_cells=2, min_spacing_cells=3.0
    )
    second = sample_borehole_observations(
        field, n_boreholes=8, seed=123, margin_cells=2, min_spacing_cells=3.0
    )

    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.permeability, second.permeability)
    assert np.all(first.indices[:, 0] >= 2)
    assert np.all(first.indices[:, 0] < 18)
    assert np.all(first.indices[:, 1] >= 2)
    assert np.all(first.indices[:, 1] < 22)
    for i in range(first.count):
        for j in range(i):
            distance = np.linalg.norm(first.indices[i] - first.indices[j])
            assert distance >= 3.0


def test_covariance_calibration_returns_all_candidates():
    y, x = np.mgrid[0:18, 0:20]
    fields = np.stack(
        [
            10.0 ** (-9.5 + 0.15 * np.sin(x / 4.0) + 0.08 * np.cos(y / 3.0)),
            10.0 ** (-9.45 + 0.12 * np.sin((x + 1) / 4.0) + 0.09 * np.cos(y / 3.0)),
        ]
    )
    results = calibrate_covariance_candidates(
        fields, cell_size_m=5.0, max_lag_cells=6, spatial_stride=1
    )

    assert {item.covariance_model for item in results} == {
        "matern32",
        "exponential",
        "radial_exponential",
    }
    assert all(item.length_scale_y_m > 0.0 for item in results)
    assert all(item.length_scale_x_m > 0.0 for item in results)
    assert all(np.isfinite(item.variogram_rmse) for item in results)
    assert list(item.variogram_rmse for item in results) == sorted(
        item.variogram_rmse for item in results
    )


def test_radial_exponential_sampler_is_reproducible_and_conditions_exactly():
    observation_indices = np.asarray([[2, 3], [6, 7]], dtype=np.int64)
    observation_k = np.asarray([2.5e-10, 7.5e-10], dtype=np.float64)
    sampler = RadialExponentialPermeabilitySampler(
        shape=(9, 11),
        cell_size_m=5.0,
        mean_log10_k=-9.5,
        std_log10_k=0.25,
        length_scale_y_m=18.0,
        length_scale_x_m=30.0,
        angle_rad=0.2,
        n_samples=3,
        batch_size=2,
        seed=77,
        observation_indices=observation_indices,
        observation_k=observation_k,
    )

    first = np.concatenate(list(sampler), axis=0)
    second = np.concatenate(list(sampler), axis=0)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 9, 11)
    for row, col, expected in zip(
        observation_indices[:, 0], observation_indices[:, 1], observation_k
    ):
        np.testing.assert_allclose(first[:, row, col], expected, rtol=2e-4, atol=0.0)


def test_realistic_permeability_cli_has_calibrate_and_generate_subcommands():
    parser = build_parser()
    calibrate = parser.parse_args(
        [
            "calibrate",
            "--fields",
            "fields.npy",
            "--cell-size-m",
            "5",
            "--output",
            "calibration.yaml",
        ]
    )
    generate = parser.parse_args(
        [
            "generate",
            "--fields",
            "fields.npy",
            "--calibration",
            "calibration.yaml",
            "--n-boreholes",
            "5",
            "--output",
            "ensemble.npz",
        ]
    )

    assert calibrate.command == "calibrate"
    assert generate.command == "generate"
