import json
import h5py
import numpy as np

from subsurface_uq.experiments.realistic_permeability import build_parser
from subsurface_uq.sampling import (
    RadialExponentialPermeabilitySampler,
    calibrate_covariance_candidates,
    correlation_for_offsets,
    exact_simple_kriging_posterior,
    load_release25_raw_permeability_dataset,
    sample_borehole_observations,
)


def test_release25_raw_h5_loader_matches_dataset_layout(tmp_path):
    root = tmp_path / "real_raw"
    root.mkdir()
    (root / "settings.yaml").write_text(
        "grid:\n  size [m]: [20, 15]\n",
        encoding="utf-8",
    )
    expected = []
    for run_number in (2, 1):
        run = root / f"RUN_{run_number}"
        run.mkdir()
        values = (
            np.arange(12, dtype=np.float64).reshape(4, 3, 1)
            + 1.0
            + run_number * 100.0
        ) * 1.0e-12
        expected.append((run_number, values.squeeze()))
        with h5py.File(run / "pflotran.h5", "w") as handle:
            group = handle.create_group("   0 Time  0.00000E+00 y")
            group.create_dataset("Permeability X [m^2]", data=values.reshape(-1))

    fields, runs = load_release25_raw_permeability_dataset(root, cell_size_m=5.0)

    assert runs == ("RUN_1", "RUN_2")
    assert fields.shape == (2, 4, 3)
    by_run = {number: field for number, field in expected}
    np.testing.assert_allclose(fields[0], by_run[1], rtol=1e-7)
    np.testing.assert_allclose(fields[1], by_run[2], rtol=1e-7)


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


def test_exact_simple_kriging_matches_dense_manual_posterior():
    shape = (4, 5)
    cell_size = 10.0
    mean = -9.4
    std = 0.3
    ly = 25.0
    lx = 40.0
    indices = np.asarray([[0, 1], [3, 4]], dtype=np.int64)
    values = np.asarray([-9.8, -9.1], dtype=np.float64)

    posterior = exact_simple_kriging_posterior(
        shape=shape,
        cell_size_m=cell_size,
        covariance_model="radial_exponential",
        mean_log10_k=mean,
        std_log10_k=std,
        length_scale_y_m=ly,
        length_scale_x_m=lx,
        observation_indices=indices,
        observation_log10_k=values,
        observation_std_log10_k=0.0,
        chunk_rows=2,
    )

    yy, xx = np.meshgrid(
        (np.arange(shape[0]) + 0.5) * cell_size,
        (np.arange(shape[1]) + 0.5) * cell_size,
        indexing="ij",
    )
    positions = np.column_stack((yy.ravel(), xx.ravel()))
    obs_positions = np.column_stack(
        (
            (indices[:, 0] + 0.5) * cell_size,
            (indices[:, 1] + 0.5) * cell_size,
        )
    )
    dy_dd = obs_positions[:, 0, None] - obs_positions[None, :, 0]
    dx_dd = obs_positions[:, 1, None] - obs_positions[None, :, 1]
    kdd = std**2 * correlation_for_offsets(
        "radial_exponential",
        delta_y_m=dy_dd,
        delta_x_m=dx_dd,
        length_scale_y_m=ly,
        length_scale_x_m=lx,
    )
    dy_xd = positions[:, 0, None] - obs_positions[None, :, 0]
    dx_xd = positions[:, 1, None] - obs_positions[None, :, 1]
    kxd = std**2 * correlation_for_offsets(
        "radial_exponential",
        delta_y_m=dy_xd,
        delta_x_m=dx_xd,
        length_scale_y_m=ly,
        length_scale_x_m=lx,
    )
    inverse = np.linalg.inv(kdd)
    expected_mean = mean + kxd @ inverse @ (values - mean)
    expected_variance = std**2 - np.sum((kxd @ inverse) * kxd, axis=1)

    np.testing.assert_allclose(
        posterior.mean_log10_k.ravel(), expected_mean, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        posterior.variance_log10_k.ravel(), expected_variance, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        posterior.mean_log10_k[indices[:, 0], indices[:, 1]],
        values,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        posterior.std_log10_k[indices[:, 0], indices[:, 1]],
        0.0,
        rtol=0.0,
        atol=1e-5,
    )


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


def test_leave_one_out_evaluation_writes_comparison_outputs(tmp_path):
    y, x = np.mgrid[0:12, 0:14]
    fields = np.stack(
        [
            10.0 ** (-9.5 + 0.10 * np.sin((x + shift) / 3.0) + 0.07 * np.cos(y / 2.5))
            for shift in (0.0, 0.8, 1.6)
        ]
    ).astype(np.float32)
    source = tmp_path / "fields.npy"
    np.save(source, fields)
    output_dir = tmp_path / "evaluation"

    args = build_parser().parse_args(
        [
            "evaluate",
            "--fields",
            str(source),
            "--cell-size-m",
            "5",
            "--truth-index",
            "0",
            "--max-lag-cells",
            "4",
            "--spatial-stride",
            "1",
            "--evaluation-stride",
            "1",
            "--n-boreholes",
            "2",
            "--margin-cells",
            "1",
            "--min-spacing-cells",
            "2",
            "--n-samples",
            "3",
            "--n-modes",
            "20",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0

    assert (output_dir / "calibration_leave_one_out.yaml").is_file()
    assert (output_dir / "model_comparison.csv").is_file()
    assert (output_dir / "model_comparison.json").is_file()
    comparison = json.loads(
        (output_dir / "model_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["posterior_evaluation"]["method"] == "exact_full_covariance_simple_kriging"
    assert comparison["posterior_evaluation"]["kl_truncation"] is None
    assert comparison["posterior_evaluation"]["monte_carlo_sampling"] is None
    assert (output_dir / "figures" / "variogram_fits.png").is_file()
    assert (output_dir / "figures" / "length_scales.png").is_file()
    assert (output_dir / "figures" / "heldout_metric_comparison.png").is_file()
    for model in ("matern32", "exponential", "radial_exponential"):
        assert (output_dir / "figures" / f"heldout_{model}.png").is_file()
        assert (
            output_dir / "arrays" / model / "posterior_mean_log10_k.npy"
        ).is_file()
