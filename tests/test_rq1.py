from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from subsurface_uq.propagation import MonteCarloRunner
from subsurface_uq.rq1.accumulators import (
    DiskTemperatureStoreAccumulator,
    RQ1QoIAccumulator,
    ReferenceConvergenceFieldStatisticsAccumulator,
)
from subsurface_uq.rq1.analysis import (
    global_uncertainty_metrics,
    prefix_field_convergence,
    scalar_summary,
    write_temperature_field_products,
)
from subsurface_uq.rq1.config import load_rq1_config
from subsurface_uq.sampling import (
    EmpiricalPermeabilitySampler,
    PermeabilityDiagnostics,
    ScrambledSobolGaussianPermeabilitySampler,
)
from subsurface_uq.surrogates import CallableTemperatureSurrogate


@dataclass
class _GaussianToyMap:
    dimension: int = 2
    field_shape: tuple[int, int] = (2, 3)

    def map_coordinates(self, coordinates):
        xi = np.asarray(coordinates, dtype=np.float64)
        if xi.ndim == 1:
            xi = xi[None, :]
        base = -9.5 + 0.1 * xi[:, 0, None, None] + 0.05 * xi[:, 1, None, None]
        pattern = np.asarray([[0.0, 0.02, 0.04], [0.01, 0.03, 0.05]])
        return np.power(10.0, base + pattern[None, ...]).astype(np.float32)


def test_scrambled_sobol_gaussian_sampler_is_reproducible():
    field_map = _GaussianToyMap()
    a = np.concatenate(
        list(
            ScrambledSobolGaussianPermeabilitySampler(
                field_map=field_map, n_samples=8, batch_size=3, seed=17
            )
        ),
        axis=0,
    )
    b = np.concatenate(
        list(
            ScrambledSobolGaussianPermeabilitySampler(
                field_map=field_map, n_samples=8, batch_size=5, seed=17
            )
        ),
        axis=0,
    )
    c = np.concatenate(
        list(
            ScrambledSobolGaussianPermeabilitySampler(
                field_map=field_map, n_samples=8, batch_size=4, seed=18
            )
        ),
        axis=0,
    )
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.all(a > 0.0)


def test_scrambled_sobol_requires_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        ScrambledSobolGaussianPermeabilitySampler(
            field_map=_GaussianToyMap(), n_samples=6
        )


def test_rq1_permeability_diagnostics_include_range_and_conditioning():
    fields = np.asarray(
        [
            [[1.0e-10, 2.0e-10], [3.0e-10, 4.0e-10]],
            [[1.0e-12, 2.0e-10], [3.0e-10, 6.0e-9]],
        ],
        dtype=np.float64,
    )
    observations = np.asarray([[0, 1], [1, 0]])
    observed_log = np.log10(np.asarray([2.0e-10, 3.0e-10]))
    diagnostics = PermeabilityDiagnostics(
        preview_count=0,
        ddof=1,
        training_k_range=(1.0e-11, 5.0e-9),
        observation_indices=observations,
        observation_log10_k=observed_log,
    )
    diagnostics.update(fields)
    result = diagnostics.finalize()

    assert result.minimum_k == pytest.approx(1.0e-12)
    assert result.maximum_k == pytest.approx(6.0e-9)
    assert result.outside_training_fraction == pytest.approx(2 / 8)
    assert result.outside_training_fraction_by_sample == pytest.approx((0.0, 0.5))
    assert result.conditioning_max_abs_log10_residual == pytest.approx(0.0, abs=1e-12)


def test_disk_store_exact_quantiles_and_prefix_convergence(tmp_path):
    temperatures = np.asarray(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[11.0, 12.0], [13.0, 14.0]],
            [[12.0, 13.0], [14.0, 15.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        dtype=np.float32,
    )
    store_acc = DiskTemperatureStoreAccumulator(
        tmp_path / "samples.npy", expected_samples=4
    )
    store_acc.update(temperatures[:3])
    store_acc.update(temperatures[3:])
    store = store_acc.finalize()

    fields = write_temperature_field_products(
        temperature_store=store.path,
        final_mean=np.mean(temperatures, axis=0),
        final_std=np.std(temperatures, axis=0, ddof=1),
        output_directory=tmp_path / "fields",
        chunk_rows=1,
    )
    np.testing.assert_allclose(
        np.load(fields["temperature_q05"]),
        np.quantile(temperatures, 0.05, axis=0),
    )
    np.testing.assert_allclose(
        np.load(fields["temperature_width90"]),
        np.quantile(temperatures, 0.95, axis=0)
        - np.quantile(temperatures, 0.05, axis=0),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    points = prefix_field_convergence(
        temperature_store=store.path,
        checkpoints=(2, 4),
        reference_mean=np.mean(temperatures, axis=0),
        reference_std=np.std(temperatures, axis=0, ddof=1),
        chunk_rows=1,
    )
    assert points[-1]["e_mu"] == pytest.approx(0.0)
    assert points[-1]["e_sigma"] == pytest.approx(0.0)


def test_qoi_and_reference_convergence_accumulators_reuse_monte_carlo_runner():
    fields = np.stack(
        [
            np.full((2, 2), 1.0e-10, dtype=np.float32),
            np.full((2, 2), 2.0e-10, dtype=np.float32),
            np.full((2, 2), 3.0e-10, dtype=np.float32),
            np.full((2, 2), 4.0e-10, dtype=np.float32),
        ]
    )
    surrogate = CallableTemperatureSurrogate(lambda k: 10.0 + 1.0e9 * k)
    direct = np.stack([surrogate.predict_temperature(field) for field in fields])
    reference_mean = np.mean(direct, axis=0)
    reference_std = np.std(direct, axis=0, ddof=1)

    field_acc = ReferenceConvergenceFieldStatisticsAccumulator(
        checkpoints=(2, 4),
        reference_mean=reference_mean,
        reference_std=reference_std,
    )
    qoi_acc = RQ1QoIAccumulator(
        background_temperature=10.0,
        receptors=((0, 0),),
        mean_anomaly_roi=(0, 2, 0, 2),
    )
    result = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=3),
        surrogate,
    ).run(
        accumulators=(qoi_acc,),
        field_statistics_accumulator=field_acc,
    )

    assert result.count == 4
    assert [point.budget for point in field_acc.points] == [2, 4]
    assert field_acc.points[-1].e_mu == pytest.approx(0.0, abs=1e-7)
    qoi = result.accumulator_results["rq1_qoi"]
    np.testing.assert_allclose(qoi.mean_anomaly, direct.mean(axis=(1, 2)) - 10.0)
    np.testing.assert_allclose(qoi.receptor_values[:, 0], direct[:, 0, 0])


def test_qoi_accumulator_allows_receptors_without_mean_anomaly_roi():
    accumulator = RQ1QoIAccumulator(
        background_temperature=10.0,
        receptors=((0, 1),),
        mean_anomaly_roi=None,
    )
    temperatures = np.asarray(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[14.0, 15.0], [16.0, 17.0]],
        ],
        dtype=np.float32,
    )
    accumulator.update(temperatures)
    samples = accumulator.finalize()

    assert samples.mean_anomaly is None
    np.testing.assert_allclose(samples.receptor_values[:, 0], [11.0, 15.0])


def test_scalar_and_global_metric_definitions():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    summary = scalar_summary(values)
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["sd"] == pytest.approx(np.std(values, ddof=1))

    std = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    metrics = global_uncertainty_metrics(std)
    assert metrics["U_RMS"] == pytest.approx(np.sqrt(np.mean(std**2)))
    assert metrics["U95"] == pytest.approx(np.quantile(std, 0.95))


def test_rq1_config_allows_null_mean_anomaly_roi(tmp_path):
    config_path = tmp_path / "rq1_no_roi.yaml"
    config_path.write_text(
        f"""
experiment:
  id: no_roi
  output_root: {tmp_path / 'out_no_roi'}
release25:
  repo: release25
  cnn1_dir: cnn1
  cnn2_dir: cnn3
  prepared_pki_dir: prepared
  fixed_run_id: RUN_1
sampling:
  budgets: [2, 4]
  comparison_budgets: [2]
  repetitions: 1
  main_seed: 10
  repetition_seeds: [20]
qoi:
  receptors: [[0, 0]]
  mean_anomaly_roi: null
grf:
  mean_log10_k: -9.5
  std_log10_k: 0.3
  length_scale_y_m: 100
  length_scale_x_m: 150
  n_modes: 4
  energy_threshold: null
  observations:
    - [0, 0, 2.0e-10]
""",
        encoding="utf-8",
    )
    config = load_rq1_config(config_path)
    assert config.mean_anomaly_roi is None


def test_rq1_config_and_cli(tmp_path):
    config_path = tmp_path / "rq1.yaml"
    config_path.write_text(
        f"""
experiment:
  id: test
  output_root: {tmp_path / 'out'}
release25:
  repo: release25
  cnn1_dir: cnn1
  cnn2_dir: cnn3
  prepared_pki_dir: prepared
  fixed_run_id: RUN_1
sampling:
  budgets: [2, 4, 8]
  comparison_budgets: [2, 4]
  repetitions: 2
  main_seed: 10
  repetition_seeds: [20, 30]
  quantiles: [0.05, 0.5, 0.95]
qoi:
  receptors: [[0, 0]]
  mean_anomaly_roi: [0, 2, 0, 2]
grf:
  mean_log10_k: -9.5
  std_log10_k: 0.3
  length_scale_y_m: 100
  length_scale_x_m: 150
  n_modes: 4
  energy_threshold: null
  observations:
    - [0, 0, 2.0e-10]
streamlines:
  mode: bounded
""",
        encoding="utf-8",
    )
    config = load_rq1_config(config_path)
    assert config.budgets == (2, 4, 8)
    assert config.comparison_budgets == (2, 4)
    assert config.repetition_seeds == (20, 30)

    completed = subprocess.run(
        [sys.executable, "-m", "subsurface_uq.experiments.release25_rq1", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--config" in completed.stdout
