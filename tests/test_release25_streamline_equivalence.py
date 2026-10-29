import numpy as np
import pytest

from subsurface_uq.experiments.release25_streamline_equivalence import (
    build_parser,
    difference_metrics,
    select_lhs_coordinates,
)
from subsurface_uq.sampling import RELEASE25_PERLIN_DEFAULT_SEED


def test_difference_metrics_zero_for_identical_arrays():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    metrics = difference_metrics(values, values.copy())

    assert metrics == {
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rmse": 0.0,
        "relative_l2": 0.0,
    }


def test_difference_metrics_reports_known_error():
    reference = np.array([1.0, 2.0])
    candidate = np.array([2.0, 2.0])

    metrics = difference_metrics(reference, candidate)

    assert metrics["max_abs"] == pytest.approx(1.0)
    assert metrics["mean_abs"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(0.5))
    assert metrics["relative_l2"] == pytest.approx(1.0 / np.sqrt(5.0))


def test_select_lhs_coordinates_is_reproducible_and_one_based():
    selected = select_lhs_coordinates(
        seed=2907,
        design_size=4,
        sample_indices=(1, 2, 3),
    )
    full = select_lhs_coordinates(
        seed=2907,
        design_size=4,
        sample_indices=(1, 2, 3, 4),
    )

    np.testing.assert_allclose(selected, full[:3])
    assert selected.shape == (3, 2)
    assert np.all(selected >= -1.0)
    assert np.all(selected <= 1.0)


def test_select_lhs_coordinates_rejects_invalid_indices():
    with pytest.raises(ValueError, match="must lie"):
        select_lhs_coordinates(seed=1, design_size=4, sample_indices=(0,))
    with pytest.raises(ValueError, match="unique"):
        select_lhs_coordinates(seed=1, design_size=4, sample_indices=(1, 1))


def test_equivalence_cli_defaults_to_known_non_pathological_smoke_samples():
    args = build_parser().parse_args(
        [
            "--release25-repo",
            "release25",
            "--cnn1-dir",
            "cnn1",
            "--cnn2-dir",
            "cnn3",
            "--prepared-pki-dir",
            "prepared",
            "--fixed-run-id",
            "RUN_1",
        ]
    )

    assert args.design_seed == RELEASE25_PERLIN_DEFAULT_SEED
    assert args.design_size == 4
    assert tuple(args.sample_indices) == (1, 2, 3)
    assert args.streamline_method == "RK45"
    assert args.streamline_max_nfev == 100_000
    assert args.streamline_diagnostics is False


def test_equivalence_cli_accepts_documented_arguments():
    args = build_parser().parse_args(
        [
            "--release25-repo",
            "external/release25-repo/Heat-Plume-Prediction",
            "--cnn1-dir",
            "models/LGCNN_step1_randomK",
            "--cnn2-dir",
            "models/LGCNN_step3_randomK",
            "--prepared-pki-dir",
            "data/prepared_pki",
            "--fixed-run-id",
            "RUN_1",
            "--design-seed",
            "2907",
            "--design-size",
            "4",
            "--sample-indices",
            "1",
            "2",
            "3",
            "--device",
            "cpu",
            "--streamline-max-nfev",
            "100000",
            "--streamline-diagnostics",
            "--output",
            "run_output/release25_streamline_equivalence.npz",
        ]
    )

    assert args.release25_repo == "external/release25-repo/Heat-Plume-Prediction"
    assert args.fixed_run_id == "RUN_1"
    assert args.design_seed == 2907
    assert args.design_size == 4
    assert args.sample_indices == [1, 2, 3]
    assert args.device == "cpu"
    assert args.streamline_max_nfev == 100000
    assert args.streamline_diagnostics is True
    assert args.output == "run_output/release25_streamline_equivalence.npz"
