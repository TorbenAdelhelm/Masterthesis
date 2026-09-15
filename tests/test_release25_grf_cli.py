import numpy as np
import pytest

from subsurface_uq.experiments.release25_grf_mc import build_grf_map, build_parser
from subsurface_uq.sampling import (
    ConditionalKLLogGaussianPermeabilityMap,
    KLLogGaussianPermeabilityMap,
)


def _base_args(*extra: str):
    return build_parser().parse_args(
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
            "--mean-log10-k",
            "-9.5",
            "--std-log10-k",
            "0.3",
            "--length-scale-y-m",
            "120",
            "--length-scale-x-m",
            "180",
            "--n-modes",
            "6",
            "--n-samples",
            "4",
            *extra,
        ]
    )


def test_grf_cli_uses_bounded_streamlines_and_explicit_grf_parameters():
    args = _base_args()

    assert args.streamline_mode == "bounded"
    assert args.n_modes == 6
    assert args.energy_threshold is None
    assert args.mean_log10_k == -9.5
    assert args.std_log10_k == 0.3
    assert args.length_scale_y_m == 120.0
    assert args.length_scale_x_m == 180.0
    assert args.observation is None
    assert args.observation_std_log10_k == 0.0


def test_build_grf_map_supports_unconditional_and_conditioned_modes():
    prior_args = _base_args()
    prior = build_grf_map(prior_args, shape=(8, 9))

    assert isinstance(prior, KLLogGaussianPermeabilityMap)
    assert prior.dimension == 6
    assert prior.field_shape == (8, 9)

    conditioned_args = _base_args(
        "--observation",
        "2",
        "3",
        str(10.0**-9.5),
        "--observation",
        "5",
        "7",
        str(10.0**-9.5),
    )
    conditioned = build_grf_map(conditioned_args, shape=(8, 9))

    assert isinstance(conditioned, ConditionalKLLogGaussianPermeabilityMap)
    assert conditioned.field_shape == (8, 9)
    assert conditioned.dimension < prior.dimension
    sample = conditioned.map_coordinates(np.zeros(conditioned.dimension))
    np.testing.assert_allclose(sample[2, 3], 10.0**-9.5, rtol=2e-6)
    np.testing.assert_allclose(sample[5, 7], 10.0**-9.5, rtol=2e-6)


def test_observation_uncertainty_without_observation_is_rejected():
    args = _base_args("--observation-std-log10-k", "0.05")

    with pytest.raises(ValueError, match="requires at least one --observation"):
        build_grf_map(args, shape=(5, 5))
