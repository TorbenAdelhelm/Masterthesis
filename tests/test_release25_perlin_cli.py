from subsurface_uq.experiments.release25_perlin import build_parser
from subsurface_uq.sampling import RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C


def test_perlin_cli_uses_exact_synthetic_background_and_auto_ddof():
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
            "--n-samples",
            "5",
        ]
    )

    assert args.background_temperature == RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C
    assert args.exceedance_thresholds == (0.1, 1.0)
    assert args.ddof is None
