from subsurface_uq.experiments.release25_perlin_pce import build_parser
from subsurface_uq.sampling import (
    RELEASE25_PERLIN_DEFAULT_SEED,
    RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
)


def test_perlin_pce_cli_uses_reproducible_scientific_defaults():
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
            "--n-train",
            "60",
            "--n-validation",
            "100",
        ]
    )

    assert args.degree == 4
    assert args.train_seed == RELEASE25_PERLIN_DEFAULT_SEED
    assert args.validation_seed == RELEASE25_PERLIN_DEFAULT_SEED + 1
    assert args.background_temperature == RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C
    assert args.x_base_shift == 0.0
