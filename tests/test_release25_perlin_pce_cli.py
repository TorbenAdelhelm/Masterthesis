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
    assert args.streamline_mode == "release25"
    assert args.streamline_max_nfev == 100_000
    assert args.streamline_diagnostics is False
    assert args.streamline_slow_seconds == 2.0


def test_perlin_pce_cli_accepts_documented_module_invocation_arguments():
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
            "--degree",
            "4",
            "--n-train",
            "60",
            "--n-validation",
            "100",
            "--train-seed",
            "2907",
            "--validation-seed",
            "2908",
            "--device",
            "cpu",
            "--streamline-mode",
            "bounded",
            "--streamline-max-nfev",
            "50000",
            "--streamline-diagnostics",
            "--streamline-slow-seconds",
            "1.5",
            "--output",
            "run_output/release25_perlin_pce_d4_n60.npz",
        ]
    )

    assert args.release25_repo == "external/release25-repo/Heat-Plume-Prediction"
    assert args.cnn1_dir == "models/LGCNN_step1_randomK"
    assert args.cnn2_dir == "models/LGCNN_step3_randomK"
    assert args.prepared_pki_dir == "data/prepared_pki"
    assert args.fixed_run_id == "RUN_1"
    assert args.degree == 4
    assert args.n_train == 60
    assert args.n_validation == 100
    assert args.train_seed == 2907
    assert args.validation_seed == 2908
    assert args.device == "cpu"
    assert args.streamline_mode == "bounded"
    assert args.streamline_max_nfev == 50_000
    assert args.streamline_diagnostics is True
    assert args.streamline_slow_seconds == 1.5
    assert args.output == "run_output/release25_perlin_pce_d4_n60.npz"
