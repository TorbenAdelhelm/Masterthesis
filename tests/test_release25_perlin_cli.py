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


def test_perlin_cli_accepts_documented_module_invocation_arguments():
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
            "--n-samples",
            "5",
            "--seed",
            "2907",
            "--device",
            "cpu",
            "--output",
            "run_output/release25_perlin_mc_5_v3.npz",
            "--plots-dir",
            "run_output/release25_perlin_mc_5_v3_plots",
        ]
    )

    assert args.release25_repo == "external/release25-repo/Heat-Plume-Prediction"
    assert args.cnn1_dir == "models/LGCNN_step1_randomK"
    assert args.cnn2_dir == "models/LGCNN_step3_randomK"
    assert args.prepared_pki_dir == "data/prepared_pki"
    assert args.fixed_run_id == "RUN_1"
    assert args.n_samples == 5
    assert args.seed == 2907
    assert args.device == "cpu"
    assert args.output == "run_output/release25_perlin_mc_5_v3.npz"
    assert args.plots_dir == "run_output/release25_perlin_mc_5_v3_plots"
