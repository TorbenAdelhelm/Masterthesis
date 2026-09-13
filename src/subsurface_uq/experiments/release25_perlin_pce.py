from __future__ import annotations

import argparse
from pathlib import Path

from ..io import git_head, runtime_versions, sha256_file
from ..pce import (
    CoordinateQoIEvaluator,
    MeanTemperatureAnomaly,
    run_uniform_pce_proof_of_concept,
    save_pce_proof_of_concept_result,
)
from ..sampling import (
    RELEASE25_PERLIN_DEFAULT_SEED,
    RELEASE25_PERLIN_FREQUENCY,
    RELEASE25_PERLIN_K_MAX,
    RELEASE25_PERLIN_K_MIN,
    RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
    PerlinCoordinatePermeabilityMap,
)
from ..surrogates import Release25Surrogate
from ..surrogates.bounded_streamlines import configure_release25_streamlines
from ..surrogates.release25_runtime import Release25Runtime
from ..visualization.pce import plot_pce_archive

HISTORICAL_GENERATOR_COMMIT = "8549bbd9e22d2bc75ce2038c1a0397359e45c971"
HISTORICAL_GENERATOR_PATH = "scripts/create_varying_field.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit and validate a low-dimensional Legendre PCE for a continuous "
            "temperature QoI of the pretrained release25 LGCNN using the explicit "
            "Perlin stochastic-coordinate map."
        )
    )
    parser.add_argument("--release25-repo", required=True)
    parser.add_argument("--cnn1-dir", required=True)
    parser.add_argument(
        "--cnn2-dir",
        required=True,
        help=(
            "Legacy CLI name for the second CNN model folder, i.e. LGCNN Step 3/CNN3. "
            "Retained for backwards compatibility."
        ),
    )
    parser.add_argument("--prepared-pki-dir", required=True)
    parser.add_argument(
        "--fixed-run-id",
        required=True,
        help="Prepared pki datapoint supplying fixed pressure and heat-pump positions.",
    )
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument(
        "--n-train",
        type=int,
        required=True,
        help="Number of expensive LGCNN evaluations used to fit the PCE.",
    )
    parser.add_argument(
        "--n-validation",
        type=int,
        required=True,
        help=(
            "Independent iid Perlin-coordinate LGCNN evaluations used as the scalar "
            "Monte Carlo validation/reference ensemble."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=RELEASE25_PERLIN_DEFAULT_SEED)
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=RELEASE25_PERLIN_DEFAULT_SEED + 1,
    )
    parser.add_argument(
        "--frequency",
        type=float,
        nargs=2,
        metavar=("FX", "FY"),
        default=RELEASE25_PERLIN_FREQUENCY,
    )
    parser.add_argument("--k-min", type=float, default=RELEASE25_PERLIN_K_MIN)
    parser.add_argument("--k-max", type=float, default=RELEASE25_PERLIN_K_MAX)
    parser.add_argument(
        "--x-base-shift",
        type=float,
        default=0.0,
        help=(
            "Optional deterministic historical integer-style x shift. It is not a "
            "stochastic PCE coordinate."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--random-k", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--streamline-method", choices=("RK45", "RK23", "Radau"), default="RK45"
    )
    parser.add_argument(
        "--streamline-mode",
        choices=("release25", "bounded"),
        default="release25",
        help=(
            "Streamline implementation. 'release25' preserves the published behavior; "
            "'bounded' terminates trajectories at the valid grid boundary and adds "
            "a watchdog for pathological adaptive ODE steps."
        ),
    )
    parser.add_argument(
        "--streamline-max-nfev",
        type=int,
        default=100_000,
        help=(
            "Maximum RHS evaluations per streamline in bounded mode. Use 0 to disable "
            "the watchdog. Default: 100000."
        ),
    )
    parser.add_argument(
        "--streamline-diagnostics",
        action="store_true",
        help="Print bounded-streamline timing, velocity ranges and slow-trajectory diagnostics.",
    )
    parser.add_argument(
        "--streamline-slow-seconds",
        type=float,
        default=2.0,
        help="Diagnostic threshold for reporting a slow individual streamline.",
    )
    parser.add_argument(
        "--background-temperature",
        type=float,
        default=RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
        help="Background temperature used for the mean Delta-T QoI.",
    )
    parser.add_argument(
        "--output",
        default="run_output/release25_perlin_pce.npz",
    )
    parser.add_argument(
        "--plots-dir",
        help=(
            "Optional directory for three compact PCE diagnostic plots: validation "
            "parity, validation residuals, and fitted coefficients."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    runtime = Release25Runtime.from_paths(
        release25_repo=args.release25_repo,
        cnn1_dir=args.cnn1_dir,
        cnn2_dir=args.cnn2_dir,
        prepared_pki_dir=args.prepared_pki_dir,
        run_id=args.fixed_run_id,
        device=args.device,
        random_k=args.random_k,
        streamline_method=args.streamline_method,
    )
    configure_release25_streamlines(
        runtime.adapter,
        mode=args.streamline_mode,
        max_nfev=args.streamline_max_nfev,
        diagnostics=args.streamline_diagnostics,
        slow_streamline_seconds=args.streamline_slow_seconds,
    )

    shape = tuple(int(value) for value in runtime.scenario.shape)
    domain_size_m = (shape[0] * 5.0, shape[1] * 5.0)
    field_map = PerlinCoordinatePermeabilityMap(
        shape=shape,
        domain_size_m=domain_size_m,
        frequency=(float(args.frequency[0]), float(args.frequency[1])),
        k_min=float(args.k_min),
        k_max=float(args.k_max),
        x_base_shift=float(args.x_base_shift),
    )

    surrogate = Release25Surrogate(
        adapter=runtime.adapter,
        fixed_inputs=runtime.scenario.fixed,
        device=args.device,
    )
    functional = MeanTemperatureAnomaly(
        background_temperature=float(args.background_temperature)
    )
    evaluator = CoordinateQoIEvaluator(
        field_map=field_map,
        surrogate=surrogate,
        functional=functional,
        batch_size=args.batch_size,
    )

    result = run_uniform_pce_proof_of_concept(
        evaluator,
        degree=args.degree,
        n_train=args.n_train,
        n_validation=args.n_validation,
        train_seed=args.train_seed,
        validation_seed=args.validation_seed,
    )

    project_root = Path(__file__).resolve().parents[3]
    metadata = {
        "uq_mode": "release25_perlin_legendre_pce_proof_of_concept",
        "field_map": field_map.metadata,
        "qoi": {
            "name": functional.name,
            "background_temperature_c": float(args.background_temperature),
        },
        "training_design": {
            "type": "randomized_latin_hypercube_uniform_minus1_1",
            "seed": int(args.train_seed),
        },
        "validation_design": {
            "type": "iid_uniform_minus1_1",
            "seed": int(args.validation_seed),
            "interpretation": (
                "Independent expensive LGCNN evaluations used as the scalar Monte "
                "Carlo reference under the same Perlin coordinate law."
            ),
        },
        "fixed_run_id": args.fixed_run_id,
        "prepared_pki_dir": str(Path(args.prepared_pki_dir)),
        "release25_repo": str(Path(args.release25_repo)),
        "cnn1_dir": str(Path(args.cnn1_dir)),
        "cnn2_dir": str(Path(args.cnn2_dir)),
        "cnn2_dir_role": "LGCNN Step 3 / CNN3 (legacy argument name)",
        "device": args.device,
        "streamline_method": args.streamline_method,
        "streamline_mode": args.streamline_mode,
        "streamline_max_nfev": int(args.streamline_max_nfev),
        "streamline_diagnostics": bool(args.streamline_diagnostics),
        "streamline_slow_seconds": float(args.streamline_slow_seconds),
        "random_k": bool(args.random_k),
        "runtime": runtime_versions(),
        "provenance": {
            "masterthesis_git_sha": git_head(project_root),
            "release25_git_sha": git_head(args.release25_repo),
            "cnn1_checkpoint_sha256": sha256_file(runtime.adapter.cnn1.checkpoint),
            "cnn3_checkpoint_sha256": sha256_file(runtime.adapter.cnn2.checkpoint),
            "reference_published_model_doi": "10.18419/DARUS-5080",
            "reference_dataset": "dataset_giant_100hp_varyK",
            "historical_generator_repository": "JuliaPelzer/Dataset-generation-with-Pflotran",
            "historical_generator_commit": HISTORICAL_GENERATOR_COMMIT,
            "historical_generator_path": HISTORICAL_GENERATOR_PATH,
        },
    }

    destination, metadata_path = save_pce_proof_of_concept_result(
        result,
        args.output,
        metadata=metadata,
    )

    diagnostics = result.diagnostics
    print(f"Saved PCE proof-of-concept result to {destination}")
    print(f"Saved reproducibility metadata to {metadata_path}")
    print(
        "PCE basis: "
        f"dimension={result.regressor.dimension}, degree={result.regressor.degree}, "
        f"terms={result.regressor.basis_size}"
    )
    print(
        "Expensive LGCNN evaluations: "
        f"train={args.n_train}, validation={args.n_validation}, "
        f"total={args.n_train + args.n_validation}"
    )
    print(
        "Streamlines: "
        f"mode={args.streamline_mode}, method={args.streamline_method}, "
        f"max_nfev={args.streamline_max_nfev}"
    )
    print(
        "Validation metrics: "
        f"RMSE={diagnostics.rmse:.6g}, MAE={diagnostics.mae:.6g}, "
        f"relative_L2={diagnostics.relative_l2_error:.6g}, "
        f"Q2={diagnostics.q2:.6g}"
    )
    print(
        "QoI moments: "
        f"LGCNN-MC mean={diagnostics.validation_mean:.6g}, "
        f"PCE mean={diagnostics.pce_mean:.6g}, "
        f"LGCNN-MC variance={diagnostics.validation_variance:.6g}, "
        f"PCE variance={diagnostics.pce_variance:.6g}"
    )

    if args.plots_dir:
        paths = plot_pce_archive(destination, args.plots_dir)
        print(f"Saved {len(paths)} PCE diagnostic plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
