from __future__ import annotations

import argparse
from pathlib import Path

from ..io import git_head, runtime_versions, save_monte_carlo_result, sha256_file
from ..propagation import MonteCarloRunner
from ..sampling import (
    RELEASE25_PERLIN_DEFAULT_SEED,
    RELEASE25_PERLIN_FREQUENCY,
    RELEASE25_PERLIN_K_MAX,
    RELEASE25_PERLIN_K_MIN,
    RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
    Release25PerlinPermeabilitySampler,
)
from ..surrogates import Release25Surrogate
from ..surrogates.release25_runtime import Release25Runtime
from ..visualization import plot_monte_carlo_archive

HISTORICAL_GENERATOR_COMMIT = "8549bbd9e22d2bc75ce2038c1a0397359e45c971"
HISTORICAL_GENERATOR_PATH = "scripts/create_varying_field.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run forward Monte Carlo UQ through the pretrained release25 LGCNN "
            "using permeability realizations from the historical synthetic "
            "perlin_v2 input distribution."
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
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="Number of Perlin permeability realizations to propagate.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=RELEASE25_PERLIN_DEFAULT_SEED)
    parser.add_argument(
        "--frequency",
        type=float,
        nargs=2,
        metavar=("FX", "FY"),
        default=RELEASE25_PERLIN_FREQUENCY,
        help="Perlin frequencies. Default: 18 18 from the released synthetic dataset.",
    )
    parser.add_argument(
        "--k-min",
        type=float,
        default=RELEASE25_PERLIN_K_MIN,
        help="Minimum permeability in m^2 from the released synthetic dataset.",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=RELEASE25_PERLIN_K_MAX,
        help="Maximum permeability in m^2 from the released synthetic dataset.",
    )
    parser.add_argument("--base-start", type=int, default=0)
    parser.add_argument(
        "--ddof",
        type=int,
        default=None,
        help=(
            "Variance degrees of freedom. By default uses ddof=1 for N>=2 and "
            "ddof=0 for a one-sample deterministic diagnostic."
        ),
    )
    parser.add_argument("--store-all", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--random-k", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--streamline-method", choices=("RK45", "RK23", "Radau"), default="RK45"
    )
    parser.add_argument(
        "--background-temperature",
        type=float,
        default=RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
        help=(
            "Background temperature for Delta-T UQ statistics in degC. "
            "Default: 10.6, the initial groundwater temperature in the historical "
            "synthetic PFLOTRAN setup."
        ),
    )
    parser.add_argument(
        "--exceedance-thresholds",
        type=float,
        nargs="+",
        default=(0.1, 1.0),
        metavar="DELTA_T",
        help=(
            "Delta-T thresholds in degC for online exceedance probabilities. "
            "Default: 0.1 1.0."
        ),
    )
    parser.add_argument("--output", default="run_output/release25_perlin_mc.npz")
    parser.add_argument(
        "--plots-dir",
        help="Optional directory for mean/std/range/Delta-T/exceedance UQ maps.",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=5.0,
        help="Grid-cell size used for UQ plot axes in metres (default: 5.0).",
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

    # The sampler uses the fixed scenario shape/domain. For the published
    # synthetic baseline this is 2560 x 2560 at 5 m, i.e. 12.8 km square.
    shape = tuple(int(value) for value in runtime.scenario.shape)
    domain_size_m = (shape[0] * 5.0, shape[1] * 5.0)
    sampler = Release25PerlinPermeabilitySampler(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        shape=shape,
        domain_size_m=domain_size_m,
        frequency=(float(args.frequency[0]), float(args.frequency[1])),
        k_min=args.k_min,
        k_max=args.k_max,
        base_start=args.base_start,
    )

    surrogate = Release25Surrogate(
        adapter=runtime.adapter,
        fixed_inputs=runtime.scenario.fixed,
        device=args.device,
    )
    effective_ddof = args.ddof if args.ddof is not None else (0 if args.n_samples == 1 else 1)
    result = MonteCarloRunner(sampler=sampler, surrogate=surrogate).run(
        n_samples=args.n_samples,
        store_all=args.store_all,
        ddof=effective_ddof,
        background_temperature=args.background_temperature,
        exceedance_thresholds=args.exceedance_thresholds,
    )

    project_root = Path(__file__).resolve().parents[3]
    metadata = {
        **sampler.metadata,
        "uq_mode": "synthetic_perlin_forward_monte_carlo",
        "fixed_run_id": args.fixed_run_id,
        "prepared_pki_dir": str(Path(args.prepared_pki_dir)),
        "release25_repo": str(Path(args.release25_repo)),
        "cnn1_dir": str(Path(args.cnn1_dir)),
        "cnn2_dir": str(Path(args.cnn2_dir)),
        "cnn2_dir_role": "LGCNN Step 3 / CNN3 (legacy argument name)",
        "device": args.device,
        "streamline_method": args.streamline_method,
        "random_k": bool(args.random_k),
        "ddof": int(effective_ddof),
        "ddof_requested": args.ddof,
        "store_all": bool(args.store_all),
        "cell_size_m": float(args.cell_size_m),
        "temperature_shape": [int(value) for value in result.mean.shape],
        "runtime": runtime_versions(),
        "provenance": {
            "masterthesis_git_sha": git_head(project_root),
            "release25_git_sha": git_head(args.release25_repo),
            "cnn1_checkpoint_sha256": sha256_file(runtime.adapter.cnn1.checkpoint),
            "cnn3_checkpoint_sha256": sha256_file(runtime.adapter.cnn2.checkpoint),
            "reference_published_model_doi": "10.18419/DARUS-5080",
            "reference_dataset": "dataset_giant_100hp_varyK",
            "reference_darus_settings": {
                "case": "perlin_v2",
                "frequency": [18, 18],
                "k_min": RELEASE25_PERLIN_K_MIN,
                "k_max": RELEASE25_PERLIN_K_MAX,
                "grid": [2560, 2560, 1],
                "domain_m": [12800, 12800, 5.0],
                "pressure_gradient": -0.003,
                "background_temperature_c": RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
                "injection_temperature_c": 15.6,
            },
            "historical_generator_repository": "JuliaPelzer/Dataset-generation-with-Pflotran",
            "historical_generator_commit": HISTORICAL_GENERATOR_COMMIT,
            "historical_generator_path": HISTORICAL_GENERATOR_PATH,
        },
    }

    destination, metadata_path = save_monte_carlo_result(
        result,
        args.output,
        metadata=metadata,
    )
    print(f"Saved Monte Carlo result to {destination}")
    print(f"Saved reproducibility metadata to {metadata_path}")
    print(f"Samples propagated: {result.count}")
    print(f"Temperature field shape: {result.mean.shape}")
    print(
        "Perlin parameters: "
        f"seed={sampler.seed}, frequency={sampler.frequency}, "
        f"k_min={sampler.k_min:.16e}, k_max={sampler.k_max:.16e}"
    )
    print(f"Variance ddof: {effective_ddof}")
    print(
        "Delta-T exceedance probabilities: "
        f"background={args.background_temperature:g} degC, "
        f"thresholds={tuple(float(value) for value in args.exceedance_thresholds)}"
    )

    if args.plots_dir:
        paths = plot_monte_carlo_archive(
            destination,
            args.plots_dir,
            cell_size_m=args.cell_size_m,
        )
        print("Saved Monte Carlo/UQ plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
