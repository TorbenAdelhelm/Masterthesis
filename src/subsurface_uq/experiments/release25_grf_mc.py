from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import git_head, runtime_versions, save_monte_carlo_result, sha256_file
from ..propagation import MonteCarloRunner
from ..sampling import (
    RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
    ConditionalKLLogGaussianPermeabilityMap,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
)
from ..surrogates import Release25Surrogate
from ..surrogates.bounded_streamlines import configure_release25_streamlines
from ..surrogates.release25_runtime import Release25Runtime
from ..visualization import plot_monte_carlo_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run forward Monte Carlo UQ through the pretrained release25 LGCNN "
            "using an unconditional or borehole-conditioned KL log-Gaussian "
            "permeability field. GRF hyperparameters are explicit user inputs; "
            "the command does not define calibrated scientific defaults."
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

    parser.add_argument("--mean-log10-k", type=float, required=True)
    parser.add_argument("--std-log10-k", type=float, required=True)
    parser.add_argument("--length-scale-y-m", type=float, required=True)
    parser.add_argument("--length-scale-x-m", type=float, required=True)
    truncation = parser.add_mutually_exclusive_group(required=True)
    truncation.add_argument(
        "--n-modes",
        type=int,
        help="Fixed number of retained KL modes.",
    )
    truncation.add_argument(
        "--energy-threshold",
        type=float,
        help="Retain the smallest KL basis reaching this prior variance fraction.",
    )

    parser.add_argument(
        "--observation",
        action="append",
        nargs=3,
        type=float,
        metavar=("ROW", "COL", "K_M2"),
        help=(
            "Optional conditioning observation at an integer permeability-grid cell. "
            "Repeat the option for multiple boreholes, e.g. "
            "--observation 40 70 2e-10 --observation 150 110 5e-10."
        ),
    )
    parser.add_argument(
        "--observation-std-log10-k",
        type=float,
        default=0.0,
        help=(
            "Shared Gaussian observation standard deviation in log10(K). "
            "Default 0 performs exact conditioning in the retained KL model."
        ),
    )

    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2907)
    parser.add_argument(
        "--ddof",
        type=int,
        default=None,
        help=(
            "Variance degrees of freedom. By default uses ddof=1 for N>=2 and "
            "ddof=0 for a one-sample diagnostic."
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
        "--streamline-mode",
        choices=("release25", "bounded"),
        default="bounded",
        help=(
            "Streamline implementation. GRF UQ defaults to the robust bounded mode; "
            "'release25' remains available for reference comparisons."
        ),
    )
    parser.add_argument(
        "--streamline-max-nfev",
        type=int,
        default=100_000,
        help="Maximum RHS evaluations per streamline in bounded mode; 0 disables the watchdog.",
    )
    parser.add_argument("--streamline-diagnostics", action="store_true")
    parser.add_argument("--streamline-slow-seconds", type=float, default=2.0)

    parser.add_argument(
        "--background-temperature",
        type=float,
        default=RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C,
    )
    parser.add_argument(
        "--exceedance-thresholds",
        type=float,
        nargs="+",
        default=(0.1, 1.0),
        metavar="DELTA_T",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=5.0,
        help="Permeability-grid cell size in metres (default 5.0 for release25).",
    )
    parser.add_argument("--output", default="run_output/release25_grf_mc.npz")
    parser.add_argument(
        "--plots-dir",
        help="Optional directory for mean/std/range/Delta-T/exceedance UQ maps.",
    )
    return parser


def build_grf_map(
    args: argparse.Namespace,
    *,
    shape: tuple[int, int],
):
    """Build the unconditional or conditioned stochastic permeability map."""

    if args.cell_size_m <= 0.0 or not np.isfinite(args.cell_size_m):
        raise ValueError("cell_size_m must be finite and positive")
    domain_size_m = (
        float(shape[0]) * float(args.cell_size_m),
        float(shape[1]) * float(args.cell_size_m),
    )
    prior = KLLogGaussianPermeabilityMap(
        shape=shape,
        domain_size_m=domain_size_m,
        mean_log10_k=float(args.mean_log10_k),
        std_log10_k=float(args.std_log10_k),
        length_scale_m=(
            float(args.length_scale_y_m),
            float(args.length_scale_x_m),
        ),
        n_modes=args.n_modes,
        energy_threshold=(
            float(args.energy_threshold) if args.energy_threshold is not None else 0.95
        ),
    )

    observations = args.observation or []
    if not observations:
        if float(args.observation_std_log10_k) != 0.0:
            raise ValueError(
                "observation_std_log10_k requires at least one --observation"
            )
        return prior

    rows = np.asarray(observations, dtype=np.float64)
    indices = rows[:, :2]
    permeability = rows[:, 2]
    return ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
        prior=prior,
        observation_indices=indices,
        observation_k=permeability,
        observation_std_log10_k=float(args.observation_std_log10_k),
    )


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
    field_map = build_grf_map(args, shape=shape)
    sampler = GaussianCoordinatePermeabilitySampler(
        field_map=field_map,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
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
    conditioned = isinstance(field_map, ConditionalKLLogGaussianPermeabilityMap)
    metadata = {
        **sampler.metadata,
        "uq_mode": (
            "conditional_kl_log_gaussian_forward_monte_carlo"
            if conditioned
            else "kl_log_gaussian_forward_monte_carlo"
        ),
        "scientific_parameter_status": (
            "GRF mean, standard deviation, length scales and KL truncation are explicit "
            "user inputs and are not claimed to be calibrated defaults."
        ),
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
        "GRF prior: "
        f"mean_log10_k={args.mean_log10_k:g}, std_log10_k={args.std_log10_k:g}, "
        f"length_scale_m=({args.length_scale_y_m:g}, {args.length_scale_x_m:g}), "
        f"prior_dimension={field_map.prior.dimension if conditioned else field_map.dimension}"
    )
    if conditioned:
        print(
            "Conditioning: "
            f"observations={len(args.observation)}, posterior_dimension={field_map.dimension}, "
            f"observation_std_log10_k={args.observation_std_log10_k:g}"
        )
    else:
        print("Conditioning: none")
    print(
        "Streamlines: "
        f"mode={args.streamline_mode}, method={args.streamline_method}, "
        f"max_nfev={args.streamline_max_nfev}"
    )
    print(f"Variance ddof: {effective_ddof}")

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
