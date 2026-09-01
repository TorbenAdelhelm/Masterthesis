from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..io import load_prepared_permeability_ensemble
from ..propagation import MonteCarloRunner
from ..sampling import EmpiricalPermeabilitySampler, load_empirical_fields
from ..surrogates import Release25Surrogate
from ..surrogates.release25_runtime import Release25Runtime


def _run_ids(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one comma-separated run id is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run empirical Monte Carlo UQ through the real release25 "
            "CNN1 -> original Step 2 -> CNN3 pipeline."
        )
    )
    parser.add_argument("--release25-repo", required=True)
    parser.add_argument("--cnn1-dir", required=True)
    parser.add_argument("--cnn2-dir", required=True)
    parser.add_argument("--prepared-pki-dir", required=True)
    parser.add_argument(
        "--fixed-run-id",
        required=True,
        help="Prepared pki datapoint that supplies fixed pressure and heat-pump positions.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--ensemble-file",
        help=".npy/.npz/.pt/.pth permeability ensemble with shape [N,H,W].",
    )
    source.add_argument(
        "--permeability-run-ids",
        type=_run_ids,
        help=(
            "Comma-separated prepared pki run ids whose permeability fields form "
            "the empirical ensemble. Fixed p/i still come from --fixed-run-id."
        ),
    )
    parser.add_argument("--ensemble-key")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--ddof", type=int, default=1)
    parser.add_argument("--store-all", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--random-k", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--streamline-method", choices=("RK45", "RK23", "Radau"), default="RK45"
    )
    parser.add_argument("--output", default="run_output/release25_empirical_mc.npz")
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

    if args.ensemble_file:
        fields = load_empirical_fields(args.ensemble_file, key=args.ensemble_key)
    elif args.permeability_run_ids:
        fields = load_prepared_permeability_ensemble(
            args.prepared_pki_dir,
            args.permeability_run_ids,
            device="cpu",
        )
    else:
        fields = runtime.scenario.original_permeability.detach().cpu().numpy()[None, ...]

    if tuple(fields.shape[1:]) != runtime.scenario.shape:
        raise ValueError(
            f"ensemble field shape {tuple(fields.shape[1:])} does not match fixed "
            f"scenario {runtime.scenario.shape}"
        )

    sampler = EmpiricalPermeabilitySampler(fields, batch_size=args.batch_size)
    surrogate = Release25Surrogate(
        adapter=runtime.adapter,
        fixed_inputs=runtime.scenario.fixed,
        device=args.device,
    )
    result = MonteCarloRunner(sampler=sampler, surrogate=surrogate).run(
        n_samples=args.n_samples,
        store_all=args.store_all,
        ddof=args.ddof,
    )

    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "count": np.asarray(result.count, dtype=np.int64),
        "mean": result.mean,
        "variance": result.variance,
        "std": result.std,
        "minimum": result.minimum,
        "maximum": result.maximum,
    }
    if result.samples is not None:
        payload["samples"] = result.samples
    np.savez_compressed(destination, **payload)
    print(f"Saved Monte Carlo result to {destination}")
    print(f"Samples propagated: {result.count}")
    print(f"Temperature field shape: {result.mean.shape}")


if __name__ == "__main__":
    main()
