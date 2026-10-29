"""Hermite response PCE under a serialized Gaussian-coordinate input law."""
from __future__ import annotations

import argparse
from math import comb

import numpy as np

from ..sampling.input_model import load_conditional_kl_input_model
from ..pce import CoordinateQoIEvaluator, MeanTemperatureAnomaly, run_gaussian_pce, save_pce_proof_of_concept_result
from ..surrogates import Release25Surrogate
from ..surrogates.release25_runtime import Release25Runtime
from ..surrogates.bounded_streamlines import configure_release25_streamlines
from ..io import runtime_versions, sha256_file


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-model", "release25-repo", "cnn1-dir", "cnn2-dir", "prepared-pki-dir", "fixed-run-id"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-validation", type=int, required=True)
    parser.add_argument("--background-temperature", type=float, required=True)
    parser.add_argument("--cell-size-m", type=float, default=5.)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--train-seed", type=int, default=4901)
    parser.add_argument("--validation-seed", type=int, default=4902)
    parser.add_argument("--random-k", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--streamline-mode", choices=("release25", "bounded"), default="release25")
    parser.add_argument("--streamline-method", choices=("RK45", "RK23", "Radau"), default="RK45")
    parser.add_argument("--streamline-max-nfev", type=int, default=100000)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    loaded = load_conditional_kl_input_model(args.input_model)
    if args.degree < 0:
        parser.error("degree must be non-negative")
    term_count = comb(loaded.conditional.dimension + args.degree, args.degree)
    if args.n_train < term_count or args.n_validation < 2:
        parser.error(f"need n_train >= {term_count} basis terms and n_validation >= 2")
    runtime = Release25Runtime.from_paths(
        release25_repo=args.release25_repo, cnn1_dir=args.cnn1_dir, cnn2_dir=args.cnn2_dir,
        prepared_pki_dir=args.prepared_pki_dir, run_id=args.fixed_run_id, device=args.device,
        random_k=args.random_k, streamline_method=args.streamline_method,
    )
    expected_domain = np.asarray(runtime.scenario.shape) * args.cell_size_m
    if (tuple(runtime.scenario.shape) != loaded.conditional.field_shape
            or not np.allclose(expected_domain, loaded.prior.domain_size_m, rtol=0, atol=1e-8)):
        parser.error("input law must match the fixed LGCNN scenario's shape and physical cell size")
    configure_release25_streamlines(runtime.adapter, mode=args.streamline_mode,
                                    max_nfev=args.streamline_max_nfev, diagnostics=False, slow_streamline_seconds=2.)
    surrogate = Release25Surrogate(runtime.adapter, runtime.scenario.fixed, device=args.device)
    functional = MeanTemperatureAnomaly(args.background_temperature)
    result = run_gaussian_pce(
        CoordinateQoIEvaluator(loaded.conditional, surrogate, functional, batch_size=args.batch_size),
        degree=args.degree, n_train=args.n_train, n_validation=args.n_validation,
        train_seed=args.train_seed, validation_seed=args.validation_seed,
    )
    paths = save_pce_proof_of_concept_result(result, args.output, metadata={
        "uq_mode": "release25_gaussian_hermite_pce", "field_map": loaded.conditional.metadata,
        "input_model": str(loaded.source_path), "input_model_sha256": sha256_file(loaded.source_path),
        "fixed_run_id": args.fixed_run_id, "random_k": args.random_k,
        "qoi": {"name": functional.name, "background_temperature_c": args.background_temperature},
        "train_seed": args.train_seed, "validation_seed": args.validation_seed,
        "runtime_versions": runtime_versions(),
    })
    print(f"Saved Hermite PCE ({term_count} terms) and independent Gaussian validation to {paths[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
