"""Resolve real-K training membership and summarize geostatistical support.

This workflow is deliberately diagnostic: with only a few realistic simulations,
it reports training-support statistics and plausible sensitivity ranges but does
not claim that covariance/amplitude parameters are statistically calibrated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..sampling.darus_real import load_release25_raw_permeability_run
from ..sampling.geostatistical_target import Base10LognormalTarget, validate_hydraulic_conductivity_marginal
from ..sampling.hydraulic_conductivity import FluidProperties, permeability_to_hydraulic_conductivity
from ..sampling.spatial_diagnostics import directional_variograms, spatial_power_spectrum
from ..sampling.training_split_provenance import resolve_release25_training_split


def _summary(values: np.ndarray) -> dict[str, object]:
    logs = np.log10(values.astype(np.float64))
    return {
        "log10_mean": float(np.mean(logs)),
        "log10_std": float(np.std(logs, ddof=1)),
        "log10_variance": float(np.var(logs, ddof=1)),
        "quantiles_m_s": {
            "q01": float(np.quantile(values, .01)),
            "q05": float(np.quantile(values, .05)),
            "q50": float(np.quantile(values, .50)),
            "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--cnn1-command-line", required=True)
    p.add_argument("--cnn3-command-line", required=True)
    p.add_argument("--cnn1-prepared-inputs")
    p.add_argument("--cnn3-prepared-inputs")
    p.add_argument("--cell-size-m", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=1.0e-3)
    p.add_argument("--g", type=float, default=9.80665)
    p.add_argument("--target-log10-mean", type=float, default=-3.0)
    p.add_argument("--target-log10-std", type=float, default=0.5)
    p.add_argument("--target-lb-m-s", type=float, default=1.0e-4)
    p.add_argument("--target-ub-m-s", type=float, default=5.0e-2)
    p.add_argument("--minimum-interval-fraction", type=float, default=0.95)
    p.add_argument("--output-dir", required=True)
    return p


def _resolve(path, prepared, dataset):
    return resolve_release25_training_split(
        path,
        prepared_inputs_dir=prepared,
        raw_dataset_root=None if prepared else dataset,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fluid = FluidProperties(args.rho, args.mu, args.g)
    target = Base10LognormalTarget(
        args.target_log10_mean,
        args.target_log10_std,
        (args.target_lb_m_s, args.target_ub_m_s),
        args.minimum_interval_fraction,
    )
    splits = {
        "cnn1": _resolve(args.cnn1_command_line, args.cnn1_prepared_inputs, args.dataset_root),
        "cnn3": _resolve(args.cnn3_command_line, args.cnn3_prepared_inputs, args.dataset_root),
    }
    proven_training = sorted(set(splits["cnn1"].train_runs) & set(splits["cnn3"].train_runs))
    support_status = (
        "resolved_common_training_support" if proven_training else "resolved_but_no_common_training_runs"
    )
    runs: dict[str, object] = {}
    pooled = []
    for run in sorted(set(splits["cnn1"].inventory) | set(splits["cnn3"].inventory)):
        k = load_release25_raw_permeability_run(args.dataset_root, run, cell_size_m=args.cell_size_m)
        kh = permeability_to_hydraulic_conductivity(k, fluid=fluid)
        pooled.append(kh.reshape(-1))
        # Variograms/spectra can be expensive on 2560²; spatial_diagnostics implementations
        # already use bounded summaries.  Keep raw marginal reporting independent of them.
        entry = {
            "role": {
                name: (
                    "training" if run in split.train_runs else
                    "validation" if run in split.validation_runs else
                    "test" if run in split.test_runs else "inventory-only"
                ) for name, split in splits.items()
            },
            "marginal": _summary(kh),
            "target_validation": validate_hydraulic_conductivity_marginal(kh, target),
        }
        try:
            entry["directional_variograms"] = directional_variograms(np.log10(kh), cell_size_m=args.cell_size_m)
        except Exception as exc:  # diagnostic must not invalidate provenance/marginal analysis
            entry["directional_variograms_error"] = str(exc)
        try:
            entry["spectral"] = spatial_power_spectrum(np.log10(kh), cell_size_m=args.cell_size_m)
        except Exception as exc:
            entry["spectral_error"] = str(exc)
        runs[run] = entry

    pooled_values = np.concatenate(pooled)
    report = {
        "status": support_status,
        "scientific_status": (
            "training fields constrain support diagnostics only; residual amplitude/covariance remain model-form assumptions"
        ),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "cell_size_m": args.cell_size_m,
        "fluid_properties": fluid.metadata,
        "target": target.metadata,
        "cnn1_split": splits["cnn1"].manifest,
        "cnn3_split": splits["cnn3"].manifest,
        "common_proven_training_runs": proven_training,
        "pooled_training_dataset_inventory_marginal": _summary(pooled_values),
        "pooled_target_validation": validate_hydraulic_conductivity_marginal(pooled_values, target),
        "runs": runs,
        "interpretation": (
            "Use training-run marginal, variogram, spectral and patch statistics to choose plausible sensitivity ranges. "
            "Do not estimate a pixelwise PCA/KLT across differently located/rotated RUNs."
        ),
    }
    (root / "training_support.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    (root / "training_support.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved real-K training support analysis to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
