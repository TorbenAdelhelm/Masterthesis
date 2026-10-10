"""Resolve real-K training membership and summarize geostatistical support.

This workflow is deliberately diagnostic: with only a few realistic simulations,
it reports support statistics and plausible sensitivity ranges but does not claim
that covariance/amplitude parameters are statistically calibrated. If original
CNN split metadata are unavailable, the report is explicitly labelled as
``realistic_dataset_support`` rather than fabricated ``training_support``.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from ..sampling.darus_real import (
    load_release25_raw_permeability_dataset,
    load_release25_raw_permeability_run,
)
from ..sampling.geostatistical_target import (
    Base10LognormalTarget,
    HydraulicConductivityValidationAccumulator,
    validate_hydraulic_conductivity_marginal,
)
from ..sampling.hydraulic_conductivity import (
    FluidProperties,
    hydraulic_conversion_metadata,
    permeability_to_hydraulic_conductivity,
)
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
    p.add_argument("--cnn1-command-line")
    p.add_argument("--cnn3-command-line")
    p.add_argument("--cnn1-prepared-inputs")
    p.add_argument("--cnn3-prepared-inputs")
    p.add_argument("--runs", nargs="+", help="Optional explicit realistic RUN subset for support profiling")
    p.add_argument("--cell-size-m", type=float, default=5.0)
    p.add_argument("--permeability-convention", choices=("historical-training", "physical"),
                   default="historical-training")
    p.add_argument("--rho", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=1.0e-3)
    p.add_argument("--g", type=float, default=9.80665)
    p.add_argument("--target-log10-mean", type=float, default=-3.0)
    p.add_argument("--target-log10-std", type=float, default=0.5,
                   help="Base-10 log-space standard deviation; 0.5 means variance 0.25")
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


def _field_checksum(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


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
    if bool(args.cnn1_command_line) != bool(args.cnn3_command_line):
        raise ValueError("provide both CNN1 and CNN3 command-line metadata, or neither")

    splits = None
    if args.cnn1_command_line:
        splits = {
            "cnn1": _resolve(args.cnn1_command_line, args.cnn1_prepared_inputs, args.dataset_root),
            "cnn3": _resolve(args.cnn3_command_line, args.cnn3_prepared_inputs, args.dataset_root),
        }
        common_training = sorted(set(splits["cnn1"].train_runs) & set(splits["cnn3"].train_runs))
        inventory = sorted(set(splits["cnn1"].inventory) | set(splits["cnn3"].inventory),
                           key=lambda value: int(value.removeprefix("RUN_")))
        support_kind = "training_support" if common_training else "resolved_but_no_common_training_runs"
    else:
        _, discovered = load_release25_raw_permeability_dataset(
            args.dataset_root, cell_size_m=args.cell_size_m
        )
        inventory = list(discovered)
        common_training = []
        support_kind = "realistic_dataset_support"

    if args.runs:
        requested = list(dict.fromkeys(args.runs))
        unknown = sorted(set(requested) - set(inventory))
        if unknown:
            raise ValueError(f"requested RUNs are not present in the dataset inventory: {unknown}")
        inventory = requested

    all_accumulator = HydraulicConductivityValidationAccumulator(target)
    training_accumulator = HydraulicConductivityValidationAccumulator(target) if common_training else None
    runs: dict[str, object] = {}
    for run in inventory:
        k = load_release25_raw_permeability_run(args.dataset_root, run, cell_size_m=args.cell_size_m)
        kh = permeability_to_hydraulic_conductivity(
            k, convention=args.permeability_convention, fluid=fluid
        )
        all_accumulator.update(kh)
        if training_accumulator is not None and run in common_training:
            training_accumulator.update(kh)
        role = {"dataset": "realistic_reference"}
        if splits is not None:
            role.update({
                name: (
                    "training" if run in split.train_runs else
                    "validation" if run in split.validation_runs else
                    "test" if run in split.test_runs else "inventory-only"
                ) for name, split in splits.items()
            })
        entry: dict[str, object] = {
            "role": role,
            "permeability_array_sha256": _field_checksum(k),
            "marginal": _summary(kh),
            "target_validation": validate_hydraulic_conductivity_marginal(kh, target),
        }
        try:
            entry["directional_variograms"] = directional_variograms(
                np.log10(kh), cell_size_m=args.cell_size_m
            )
        except Exception as exc:  # diagnostics must not invalidate provenance/marginal analysis
            entry["directional_variograms_error"] = str(exc)
        try:
            entry["spectral"] = spatial_power_spectrum(
                np.log10(kh), cell_size_m=args.cell_size_m
            )
        except Exception as exc:
            entry["spectral_error"] = str(exc)
        runs[run] = entry

    report: dict[str, object] = {
        "status": support_kind,
        "scientific_status": (
            "Training/realistic fields constrain support diagnostics only; residual amplitude and "
            "covariance remain explicit model-form assumptions."
        ),
        "dataset_root": str(Path(args.dataset_root).expanduser().resolve()),
        "cell_size_m": args.cell_size_m,
        "permeability_convention": args.permeability_convention,
        "hydraulic_conversion": hydraulic_conversion_metadata(
            args.permeability_convention, fluid=fluid
        ),
        "target": target.metadata,
        "common_proven_training_runs": common_training,
        "profiled_runs": inventory,
        "dataset_inventory_validation": all_accumulator.finalize(),
        "runs": runs,
        "interpretation": (
            "Use marginal, variogram, spectral and patch statistics to choose plausible sensitivity "
            "ranges. Do not estimate a pixelwise PCA/KLT across differently located/rotated RUNs."
        ),
    }
    if splits is not None:
        report["cnn1_split"] = splits["cnn1"].manifest
        report["cnn3_split"] = splits["cnn3"].manifest
    else:
        report["training_membership"] = {
            "status": "unresolved",
            "required_evidence": (
                "original CNN1/CNN3 command_line_arguments.yaml plus prepared Inputs inventory "
                "or the exact historical raw inventory"
            ),
        }
    if training_accumulator is not None:
        report["common_training_validation"] = training_accumulator.finalize()

    (root / "training_support.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False), encoding="utf-8"
    )
    (root / "training_support.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved real-K support analysis to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
