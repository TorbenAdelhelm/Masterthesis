from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from ..sampling import (
    ConditionalKLLogGaussianPermeabilityMap,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
    RadialExponentialPermeabilitySampler,
    calibrate_covariance_candidates,
    load_empirical_fields,
    load_release25_raw_permeability_dataset,
    sample_borehole_observations,
)


def _load_source(
    path: str,
    *,
    key: str | None,
    cell_size_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        fields, runs = load_release25_raw_permeability_dataset(
            source, cell_size_m=cell_size_m
        )
        return fields, {
            "kind": "release25_raw_pflotran_h5",
            "path": str(source),
            "runs": list(runs),
        }
    fields = load_empirical_fields(source, key=key)
    return fields, {
        "kind": "empirical_array",
        "path": str(source),
        "key": key,
    }


def _calibrate(args: argparse.Namespace) -> int:
    fields, source_meta = _load_source(
        args.fields, key=args.key, cell_size_m=args.cell_size_m
    )
    results = calibrate_covariance_candidates(
        fields,
        cell_size_m=args.cell_size_m,
        models=args.models,
        max_lag_cells=args.max_lag_cells,
        spatial_stride=args.spatial_stride,
    )
    payload = {
        "schema_version": 1,
        "source": {
            **source_meta,
            "field_shape": list(fields.shape[1:]),
            "field_count": int(fields.shape[0]),
        },
        "selected": results[0].to_dict(),
        "candidates": [item.to_dict() for item in results],
        "selection_note": (
            "Candidates are ordered by joint x/y/diagonal empirical-variogram RMSE. "
            "The ranking is a calibration diagnostic and requires geological/held-out validation."
        ),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return 0


def _select_calibration(payload: dict, model: str | None) -> dict:
    if model is None:
        return dict(payload["selected"])
    for candidate in payload.get("candidates", []):
        if candidate.get("covariance_model") == model:
            return dict(candidate)
    raise ValueError(f"covariance model {model!r} is not present in calibration file")


def _validation(truth: np.ndarray, ensemble: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    truth_log = np.log10(np.asarray(truth, dtype=np.float64))
    ensemble_log = np.log10(np.asarray(ensemble, dtype=np.float64))
    mean = np.mean(ensemble_log, axis=0)
    error = mean - truth_log
    q05 = np.quantile(ensemble_log, 0.05, axis=0)
    q95 = np.quantile(ensemble_log, 0.95, axis=0)
    coverage = np.mean((truth_log >= q05) & (truth_log <= q95))
    observed_truth = truth_log[indices[:, 0], indices[:, 1]]
    observed_generated = ensemble_log[:, indices[:, 0], indices[:, 1]]
    conditioning_max_abs = np.max(
        np.abs(observed_generated - observed_truth[None, :])
    )
    return {
        "log10_mean_rmse": float(np.sqrt(np.mean(error * error))),
        "log10_mean_mae": float(np.mean(np.abs(error))),
        "empirical_90pct_coverage": float(coverage),
        "conditioning_max_abs_log10": float(conditioning_max_abs),
    }


def _generate(args: argparse.Namespace) -> int:
    with Path(args.calibration).expanduser().resolve().open("r", encoding="utf-8") as handle:
        calibration_file = yaml.safe_load(handle)
    calibration = _select_calibration(calibration_file, args.model)

    fields, source_meta = _load_source(
        args.fields,
        key=args.key,
        cell_size_m=float(calibration["cell_size_m"]),
    )
    truth_index = int(args.truth_index)
    if truth_index < 0 or truth_index >= fields.shape[0]:
        raise ValueError("truth-index is outside the empirical field ensemble")
    truth = np.asarray(fields[truth_index], dtype=np.float32)
    boreholes = sample_borehole_observations(
        truth,
        n_boreholes=args.n_boreholes,
        seed=args.borehole_seed,
        margin_cells=args.margin_cells,
        min_spacing_cells=args.min_spacing_cells,
    )

    model = str(calibration["covariance_model"])
    common = dict(
        shape=tuple(int(v) for v in truth.shape),
        mean_log10_k=float(calibration["mean_log10_k"]),
        std_log10_k=float(calibration["std_log10_k"]),
    )
    if model == "radial_exponential":
        if float(args.observation_std_log10_k) != 0.0:
            raise ValueError(
                "radial exponential GSTools generation currently supports exact boreholes only"
            )
        sampler = RadialExponentialPermeabilitySampler(
            **common,
            cell_size_m=float(calibration["cell_size_m"]),
            length_scale_y_m=float(calibration["length_scale_y_m"]),
            length_scale_x_m=float(calibration["length_scale_x_m"]),
            angle_rad=float(calibration.get("angle_rad", 0.0)),
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            observation_indices=boreholes.indices,
            observation_k=boreholes.permeability,
        )
    else:
        cell_size = float(calibration["cell_size_m"])
        prior = KLLogGaussianPermeabilityMap(
            **common,
            domain_size_m=(truth.shape[0] * cell_size, truth.shape[1] * cell_size),
            length_scale_m=(
                float(calibration["length_scale_y_m"]),
                float(calibration["length_scale_x_m"]),
            ),
            covariance_model=model,
            n_modes=args.n_modes,
            energy_threshold=args.energy_threshold,
        )
        conditional = ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
            prior=prior,
            observation_indices=boreholes.indices,
            observation_k=boreholes.permeability,
            observation_std_log10_k=args.observation_std_log10_k,
        )
        sampler = GaussianCoordinatePermeabilitySampler(
            field_map=conditional,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    ensemble = np.concatenate(list(sampler), axis=0)
    metrics = _validation(truth, ensemble, boreholes.indices)
    metadata = {
        "schema_version": 1,
        "covariance_calibration": calibration,
        "sampler": getattr(sampler, "metadata", {}),
        "truth_source": source_meta,
        "truth_index": truth_index,
        "boreholes": boreholes.to_dict(),
        "validation": metrics,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        permeability_fields=ensemble.astype(np.float32),
        borehole_indices=boreholes.indices,
        borehole_k=boreholes.permeability,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate and generate realistic log-Gaussian permeability fields."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="fit covariance candidates to real fields")
    calibrate.add_argument(
        "--fields",
        required=True,
        help=(
            "Empirical .npy/.npz/.pt/.pth ensemble or an unpacked release25 raw "
            "dataset root containing settings.yaml and RUN_*/pflotran.h5."
        ),
    )
    calibrate.add_argument("--key")
    calibrate.add_argument("--cell-size-m", type=float, required=True)
    calibrate.add_argument("--models", nargs="+", default=[
        "matern32", "exponential", "radial_exponential"
    ])
    calibrate.add_argument("--max-lag-cells", type=int, default=64)
    calibrate.add_argument("--spatial-stride", type=int, default=1)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=_calibrate)

    generate = sub.add_parser(
        "generate", help="sample synthetic boreholes and conditioned permeability fields"
    )
    generate.add_argument("--fields", required=True)
    generate.add_argument("--key")
    generate.add_argument("--calibration", required=True)
    generate.add_argument(
        "--model", choices=["matern32", "exponential", "radial_exponential"]
    )
    generate.add_argument("--truth-index", type=int, default=0)
    generate.add_argument("--n-boreholes", type=int, required=True)
    generate.add_argument("--borehole-seed", type=int, default=2907)
    generate.add_argument("--margin-cells", type=int, default=0)
    generate.add_argument("--min-spacing-cells", type=float, default=0.0)
    generate.add_argument("--n-samples", type=int, default=32)
    generate.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--seed", type=int, default=3901)
    generate.add_argument("--n-modes", type=int, default=64)
    generate.add_argument("--energy-threshold", type=float, default=0.95)
    generate.add_argument("--observation-std-log10-k", type=float, default=0.0)
    generate.add_argument("--output", required=True)
    generate.set_defaults(func=_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
