from __future__ import annotations

import argparse
import csv
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
    exact_simple_kriging_posterior,
    estimate_directional_variograms,
    load_empirical_fields,
    load_release25_raw_permeability_dataset,
    sample_borehole_observations,
)
from ..visualization.realistic_permeability import (
    plot_heldout_metric_comparison,
    plot_heldout_reconstruction,
    plot_length_scale_comparison,
    plot_variogram_fits,
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
        "schema_version": 2,
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
    if args.plots_dir:
        variograms = estimate_directional_variograms(
            fields,
            cell_size_m=args.cell_size_m,
            max_lag_cells=args.max_lag_cells,
            spatial_stride=args.spatial_stride,
        )
        root = Path(args.plots_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        plot_variogram_fits(variograms, results, root / "variogram_fits.png")
        plot_length_scale_comparison(results, root / "length_scales.png")
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


def _build_sampler(
    *,
    calibration: dict[str, object],
    truth: np.ndarray,
    boreholes,
    cell_size_m: float,
    n_samples: int,
    batch_size: int,
    seed: int,
    n_modes: int,
    energy_threshold: float,
    observation_std_log10_k: float,
):
    model = str(calibration["covariance_model"])
    common = dict(
        shape=tuple(int(v) for v in truth.shape),
        mean_log10_k=float(calibration["mean_log10_k"]),
        std_log10_k=float(calibration["std_log10_k"]),
    )
    if model == "radial_exponential":
        if float(observation_std_log10_k) != 0.0:
            raise ValueError(
                "radial exponential GSTools generation currently supports exact boreholes only"
            )
        return RadialExponentialPermeabilitySampler(
            **common,
            cell_size_m=float(cell_size_m),
            length_scale_y_m=float(calibration["length_scale_y_m"]),
            length_scale_x_m=float(calibration["length_scale_x_m"]),
            angle_rad=float(calibration.get("angle_rad", 0.0)),
            n_samples=n_samples,
            batch_size=batch_size,
            seed=seed,
            observation_indices=boreholes.indices,
            observation_k=boreholes.permeability,
        )

    prior = KLLogGaussianPermeabilityMap(
        **common,
        domain_size_m=(truth.shape[0] * cell_size_m, truth.shape[1] * cell_size_m),
        length_scale_m=(
            float(calibration["length_scale_y_m"]),
            float(calibration["length_scale_x_m"]),
        ),
        covariance_model=model,
        n_modes=n_modes,
        energy_threshold=energy_threshold,
    )
    conditional = ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
        prior=prior,
        observation_indices=boreholes.indices,
        observation_k=boreholes.permeability,
        observation_std_log10_k=observation_std_log10_k,
    )
    return GaussianCoordinatePermeabilitySampler(
        field_map=conditional,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
    )


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
    sampler = _build_sampler(
        calibration=calibration,
        truth=truth,
        boreholes=boreholes,
        cell_size_m=float(calibration["cell_size_m"]),
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        n_modes=args.n_modes,
        energy_threshold=args.energy_threshold,
        observation_std_log10_k=args.observation_std_log10_k,
    )

    ensemble = np.concatenate(list(sampler), axis=0)
    metrics = _validation(truth, ensemble, boreholes.indices)
    metadata = {
        "schema_version": 2,
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


def _posterior_validation(truth: np.ndarray, posterior) -> dict[str, float]:
    truth_log = np.log10(np.asarray(truth, dtype=np.float64))
    mean = np.asarray(posterior.mean_log10_k, dtype=np.float64)
    std = np.asarray(posterior.std_log10_k, dtype=np.float64)
    error = mean - truth_log
    z90 = 1.6448536269514722
    lower = mean - z90 * std
    upper = mean + z90 * std
    coverage = float(np.mean((truth_log >= lower) & (truth_log <= upper)))
    return {
        "log10_mean_rmse": float(np.sqrt(np.mean(error * error))),
        "log10_mean_mae": float(np.mean(np.abs(error))),
        "gaussian_90pct_coverage": coverage,
        "conditioning_max_abs_log10": float(
            posterior.conditioning_max_abs_log10_residual
        ),
        "conditioning_rms_log10": float(
            posterior.conditioning_rms_log10_residual
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty comparison table")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(args: argparse.Namespace) -> int:
    deprecated = {
        "--n-samples": args.n_samples,
        "--batch-size": args.batch_size,
        "--seed": args.seed,
        "--n-modes": args.n_modes,
        "--energy-threshold": args.energy_threshold,
    }
    supplied_deprecated = {
        key: value for key, value in deprecated.items() if value is not None
    }
    if supplied_deprecated:
        print(
            "Note: covariance-model evaluation now uses exact full-covariance "
            "simple kriging; the following legacy options are accepted but ignored: "
            + ", ".join(f"{key}={value}" for key, value in supplied_deprecated.items())
        )

    fields, source_meta = _load_source(
        args.fields, key=args.key, cell_size_m=args.cell_size_m
    )
    if fields.shape[0] < 2:
        raise ValueError("held-out evaluation requires at least two empirical fields")
    truth_index = int(args.truth_index)
    if truth_index < 0 or truth_index >= fields.shape[0]:
        raise ValueError("truth-index is outside the empirical field ensemble")

    training_fields = np.delete(fields, truth_index, axis=0)
    results = calibrate_covariance_candidates(
        training_fields,
        cell_size_m=args.cell_size_m,
        models=args.models,
        max_lag_cells=args.max_lag_cells,
        spatial_stride=args.spatial_stride,
    )
    variograms = estimate_directional_variograms(
        training_fields,
        cell_size_m=args.cell_size_m,
        max_lag_cells=args.max_lag_cells,
        spatial_stride=args.spatial_stride,
    )

    root = Path(args.output_dir).expanduser().resolve()
    figures = root / "figures"
    arrays = root / "arrays"
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)

    calibration_payload = {
        "schema_version": 2,
        "source": {
            **source_meta,
            "field_shape": list(fields.shape[1:]),
            "field_count": int(fields.shape[0]),
        },
        "heldout_truth_index": truth_index,
        "training_field_indices": [
            index for index in range(fields.shape[0]) if index != truth_index
        ],
        "selected_by_variogram_rmse": results[0].to_dict(),
        "candidates": [item.to_dict() for item in results],
    }
    with (root / "calibration_leave_one_out.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration_payload, handle, sort_keys=False)

    np.savez_compressed(
        root / "empirical_variograms.npz",
        y_distance_m=variograms["y"][0],
        y_semivariance=variograms["y"][1],
        x_distance_m=variograms["x"][0],
        x_semivariance=variograms["x"][1],
        diag_distance_m=variograms["diag"][0],
        diag_semivariance=variograms["diag"][1],
    )
    plot_variogram_fits(variograms, results, figures / "variogram_fits.png")
    plot_length_scale_comparison(results, figures / "length_scales.png")

    evaluation_stride = int(args.evaluation_stride)
    if evaluation_stride <= 0:
        raise ValueError("evaluation_stride must be positive")
    truth = np.asarray(fields[truth_index, ::evaluation_stride, ::evaluation_stride])
    effective_cell_size = float(args.cell_size_m) * evaluation_stride
    boreholes = sample_borehole_observations(
        truth,
        n_boreholes=args.n_boreholes,
        seed=args.borehole_seed,
        margin_cells=args.margin_cells,
        min_spacing_cells=args.min_spacing_cells,
    )

    comparison_rows: list[dict[str, object]] = []
    for result in results:
        model = result.covariance_model
        posterior = exact_simple_kriging_posterior(
            shape=tuple(int(value) for value in truth.shape),
            cell_size_m=effective_cell_size,
            covariance_model=model,
            mean_log10_k=result.mean_log10_k,
            std_log10_k=result.std_log10_k,
            length_scale_y_m=result.length_scale_y_m,
            length_scale_x_m=result.length_scale_x_m,
            observation_indices=boreholes.indices,
            observation_log10_k=boreholes.log10_permeability,
            observation_std_log10_k=args.observation_std_log10_k,
            angle_rad=result.angle_rad,
            chunk_rows=args.kriging_chunk_rows,
        )
        metrics = _posterior_validation(truth, posterior)
        model_dir = arrays / model
        model_dir.mkdir(parents=True, exist_ok=True)
        stale_sample = model_dir / "sample_001_log10_k.npy"
        if stale_sample.exists():
            stale_sample.unlink()
        np.save(model_dir / "posterior_mean_log10_k.npy", posterior.mean_log10_k)
        np.save(model_dir / "posterior_std_log10_k.npy", posterior.std_log10_k)
        np.save(model_dir / "posterior_variance_log10_k.npy", posterior.variance_log10_k)
        plot_heldout_reconstruction(
            truth_log10_k=np.log10(truth.astype(np.float64)),
            posterior_mean_log10_k=posterior.mean_log10_k,
            posterior_std_log10_k=posterior.std_log10_k,
            observation_indices=boreholes.indices,
            cell_size_m=effective_cell_size,
            model_name=model,
            destination=figures / f"heldout_{model}.png",
        )
        row = {
            "covariance_model": model,
            "posterior_method": "exact_full_covariance_simple_kriging",
            "length_scale_x_m": result.length_scale_x_m,
            "length_scale_y_m": result.length_scale_y_m,
            "ell_x_over_ell_y": result.length_scale_x_m / result.length_scale_y_m,
            "variogram_rmse": result.variogram_rmse,
            "variogram_rmse_x": result.variogram_rmse_x,
            "variogram_rmse_y": result.variogram_rmse_y,
            "variogram_rmse_diag": result.variogram_rmse_diag,
            **metrics,
        }
        comparison_rows.append(row)

    _write_csv(root / "model_comparison.csv", comparison_rows)
    summary = {
        "schema_version": 2,
        "source": source_meta,
        "heldout_truth_index": truth_index,
        "evaluation_stride": evaluation_stride,
        "effective_cell_size_m": effective_cell_size,
        "n_boreholes": boreholes.count,
        "borehole_seed": int(args.borehole_seed),
        "posterior_evaluation": {
            "method": "exact_full_covariance_simple_kriging",
            "kl_truncation": None,
            "monte_carlo_sampling": None,
            "chunk_rows": int(args.kriging_chunk_rows),
            "observation_std_log10_k": float(args.observation_std_log10_k),
        },
        "boreholes": boreholes.to_dict(),
        "models": comparison_rows,
        "interpretation": (
            "Covariance-family selection uses exact full-covariance simple kriging for "
            "all candidates, so the comparison is independent of KL truncation and "
            "Monte Carlo sampling noise. Variogram fit and held-out reconstruction are "
            "complementary diagnostics; no automatic geological winner is declared."
        ),
    }
    with (root / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    plot_heldout_metric_comparison(
        comparison_rows, figures / "heldout_metric_comparison.png"
    )

    print("Held-out covariance-model comparison")
    print(
        "model | ell_x [m] | ell_y [m] | ell_x/ell_y | variogram RMSE | "
        "heldout RMSE | 90% coverage"
    )
    for row in comparison_rows:
        print(
            f"{row['covariance_model']} | {row['length_scale_x_m']:.3g} | "
            f"{row['length_scale_y_m']:.3g} | {row['ell_x_over_ell_y']:.3g} | "
            f"{row['variogram_rmse']:.4g} | {row['log10_mean_rmse']:.4g} | "
            f"{row['gaussian_90pct_coverage']:.3f}"
        )
    print(f"Saved calibration inspection to {root}")
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
    calibrate.add_argument(
        "--models",
        nargs="+",
        default=["matern32", "exponential", "radial_exponential"],
    )
    calibrate.add_argument("--max-lag-cells", type=int, default=64)
    calibrate.add_argument("--spatial-stride", type=int, default=1)
    calibrate.add_argument("--plots-dir")
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

    evaluate = sub.add_parser(
        "evaluate",
        help=(
            "leave one real field out, calibrate on the remaining fields, and compare "
            "all covariance candidates by variograms and conditional reconstruction"
        ),
    )
    evaluate.add_argument("--fields", required=True)
    evaluate.add_argument("--key")
    evaluate.add_argument("--cell-size-m", type=float, required=True)
    evaluate.add_argument(
        "--models",
        nargs="+",
        default=["matern32", "exponential", "radial_exponential"],
    )
    evaluate.add_argument("--truth-index", type=int, default=0)
    evaluate.add_argument("--max-lag-cells", type=int, default=96)
    evaluate.add_argument("--spatial-stride", type=int, default=4)
    evaluate.add_argument(
        "--evaluation-stride",
        type=int,
        default=4,
        help=(
            "Downsample only the held-out conditional reconstruction grid by this "
            "integer stride; physical distances and fitted length scales stay in metres."
        ),
    )
    evaluate.add_argument("--n-boreholes", type=int, default=30)
    evaluate.add_argument("--borehole-seed", type=int, default=2907)
    evaluate.add_argument("--margin-cells", type=int, default=10)
    evaluate.add_argument("--min-spacing-cells", type=float, default=20.0)
    evaluate.add_argument(
        "--observation-std-log10-k",
        type=float,
        default=0.0,
        help="Observation-noise standard deviation in log10(K); 0 gives exact boreholes.",
    )
    evaluate.add_argument(
        "--kriging-chunk-rows",
        type=int,
        default=64,
        help=(
            "Rows evaluated per full-covariance kriging block. This controls memory "
            "only and does not approximate the covariance model."
        ),
    )
    # Backwards-compatible no-op options retained so previously copied commands
    # still run. Covariance-family evaluation no longer uses KL truncation or MC.
    evaluate.add_argument("--n-samples", type=int, default=None, help=argparse.SUPPRESS)
    evaluate.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    evaluate.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    evaluate.add_argument("--n-modes", type=int, default=None, help=argparse.SUPPRESS)
    evaluate.add_argument("--energy-threshold", type=float, default=None, help=argparse.SUPPRESS)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
