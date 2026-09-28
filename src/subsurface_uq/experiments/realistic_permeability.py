from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from ..sampling import (
    ConditionalKLLogGaussianPermeabilityMap,
    ContinuousPointConditionalKLLogGaussianPermeabilityMap,
    GEOREFERENCE_SWEEP_REPRESENTATIONS,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
    RadialExponentialPermeabilitySampler,
    calibrate_covariance_candidates,
    correlation_for_offsets,
    exact_simple_kriging_posterior,
    estimate_directional_variograms,
    load_empirical_fields,
    load_release25_raw_permeability_dataset,
    load_release25_raw_permeability_run,
    load_reference_permeability_surface,
    load_reference_permeability_sweep_surfaces,
    infer_lgcnn_domain_georeference,
    summarize_georeference_sweep,
    load_reference_horizontal_grid,
    load_munich_hydraulic_conductivity_measurements,
    aggregate_measurements_by_reference_cell,
    exact_simple_kriging_grid_from_points,
    hydraulic_conductivity_to_intrinsic_permeability,
    calibrate_point_covariance_candidates,
    sample_borehole_observations,
    select_new_lgcnn_domain,
)
from ..validation.measurements import spatial_block_cross_validate_measurements
from ..visualization.realistic_permeability import (
    plot_georeference_alignment,
    plot_heldout_metric_comparison,
    plot_heldout_reconstruction,
    plot_length_scale_comparison,
    plot_measurement_cv_comparison,
    plot_new_domain_summary,
    plot_nugget_fraction_comparison,
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


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty CSV rows")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _measurement_evaluate(args: argparse.Namespace) -> int:
    reference_grid = load_reference_horizontal_grid(
        args.reference_grid, expected_cell_size_m=args.expected_cell_size_m
    )
    measurements, qc = load_munich_hydraulic_conductivity_measurements(
        args.measurements,
        sheet_name=args.sheet_name,
        stratigraphy=args.stratigraphy,
        groundwater_state=args.groundwater_state,
        reference_grid=reference_grid,
    )
    root = Path(args.output_dir).expanduser().resolve()
    figures = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    _write_rows_csv(root / "measurements_filtered.csv", measurements.to_rows())
    _write_rows_csv(
        root / "measurements_by_reference_cell.csv",
        aggregate_measurements_by_reference_cell(measurements),
    )

    calibration, variograms, pair_counts = calibrate_point_covariance_candidates(
        measurements.coordinates_xy_m,
        measurements.hydraulic_conductivity_m_s,
        models=args.models,
        lag_bin_m=args.lag_bin_m,
        max_lag_m=args.max_lag_m,
        angle_tolerance_deg=args.angle_tolerance_deg,
        min_pairs_per_bin=args.min_pairs_per_bin,
        fit_nugget=not args.disable_nugget,
        pair_count_weighted_fit=not args.disable_pair_count_weighting,
    )
    plot_variogram_fits(
        variograms, calibration, figures / "measurement_variogram_fits.png"
    )
    plot_length_scale_comparison(
        calibration, figures / "measurement_length_scales.png"
    )
    plot_nugget_fraction_comparison(
        calibration, figures / "measurement_nugget_fraction.png"
    )
    np.savez_compressed(
        root / "measurement_variograms.npz",
        x_distance_m=variograms["x"][0],
        x_semivariance=variograms["x"][1],
        x_pair_count=pair_counts["x"],
        y_distance_m=variograms["y"][0],
        y_semivariance=variograms["y"][1],
        y_pair_count=pair_counts["y"],
        diag_distance_m=variograms["diag"][0],
        diag_semivariance=variograms["diag"][1],
        diag_pair_count=pair_counts["diag"],
    )

    cv = spatial_block_cross_validate_measurements(
        measurements.coordinates_xy_m,
        measurements.hydraulic_conductivity_m_s,
        models=args.models,
        n_folds=args.cv_folds,
        block_size_m=args.cv_block_size_m,
        fold_seed=args.cv_seed,
        lag_bin_m=args.lag_bin_m,
        max_lag_m=args.max_lag_m,
        angle_tolerance_deg=args.angle_tolerance_deg,
        min_pairs_per_bin=args.min_pairs_per_bin,
        observation_std_log10_k=args.observation_std_log10_k,
        fit_nugget=not args.disable_nugget,
        pair_count_weighted_fit=not args.disable_pair_count_weighting,
    )
    _write_rows_csv(root / "measurement_cv_summary.csv", list(cv.summary_rows))
    _write_rows_csv(root / "measurement_cv_folds.csv", list(cv.fold_rows))
    plot_measurement_cv_comparison(
        cv.summary_rows, figures / "measurement_cv_comparison.png"
    )

    full_by_model = {item.covariance_model: item for item in calibration}
    cv_by_model = {str(row["covariance_model"]): row for row in cv.summary_rows}
    comparison_rows: list[dict[str, object]] = []
    for model in args.models:
        result = full_by_model[model]
        cv_row = cv_by_model[model]
        comparison_rows.append(
            {
                "covariance_model": model,
                "full_variogram_rmse": result.variogram_rmse,
                "full_weighted_variogram_rmse": result.variogram_weighted_rmse,
                "structured_std_log10_k": result.structured_std_log10_k,
                "nugget_std_log10_k": result.nugget_std_log10_k,
                "nugget_fraction": result.nugget_fraction,
                "length_scale_x_m": result.length_scale_x_m,
                "length_scale_y_m": result.length_scale_y_m,
                "ell_x_over_ell_y": result.length_scale_x_m / result.length_scale_y_m,
                **{key: value for key, value in cv_row.items() if key != "covariance_model"},
            }
        )
    _write_rows_csv(root / "measurement_model_comparison.csv", comparison_rows)
    selected = min(comparison_rows, key=lambda row: float(row["rmse_log10_k"]))

    robustness_payload: dict[str, object] | None = None
    threshold = float(args.robustness_upper_k_m_s)
    if threshold > 0.0:
        keep = measurements.hydraulic_conductivity_m_s <= threshold
        excluded = int(np.count_nonzero(~keep))
        if excluded > 0 and int(np.count_nonzero(keep)) >= 10:
            robust_measurements = measurements.subset(keep)
            robust_calibration, _, _ = calibrate_point_covariance_candidates(
                robust_measurements.coordinates_xy_m,
                robust_measurements.hydraulic_conductivity_m_s,
                models=args.models,
                lag_bin_m=args.lag_bin_m,
                max_lag_m=args.max_lag_m,
                angle_tolerance_deg=args.angle_tolerance_deg,
                min_pairs_per_bin=args.min_pairs_per_bin,
                fit_nugget=not args.disable_nugget,
                pair_count_weighted_fit=not args.disable_pair_count_weighting,
            )
            robust_cv = spatial_block_cross_validate_measurements(
                robust_measurements.coordinates_xy_m,
                robust_measurements.hydraulic_conductivity_m_s,
                models=args.models,
                n_folds=args.cv_folds,
                block_size_m=args.cv_block_size_m,
                fold_seed=args.cv_seed,
                lag_bin_m=args.lag_bin_m,
                max_lag_m=args.max_lag_m,
                angle_tolerance_deg=args.angle_tolerance_deg,
                min_pairs_per_bin=args.min_pairs_per_bin,
                observation_std_log10_k=args.observation_std_log10_k,
                fit_nugget=not args.disable_nugget,
                pair_count_weighted_fit=not args.disable_pair_count_weighting,
                fold_assignment=cv.fold_assignment[keep],
            )
            robust_cal_by_model = {
                item.covariance_model: item for item in robust_calibration
            }
            robust_cv_by_model = {
                str(row["covariance_model"]): row for row in robust_cv.summary_rows
            }
            robustness_rows: list[dict[str, object]] = []
            baseline_by_model = {
                str(row["covariance_model"]): row for row in comparison_rows
            }
            for model in args.models:
                base = baseline_by_model[model]
                fitted = robust_cal_by_model[model]
                cv_row = robust_cv_by_model[model]
                robustness_rows.append(
                    {
                        "scenario": "all_measurements",
                        "covariance_model": model,
                        "n_measurements": len(measurements),
                        "rmse_log10_k": base["rmse_log10_k"],
                        "mae_log10_k": base["mae_log10_k"],
                        "coverage_90": base["coverage_90"],
                        "gaussian_nlpd": base["gaussian_nlpd"],
                        "length_scale_x_m": base["length_scale_x_m"],
                        "length_scale_y_m": base["length_scale_y_m"],
                        "nugget_fraction": base["nugget_fraction"],
                    }
                )
                robustness_rows.append(
                    {
                        "scenario": "exclude_above_threshold",
                        "covariance_model": model,
                        "n_measurements": len(robust_measurements),
                        "rmse_log10_k": cv_row["rmse_log10_k"],
                        "mae_log10_k": cv_row["mae_log10_k"],
                        "coverage_90": cv_row["coverage_90"],
                        "gaussian_nlpd": cv_row["gaussian_nlpd"],
                        "length_scale_x_m": fitted.length_scale_x_m,
                        "length_scale_y_m": fitted.length_scale_y_m,
                        "nugget_fraction": fitted.nugget_fraction,
                        "rmse_change_percent": 100.0
                        * (float(cv_row["rmse_log10_k"]) - float(base["rmse_log10_k"]))
                        / float(base["rmse_log10_k"]),
                        "length_scale_x_change_percent": 100.0
                        * (fitted.length_scale_x_m - float(base["length_scale_x_m"]))
                        / float(base["length_scale_x_m"]),
                        "length_scale_y_change_percent": 100.0
                        * (fitted.length_scale_y_m - float(base["length_scale_y_m"]))
                        / float(base["length_scale_y_m"]),
                        "nugget_fraction_change": fitted.nugget_fraction
                        - float(base["nugget_fraction"]),
                    }
                )
            _write_rows_csv(
                root / "measurement_upper_tail_robustness.csv", robustness_rows
            )
            robustness_payload = {
                "threshold_m_s": threshold,
                "excluded_measurements": excluded,
                "retained_measurements": len(robust_measurements),
                "rows": robustness_rows,
            }

    payload = {
        "schema_version": 2,
        "source": {
            "measurements": str(Path(args.measurements).expanduser().resolve()),
            "reference_grid": str(Path(args.reference_grid).expanduser().resolve()),
            "sheet_name": args.sheet_name,
        },
        "measurement_qc": qc,
        "reference_grid": {
            "x_min_m": reference_grid.x_min_m,
            "y_min_m": reference_grid.y_min_m,
            "cell_size_x_m": reference_grid.cell_size_x_m,
            "cell_size_y_m": reference_grid.cell_size_y_m,
            "nx": reference_grid.nx,
            "ny": reference_grid.ny,
            "active_horizontal_cells": int(np.count_nonzero(reference_grid.active_mask)),
        },
        "variogram_settings": {
            "lag_bin_m": float(args.lag_bin_m),
            "max_lag_m": float(args.max_lag_m),
            "angle_tolerance_deg": float(args.angle_tolerance_deg),
            "min_pairs_per_bin": int(args.min_pairs_per_bin),
            "fit_nugget": not args.disable_nugget,
            "pair_count_weighted_fit": not args.disable_pair_count_weighting,
        },
        "spatial_cv": {
            "n_folds": int(args.cv_folds),
            "block_size_m": float(args.cv_block_size_m),
            "seed": int(args.cv_seed),
            "summary": list(cv.summary_rows),
        },
        "candidates_full_data": [item.to_dict() for item in calibration],
        "selected_by_spatial_cv_rmse": selected,
        "upper_tail_robustness": robustness_payload,
        "nugget_interpretation": (
            "The fitted nugget is an effective unresolved short-scale component. "
            "With the available source data it cannot be uniquely separated into "
            "measurement error, sub-bin spatial variability, and true microscale heterogeneity."
        ),
        "selection_note": (
            "The selected entry is the lowest spatial-block-CV RMSE diagnostic, "
            "not a claim that the covariance family is uniquely geologically correct."
        ),
        "units_note": (
            "Calibration and CV use measured hydraulic conductivity K [m/s] in log10 space. "
            "For constant fluid properties, conversion to intrinsic permeability shifts log10(K) "
            "by a constant and leaves covariance length scales/variograms unchanged."
        ),
        "crs_assumption": (
            "DHDN / Gauss-Kruger zone 4 (EPSG:31468), inferred from the source coordinates "
            "and pending external confirmation."
        ),
    }
    with (root / "measurement_calibration.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    with (root / "measurement_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"Accepted real measurements: {len(measurements)}")
    print(
        "model | weighted variogram RMSE | nugget fraction | spatial-CV RMSE | "
        "MAE | 90% coverage | NLPD"
    )
    for row in comparison_rows:
        print(
            f"{row['covariance_model']} | {row['full_weighted_variogram_rmse']:.4g} | "
            f"{row['nugget_fraction']:.3f} | {row['rmse_log10_k']:.4g} | "
            f"{row['mae_log10_k']:.4g} | "
            f"{row['coverage_90']:.3f} | {row['gaussian_nlpd']:.4g}"
        )
    print(f"Saved real-measurement calibration to {root}")
    return 0


def _measurement_condition(args: argparse.Namespace) -> int:
    calibration_path = Path(args.calibration).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    candidates = {
        item["covariance_model"]: item for item in payload["candidates_full_data"]
    }
    model = args.model
    if model is None:
        model = str(payload["selected_by_spatial_cv_rmse"]["covariance_model"])
    if model not in candidates:
        raise ValueError(f"model {model!r} is not present in measurement calibration")
    selected = candidates[model]

    reference_grid = load_reference_horizontal_grid(
        args.reference_grid, expected_cell_size_m=args.expected_cell_size_m
    )
    measurements, qc = load_munich_hydraulic_conductivity_measurements(
        args.measurements,
        sheet_name=args.sheet_name,
        stratigraphy=args.stratigraphy,
        groundwater_state=args.groundwater_state,
        reference_grid=reference_grid,
    )
    posterior = exact_simple_kriging_grid_from_points(
        observation_coordinates_xy_m=measurements.coordinates_xy_m,
        observation_log10_k=measurements.log10_hydraulic_conductivity,
        x_min_m=reference_grid.x_min_m,
        y_min_m=reference_grid.y_min_m,
        nx=reference_grid.nx,
        ny=reference_grid.ny,
        cell_size_x_m=reference_grid.cell_size_x_m,
        cell_size_y_m=reference_grid.cell_size_y_m,
        covariance_model=model,
        mean_log10_k=float(selected["mean_log10_k"]),
        std_log10_k=float(selected["std_log10_k"]),
        length_scale_y_m=float(selected["length_scale_y_m"]),
        length_scale_x_m=float(selected["length_scale_x_m"]),
        observation_std_log10_k=args.observation_std_log10_k,
        structured_std_log10_k=float(
            selected.get("structured_std_log10_k") or selected["std_log10_k"]
        ),
        nugget_std_log10_k=float(selected.get("nugget_std_log10_k", 0.0)),
        include_query_nugget=False,
        angle_rad=float(selected.get("angle_rad", 0.0)),
        active_mask=reference_grid.active_mask,
        chunk_size=args.kriging_chunk_size,
    )
    mean_log = np.asarray(posterior.mean_log10_k, dtype=np.float64)
    structured_variance_log = np.asarray(
        posterior.variance_log10_k, dtype=np.float64
    )
    nugget_variance_log = float(selected.get("nugget_std_log10_k", 0.0)) ** 2
    predictive_variance_log = structured_variance_log + nugget_variance_log
    active = reference_grid.active_mask
    median_kh = np.full(mean_log.shape, np.nan, dtype=np.float64)
    mean_structured_kh = np.full(mean_log.shape, np.nan, dtype=np.float64)
    mean_predictive_kh = np.full(mean_log.shape, np.nan, dtype=np.float64)
    median_kh[active] = np.power(10.0, mean_log[active])
    mean_structured_kh[active] = (
        np.power(10.0, mean_log[active])
        * np.exp(
            0.5
            * (np.log(10.0) ** 2)
            * structured_variance_log[active]
        )
    )
    mean_predictive_kh[active] = (
        np.power(10.0, mean_log[active])
        * np.exp(
            0.5
            * (np.log(10.0) ** 2)
            * predictive_variance_log[active]
        )
    )
    factor = float(
        hydraulic_conductivity_to_intrinsic_permeability(
            np.asarray([1.0]),
            dynamic_viscosity_pa_s=args.dynamic_viscosity_pa_s,
            density_kg_m3=args.density_kg_m3,
            gravity_m_s2=args.gravity_m_s2,
        )[0]
    )
    median_intrinsic = median_kh * factor
    mean_structured_intrinsic = mean_structured_kh * factor
    mean_predictive_intrinsic = mean_predictive_kh * factor

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        posterior_mean_log10_hydraulic_conductivity=posterior.mean_log10_k,
        posterior_structured_std_log10_hydraulic_conductivity=posterior.std_log10_k,
        posterior_structured_variance_log10_hydraulic_conductivity=posterior.variance_log10_k,
        posterior_predictive_std_log10_hydraulic_conductivity=np.sqrt(
            predictive_variance_log
        ).astype(np.float32),
        posterior_predictive_variance_log10_hydraulic_conductivity=predictive_variance_log.astype(
            np.float32
        ),
        posterior_median_hydraulic_conductivity_m_s=median_kh.astype(np.float32),
        posterior_mean_structured_hydraulic_conductivity_m_s=mean_structured_kh.astype(
            np.float32
        ),
        posterior_mean_predictive_hydraulic_conductivity_m_s=mean_predictive_kh.astype(
            np.float32
        ),
        posterior_median_intrinsic_permeability_m2=median_intrinsic.astype(np.float32),
        posterior_mean_structured_intrinsic_permeability_m2=mean_structured_intrinsic.astype(
            np.float32
        ),
        posterior_mean_predictive_intrinsic_permeability_m2=mean_predictive_intrinsic.astype(
            np.float32
        ),
        active_mask=active,
    )
    metadata = {
        "schema_version": 1,
        "model": model,
        "calibration": selected,
        "measurement_qc": qc,
        "conditioning_measurements": len(measurements),
        "nugget_interpretation": (
            "The fitted nugget is treated as unresolved microscale/measurement-scale "
            "variance. The gridded posterior mean/structured variance exclude an "
            "independent nugget realization; predictive variance additionally includes it."
        ),
        "conditioning_max_abs_log10_residual": posterior.conditioning_max_abs_log10_residual,
        "conditioning_rms_log10_residual": posterior.conditioning_rms_log10_residual,
        "fluid_conversion": {
            "dynamic_viscosity_pa_s": float(args.dynamic_viscosity_pa_s),
            "density_kg_m3": float(args.density_kg_m3),
            "gravity_m_s2": float(args.gravity_m_s2),
            "intrinsic_permeability_factor_m_s_to_m2": factor,
            "note": "Default constants correspond approximately to liquid water near 20 degC.",
        },
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(f"Conditioned {model} posterior on {len(measurements)} real measurements")
    print(f"Saved posterior grid to {output}")
    return 0


def _georeference_domain(args: argparse.Namespace) -> int:
    raw_field = load_release25_raw_permeability_run(
        args.raw_dataset,
        args.run,
        cell_size_m=args.raw_cell_size_m,
    )
    reference = load_reference_permeability_surface(
        args.reference_grid,
        column=args.reference_column,
        z_mode=args.reference_z_mode,
        z_value_m=args.reference_z_m,
    )
    mapping = infer_lgcnn_domain_georeference(
        raw_field,
        reference,
        run_name=args.run,
        raw_cell_size_m=args.raw_cell_size_m,
        coarse_anchor_stride=args.coarse_anchor_stride,
        coarse_keep_per_transform=args.coarse_keep_per_transform,
        refine_radius_m=args.refine_radius_m,
        refine_step_m=args.refine_step_m,
        refine_sample_stride_cells=args.refine_sample_stride_cells,
        min_reference_coverage=args.min_reference_coverage,
        min_correlation=args.min_correlation,
        max_centered_rmse_log10=args.max_centered_rmse_log10,
        dynamic_viscosity_pa_s=args.dynamic_viscosity_pa_s,
        density_kg_m3=args.density_kg_m3,
        gravity_m_s2=args.gravity_m_s2,
    )
    payload = {
        **mapping.to_dict(),
        "source": {
            "raw_dataset": str(Path(args.raw_dataset).expanduser().resolve()),
            "reference_grid": str(Path(args.reference_grid).expanduser().resolve()),
        },
        "inference": {
            "coarse_anchor_stride": int(args.coarse_anchor_stride),
            "coarse_keep_per_transform": int(args.coarse_keep_per_transform),
            "refine_radius_m": float(args.refine_radius_m),
            "refine_step_m": (
                None if args.refine_step_m is None else float(args.refine_step_m)
            ),
            "refine_sample_stride_cells": int(args.refine_sample_stride_cells),
            "min_reference_coverage": float(args.min_reference_coverage),
            "min_correlation": float(args.min_correlation),
            "max_centered_rmse_log10": float(args.max_centered_rmse_log10),
        },
        "release25_domain": {
            "cell_size_m": float(args.raw_cell_size_m),
            "expected_standard_shape": [2560, 2560],
            "expected_standard_size_m": [12800.0, 12800.0],
            "note": (
                "DARUS-5065 standard real-permeability data points are 12.8 km x "
                "12.8 km with 2560 x 2560 cells at 5 m resolution."
            ),
        },
    }

    if args.measurements:
        grid = load_reference_horizontal_grid(
            args.reference_grid, expected_cell_size_m=args.expected_reference_cell_size_m
        )
        measurements, qc = load_munich_hydraulic_conductivity_measurements(
            args.measurements,
            sheet_name=args.sheet_name,
            stratigraphy=args.stratigraphy,
            groundwater_state=args.groundwater_state,
            reference_grid=grid,
        )
        inside = mapping.contains_xy(measurements.x_m, measurements.y_m)
        rows, cols = mapping.xy_to_fractional_indices(
            measurements.x_m[inside], measurements.y_m[inside]
        )
        payload["measurement_overlap"] = {
            "accepted_measurements_total": len(measurements),
            "measurements_inside_domain": int(np.count_nonzero(inside)),
            "measurement_fraction_inside_domain": float(np.mean(inside)),
            "conditioning_coordinates_are_continuous": True,
            "fractional_row_range": (
                [float(np.min(rows)), float(np.max(rows))] if rows.size else None
            ),
            "fractional_col_range": (
                [float(np.min(cols)), float(np.max(cols))] if cols.size else None
            ),
            "measurement_qc": qc,
        }

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)

    if args.plots_dir:
        plots_dir = Path(args.plots_dir).expanduser().resolve()
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_georeference_alignment(
            reference=reference,
            raw_field=raw_field,
            mapping=mapping,
            destination=plots_dir / f"georeference_{args.run}.png",
        )

    print("LGCNN real-domain georeference")
    print(f"  run: {mapping.run_name}")
    print(f"  transform: {mapping.transform}")
    print(
        "  first cell center [m]: "
        f"x={mapping.first_cell_center_x_m:.3f}, "
        f"y={mapping.first_cell_center_y_m:.3f}"
    )
    print(
        "  domain edges [m]: "
        f"W={mapping.west_edge_m:.3f}, S={mapping.south_edge_m:.3f}, "
        f"E={mapping.east_edge_m:.3f}, N={mapping.north_edge_m:.3f}"
    )
    print(
        "  spatial match: "
        f"corr={mapping.correlation:.5f}, "
        f"centered RMSE={mapping.centered_rmse_log10:.5f} log10, "
        f"coverage={mapping.reference_coverage_fraction:.3f}"
    )
    print(
        "  log10(raw/reference) shift: "
        f"{mapping.log10_unit_shift_raw_minus_reference:.6f} "
        f"({mapping.unit_shift_interpretation})"
    )
    print(f"  validated: {mapping.validated}")
    print(f"Saved georeference manifest to {output}")
    if args.require_validated and not mapping.validated:
        raise RuntimeError(
            "geospatial match did not satisfy validation thresholds; inspect the "
            "reference surface/z mode and do not use this manifest for stochastic generation"
        )
    return 0


def _georeference_sweep(args: argparse.Namespace) -> int:
    run_names = tuple(dict.fromkeys(str(run) for run in args.runs))
    if len(run_names) < 2:
        raise ValueError("georeference-sweep requires at least two distinct RUN_n values")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = (
        None
        if args.plots_dir is None
        else Path(args.plots_dir).expanduser().resolve()
    )
    if plots_dir is not None:
        plots_dir.mkdir(parents=True, exist_ok=True)

    raw_fields = {
        run: load_release25_raw_permeability_run(
            args.raw_dataset,
            run,
            cell_size_m=args.raw_cell_size_m,
        )
        for run in run_names
    }
    reference_surfaces = load_reference_permeability_sweep_surfaces(
        args.reference_grid
    )

    measurements = None
    measurement_qc = None
    if args.measurements:
        grid = load_reference_horizontal_grid(
            args.reference_grid,
            expected_cell_size_m=args.expected_reference_cell_size_m,
        )
        measurements, measurement_qc = load_munich_hydraulic_conductivity_measurements(
            args.measurements,
            sheet_name=args.sheet_name,
            stratigraphy=args.stratigraphy,
            groundwater_state=args.groundwater_state,
            reference_grid=grid,
        )

    rows: list[dict[str, object]] = []
    mapping_by_key: dict[tuple[str, str, str], object] = {}
    for column, z_mode in GEOREFERENCE_SWEEP_REPRESENTATIONS:
        reference = reference_surfaces[(column, z_mode)]
        for run in run_names:
            row: dict[str, object] = {
                "run_name": run,
                "reference_column": column,
                "reference_z_mode": z_mode,
                "error": None,
            }
            try:
                mapping = infer_lgcnn_domain_georeference(
                    raw_fields[run],
                    reference,
                    run_name=run,
                    raw_cell_size_m=args.raw_cell_size_m,
                    coarse_anchor_stride=args.coarse_anchor_stride,
                    coarse_keep_per_transform=args.coarse_keep_per_transform,
                    refine_radius_m=args.refine_radius_m,
                    refine_step_m=args.refine_step_m,
                    refine_sample_stride_cells=args.refine_sample_stride_cells,
                    min_reference_coverage=args.min_reference_coverage,
                    min_correlation=args.min_correlation,
                    max_centered_rmse_log10=args.max_centered_rmse_log10,
                    dynamic_viscosity_pa_s=args.dynamic_viscosity_pa_s,
                    density_kg_m3=args.density_kg_m3,
                    gravity_m_s2=args.gravity_m_s2,
                )
                mapping_by_key[(run, column, z_mode)] = mapping
                row.update(
                    {
                        "transform": mapping.transform,
                        "first_cell_center_x_m": mapping.first_cell_center_x_m,
                        "first_cell_center_y_m": mapping.first_cell_center_y_m,
                        "west_edge_m": mapping.west_edge_m,
                        "south_edge_m": mapping.south_edge_m,
                        "east_edge_m": mapping.east_edge_m,
                        "north_edge_m": mapping.north_edge_m,
                        "correlation": mapping.correlation,
                        "centered_rmse_log10": mapping.centered_rmse_log10,
                        "reference_coverage_fraction": mapping.reference_coverage_fraction,
                        "log10_unit_shift_raw_minus_reference": (
                            mapping.log10_unit_shift_raw_minus_reference
                        ),
                        "unit_shift_interpretation": mapping.unit_shift_interpretation,
                        "matched_points": mapping.matched_points,
                        "validated": mapping.validated,
                    }
                )
                if measurements is not None:
                    inside = mapping.contains_xy(measurements.x_m, measurements.y_m)
                    row["measurements_inside_domain"] = int(np.count_nonzero(inside))
                    row["measurement_fraction_inside_domain"] = float(np.mean(inside))
                if plots_dir is not None:
                    representation_dir = plots_dir / f"{column}_{z_mode}"
                    representation_dir.mkdir(parents=True, exist_ok=True)
                    plot_georeference_alignment(
                        reference=reference,
                        raw_field=raw_fields[run],
                        mapping=mapping,
                        destination=representation_dir / f"georeference_{run}.png",
                    )
            except Exception as exc:
                row.update(
                    {
                        "transform": None,
                        "correlation": None,
                        "centered_rmse_log10": None,
                        "reference_coverage_fraction": None,
                        "log10_unit_shift_raw_minus_reference": None,
                        "unit_shift_interpretation": None,
                        "validated": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            rows.append(row)

    for run in run_names:
        successful = [
            row
            for row in rows
            if row["run_name"] == run and not row.get("error")
        ]
        successful.sort(
            key=lambda row: (
                not bool(row.get("validated", False)),
                -float(row["correlation"]),
                float(row["centered_rmse_log10"]),
            )
        )
        for rank, row in enumerate(successful, start=1):
            row["run_rank"] = rank

    summary = summarize_georeference_sweep(
        rows,
        run_names=run_names,
        max_unit_shift_spread_log10=args.max_unit_shift_spread_log10,
    )
    comparison_rows = list(summary["rows"])
    _write_rows_csv(output_dir / "georeference_sweep.csv", comparison_rows)

    summary_payload = {
        "schema_version": 1,
        "source": {
            "raw_dataset": str(Path(args.raw_dataset).expanduser().resolve()),
            "reference_grid": str(Path(args.reference_grid).expanduser().resolve()),
            "measurements": (
                None
                if args.measurements is None
                else str(Path(args.measurements).expanduser().resolve())
            ),
        },
        "runs": list(run_names),
        "representations": [
            {"reference_column": column, "reference_z_mode": z_mode}
            for column, z_mode in GEOREFERENCE_SWEEP_REPRESENTATIONS
        ],
        "single_run_validation_thresholds": {
            "min_reference_coverage": float(args.min_reference_coverage),
            "min_correlation": float(args.min_correlation),
            "max_centered_rmse_log10": float(args.max_centered_rmse_log10),
        },
        "cross_run_consistency_thresholds": {
            "max_unit_shift_spread_log10": float(args.max_unit_shift_spread_log10),
            "require_common_transform": True,
            "require_common_unit_shift_interpretation": True,
            "require_all_runs_individually_validated": True,
        },
        "measurement_qc": measurement_qc,
        **{key: value for key, value in summary.items() if key != "rows"},
    }
    with (output_dir / "georeference_sweep_summary.yaml").open(
        "w", encoding="utf-8"
    ) as handle:
        yaml.safe_dump(summary_payload, handle, sort_keys=False)

    selected = summary["selected_representation"]
    print("Georeference sweep")
    print(f"  runs: {', '.join(run_names)}")
    print("  representations: 9 (K_P10/K_P50/K_P90 x top/bottom/log_geomean)")
    print(
        "  consistent, defensible mapping exists: "
        f"{summary['consistent_defensible_mapping_exists']}"
    )
    print(
        "  unique defensible reference representation: "
        f"{summary['unique_defensible_representation']}"
    )
    if selected is None:
        print(
            "  selected representation: none; do not assign exact Munich coordinates "
            "to the LGCNN runs from this reference table"
        )
    else:
        print(
            "  selected representation: "
            f"{selected['reference_column']} / {selected['reference_z_mode']}"
        )
        print(
            "  cross-run diagnostics: "
            f"mean corr={selected['mean_correlation']:.4f}, "
            f"max RMSE={selected['max_centered_rmse_log10']:.4f}, "
            f"unit-shift spread={selected['unit_shift_spread_log10']:.4f}, "
            f"transform={selected['consistent_transform']}"
        )
    print(f"Saved comparison CSV to {output_dir / 'georeference_sweep.csv'}")
    print(
        f"Saved decision summary to "
        f"{output_dir / 'georeference_sweep_summary.yaml'}"
    )
    if args.require_defensible and not summary["consistent_defensible_mapping_exists"]:
        raise RuntimeError(
            "no consistent, defensible mapping exists across the requested runs and "
            "nine reference representations; do not use an inferred exact Munich "
            "crop for stochastic generation"
        )
    return 0


def _new_domain_generate(args: argparse.Namespace) -> int:
    calibration_path = Path(args.calibration).expanduser().resolve()
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration_payload = yaml.safe_load(handle)
    candidates = {
        str(item["covariance_model"]): item
        for item in calibration_payload["candidates_full_data"]
    }
    model = args.model
    if model is None:
        model = str(calibration_payload["selected_by_spatial_cv_rmse"]["covariance_model"])
    if model not in {"matern32", "exponential"}:
        raise ValueError(
            "new-domain KL generation currently supports the separable matern32 or "
            "exponential covariance; select --model exponential for the current Munich fit"
        )
    if model not in candidates:
        raise ValueError(f"model {model!r} is not present in measurement calibration")
    selected = candidates[model]

    reference_grid = load_reference_horizontal_grid(
        args.reference_grid, expected_cell_size_m=args.expected_reference_cell_size_m
    )
    measurements, measurement_qc = load_munich_hydraulic_conductivity_measurements(
        args.measurements,
        sheet_name=args.sheet_name,
        stratigraphy=args.stratigraphy,
        groundwater_state=args.groundwater_state,
        reference_grid=reference_grid,
    )
    domain, inside = select_new_lgcnn_domain(
        measurements.coordinates_xy_m,
        domain_size_m=args.domain_size_m,
        cell_size_m=args.cell_size_m,
        west_edge_m=args.domain_origin_x_m,
        south_edge_m=args.domain_origin_y_m,
    )
    selected_measurements = measurements.subset(inside)
    if len(selected_measurements) < int(args.min_conditioning_measurements):
        raise ValueError(
            f"selected domain contains only {len(selected_measurements)} measurements; "
            f"minimum requested is {args.min_conditioning_measurements}"
        )

    reference_rows, reference_cols = np.indices(reference_grid.shape)
    reference_x = (
        reference_grid.x_min_m
        + reference_cols.astype(np.float64) * reference_grid.cell_size_x_m
    )
    reference_y = (
        reference_grid.y_min_m
        + reference_rows.astype(np.float64) * reference_grid.cell_size_y_m
    )
    reference_in_domain = domain.contains_xy(reference_x, reference_y)
    active_reference_in_domain = reference_in_domain & reference_grid.active_mask
    expected_reference_cells = max(
        domain.size_m[0]
        * domain.size_m[1]
        / (reference_grid.cell_size_x_m * reference_grid.cell_size_y_m),
        1.0,
    )
    active_reference_coverage = min(
        1.0,
        float(np.count_nonzero(active_reference_in_domain) / expected_reference_cells),
    )

    conversion_factor = float(
        hydraulic_conductivity_to_intrinsic_permeability(
            np.asarray([1.0]),
            dynamic_viscosity_pa_s=args.dynamic_viscosity_pa_s,
            density_kg_m3=args.density_kg_m3,
            gravity_m_s2=args.gravity_m_s2,
        )[0]
    )
    log10_shift = float(np.log10(conversion_factor))
    structured_std = float(
        selected.get("structured_std_log10_k") or selected["std_log10_k"]
    )
    nugget_std = float(selected.get("nugget_std_log10_k", 0.0))
    extra_std = float(args.observation_std_log10_k)
    if extra_std < 0.0:
        raise ValueError("observation-std-log10-k must be non-negative")
    effective_observation_std = float(np.sqrt(nugget_std**2 + extra_std**2))
    if effective_observation_std <= 0.0:
        raise ValueError(
            "continuous new-domain conditioning needs positive nugget/measurement noise"
        )

    prior = KLLogGaussianPermeabilityMap(
        shape=domain.shape,
        domain_size_m=domain.size_m,
        mean_log10_k=float(selected["mean_log10_k"]) + log10_shift,
        std_log10_k=structured_std,
        length_scale_m=(
            float(selected["length_scale_y_m"]),
            float(selected["length_scale_x_m"]),
        ),
        covariance_model=model,
        n_modes=args.n_modes,
        energy_threshold=args.energy_threshold,
    )

    observation_local_yx = domain.projected_xy_to_local_yx(
        selected_measurements.x_m, selected_measurements.y_m
    )
    observation_log10_intrinsic = (
        selected_measurements.log10_hydraulic_conductivity + log10_shift
    )
    conditional = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
        prior=prior,
        observation_coordinates_yx_m=observation_local_yx,
        observation_log10_k=observation_log10_intrinsic,
        observation_std_log10_k=effective_observation_std,
    )
    sampler = GaussianCoordinatePermeabilitySampler(
        field_map=conditional,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Quantify covariance loss from finite KL truncation at the actual conditioning points.
    A = prior.mode_matrix_at_coordinates(observation_local_yx)
    approximate_covariance = A @ A.T
    delta_y = observation_local_yx[:, 0, None] - observation_local_yx[None, :, 0]
    delta_x = observation_local_yx[:, 1, None] - observation_local_yx[None, :, 1]
    exact_covariance = structured_std**2 * correlation_for_offsets(
        model,
        delta_y_m=delta_y,
        delta_x_m=delta_x,
        length_scale_y_m=float(selected["length_scale_y_m"]),
        length_scale_x_m=float(selected["length_scale_x_m"]),
    )
    covariance_denominator = max(float(np.linalg.norm(exact_covariance)), np.finfo(float).eps)
    conditioning_covariance_relative_error = float(
        np.linalg.norm(approximate_covariance - exact_covariance)
        / covariance_denominator
    )

    latent_mean_obs, latent_std_obs = conditional.posterior_moments_at_points(
        observation_local_yx
    )
    predictive_std_obs = np.sqrt(latent_std_obs**2 + effective_observation_std**2)
    standardized = (observation_log10_intrinsic - latent_mean_obs) / predictive_std_obs
    z90 = 1.6448536269514722
    observation_coverage90 = float(np.mean(np.abs(standardized) <= z90))

    diagnostic_stride = int(args.diagnostic_grid_stride)
    if diagnostic_stride <= 0:
        raise ValueError("diagnostic-grid-stride must be positive")
    rows = np.arange(0, domain.ny, diagnostic_stride, dtype=np.int64)
    cols = np.arange(0, domain.nx, diagnostic_stride, dtype=np.int64)
    gy, gx = np.meshgrid(rows, cols, indexing="ij")
    diagnostic_rows = gy.reshape(-1)
    diagnostic_cols = gx.reshape(-1)
    diagnostic_local_yx = np.column_stack(
        (
            (diagnostic_rows.astype(np.float64) + 0.5) * domain.cell_size_m,
            (diagnostic_cols.astype(np.float64) + 0.5) * domain.cell_size_m,
        )
    )
    analytic_mean_diag, analytic_std_diag = conditional.posterior_moments_at_points(
        diagnostic_local_yx
    )

    root = Path(args.output_dir).expanduser().resolve()
    samples_dir = root / "samples"
    root.mkdir(parents=True, exist_ok=True)
    if args.save_samples:
        samples_dir.mkdir(parents=True, exist_ok=True)

    mean_log = np.zeros(domain.shape, dtype=np.float64)
    m2_log = np.zeros(domain.shape, dtype=np.float64)
    diag_mean = np.zeros(analytic_mean_diag.shape, dtype=np.float64)
    diag_m2 = np.zeros(analytic_mean_diag.shape, dtype=np.float64)
    sample_count = 0
    global_min_k = np.inf
    global_max_k = -np.inf
    outside_total = 0
    total_cells = 0
    outside_by_sample: list[float] = []
    training_min = float(args.training_k_min_m2)
    training_max = float(args.training_k_max_m2)
    if not (0.0 < training_min < training_max):
        raise ValueError("training permeability bounds must satisfy 0 < min < max")

    for batch in sampler:
        for field in np.asarray(batch):
            sample_count += 1
            physical = np.asarray(field, dtype=np.float64)
            log_field = np.log10(physical)
            delta = log_field - mean_log
            mean_log += delta / sample_count
            m2_log += delta * (log_field - mean_log)

            diag_values = log_field[diagnostic_rows, diagnostic_cols]
            diag_delta = diag_values - diag_mean
            diag_mean += diag_delta / sample_count
            diag_m2 += diag_delta * (diag_values - diag_mean)

            global_min_k = min(global_min_k, float(np.min(physical)))
            global_max_k = max(global_max_k, float(np.max(physical)))
            outside = (physical < training_min) | (physical > training_max)
            outside_count = int(np.count_nonzero(outside))
            outside_total += outside_count
            total_cells += int(physical.size)
            outside_by_sample.append(outside_count / physical.size)
            if args.save_samples:
                np.save(
                    samples_dir / f"sample_{sample_count:04d}_permeability_m2.npy",
                    np.asarray(field, dtype=np.float32),
                )

    if sample_count != int(args.n_samples):
        raise RuntimeError(
            f"sampler produced {sample_count} fields, expected {args.n_samples}"
        )
    if sample_count > 1:
        std_log = np.sqrt(m2_log / (sample_count - 1))
        diag_std = np.sqrt(diag_m2 / (sample_count - 1))
    else:
        std_log = np.zeros_like(mean_log)
        diag_std = np.zeros_like(diag_mean)

    np.save(root / "empirical_mean_log10_permeability_m2.npy", mean_log.astype(np.float32))
    np.save(root / "empirical_std_log10_permeability_m2.npy", std_log.astype(np.float32))
    plot_new_domain_summary(
        mean_log10_k=mean_log,
        std_log10_k=std_log,
        domain=domain,
        measurement_x_m=selected_measurements.x_m,
        measurement_y_m=selected_measurements.y_m,
        destination=root / "new_domain_mean_std.png",
    )
    conditioning_rows = selected_measurements.to_rows()
    for row_payload, local_yx in zip(conditioning_rows, observation_local_yx):
        row_payload["new_domain_local_y_m"] = float(local_yx[0])
        row_payload["new_domain_local_x_m"] = float(local_yx[1])
        row_payload["new_domain_fractional_row"] = float(
            local_yx[0] / domain.cell_size_m - 0.5
        )
        row_payload["new_domain_fractional_col"] = float(
            local_yx[1] / domain.cell_size_m - 0.5
        )
    _write_rows_csv(root / "conditioning_measurements.csv", conditioning_rows)

    posterior_reproduction = {
        "diagnostic_point_count": int(analytic_mean_diag.size),
        "diagnostic_grid_stride_cells": diagnostic_stride,
        "empirical_mean_rmse_log10": float(
            np.sqrt(np.mean((diag_mean - analytic_mean_diag) ** 2))
        ),
        "empirical_std_rmse_log10": (
            None
            if sample_count < 2
            else float(np.sqrt(np.mean((diag_std - analytic_std_diag) ** 2)))
        ),
        "analytic_mean_log10_range": [
            float(np.min(analytic_mean_diag)),
            float(np.max(analytic_mean_diag)),
        ],
        "analytic_std_log10_range": [
            float(np.min(analytic_std_diag)),
            float(np.max(analytic_std_diag)),
        ],
    }
    diagnostics = {
        "schema_version": 1,
        "domain": {
            **domain.to_dict(),
            "active_reference_100m_coverage_fraction": active_reference_coverage,
            "active_reference_cells_inside": int(
                np.count_nonzero(active_reference_in_domain)
            ),
            "interpretation": (
                "Reference-grid coverage is a support diagnostic only; the stochastic "
                "field is generated from real-measurement covariance/conditioning, not "
                "from the failed historical DaRUS georeference."
            ),
        },
        "measurement_qc": measurement_qc,
        "conditioning": {
            "measurement_count": len(selected_measurements),
            "measurement_fraction_of_filtered_total": float(len(selected_measurements) / len(measurements)),
            "fitted_nugget_std_log10_k": nugget_std,
            "additional_observation_std_log10_k": extra_std,
            "effective_observation_std_log10_k": effective_observation_std,
            "posterior_predictive_90pct_coverage_at_measurements": observation_coverage90,
            "standardized_residual_mean": float(np.mean(standardized)),
            "standardized_residual_std": float(np.std(standardized, ddof=1)) if standardized.size > 1 else 0.0,
        },
        "kl": {
            "dimension": prior.dimension,
            "retained_energy_fraction": prior.retained_energy_fraction,
            "requested_n_modes": args.n_modes,
            "requested_energy_threshold": float(args.energy_threshold),
            "conditioning_point_covariance_relative_frobenius_error": conditioning_covariance_relative_error,
        },
        "posterior_reproduction": posterior_reproduction,
        "generated_ensemble": {
            "n_samples": sample_count,
            "seed": int(args.seed),
            "minimum_intrinsic_permeability_m2": float(global_min_k),
            "maximum_intrinsic_permeability_m2": float(global_max_k),
            "training_k_min_m2": training_min,
            "training_k_max_m2": training_max,
            "outside_training_fraction": float(outside_total / total_cells),
            "outside_training_fraction_by_sample": outside_by_sample,
        },
        "calibration": selected,
        "hydraulic_to_intrinsic_conversion": {
            "factor": conversion_factor,
            "log10_shift": log10_shift,
            "dynamic_viscosity_pa_s": float(args.dynamic_viscosity_pa_s),
            "density_kg_m3": float(args.density_kg_m3),
            "gravity_m_s2": float(args.gravity_m_s2),
        },
        "array_convention": {
            "stored_samples": "[y,x] projected geographic grid, south-to-north then west-to-east",
            "lgcnn_shape_compatible": list(domain.shape),
            "note": (
                "This new domain does not claim correspondence to a historical DaRUS RUN_n. "
                "Absolute projected coordinates describe the new measurement-conditioned field only."
            ),
        },
        "sampler": sampler.metadata,
    }
    with (root / "new_domain_generator.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(diagnostics, handle, sort_keys=False)
    with (root / "new_domain_generator.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)

    print("Measurement-conditioned new LGCNN domain")
    print(
        f"  domain: W={domain.west_edge_m:.1f}, S={domain.south_edge_m:.1f}, "
        f"E={domain.east_edge_m:.1f}, N={domain.north_edge_m:.1f} m"
    )
    print(f"  grid: {domain.ny} x {domain.nx} at {domain.cell_size_m:g} m")
    print(f"  conditioning measurements: {len(selected_measurements)} / {len(measurements)}")
    print(f"  active 100 m reference coverage: {active_reference_coverage:.2%}")
    print(
        f"  KL modes: {prior.dimension}, retained energy={prior.retained_energy_fraction:.5f}, "
        f"conditioning covariance error={conditioning_covariance_relative_error:.4g}"
    )
    print(
        f"  generated samples: {sample_count}, outside release25 K range="
        f"{diagnostics['generated_ensemble']['outside_training_fraction']:.4%}"
    )
    print(f"Saved new-domain generator outputs to {root}")
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

    measurement_evaluate = sub.add_parser(
        "measurement-evaluate",
        help=(
            "load the real Munich hydraulic-conductivity measurements, fit irregular-point "
            "variograms, and compare covariance models by spatial block cross-validation"
        ),
    )
    measurement_evaluate.add_argument("--measurements", required=True)
    measurement_evaluate.add_argument("--reference-grid", required=True)
    measurement_evaluate.add_argument("--sheet-name", default="kf_werte_180223")
    measurement_evaluate.add_argument("--stratigraphy", default="q")
    measurement_evaluate.add_argument("--groundwater-state", default="ungespannt")
    measurement_evaluate.add_argument(
        "--models",
        nargs="+",
        default=["matern32", "exponential", "radial_exponential"],
    )
    measurement_evaluate.add_argument("--expected-cell-size-m", type=float, default=100.0)
    measurement_evaluate.add_argument("--lag-bin-m", type=float, default=250.0)
    measurement_evaluate.add_argument("--max-lag-m", type=float, default=3000.0)
    measurement_evaluate.add_argument("--angle-tolerance-deg", type=float, default=22.5)
    measurement_evaluate.add_argument("--min-pairs-per-bin", type=int, default=8)
    measurement_evaluate.add_argument("--cv-folds", type=int, default=5)
    measurement_evaluate.add_argument("--cv-block-size-m", type=float, default=2000.0)
    measurement_evaluate.add_argument("--cv-seed", type=int, default=2907)
    measurement_evaluate.add_argument(
        "--observation-std-log10-k",
        type=float,
        default=0.0,
        help=(
            "Additional known measurement-error standard deviation in log10(K_h). "
            "This is separate from the fitted variogram nugget."
        ),
    )
    measurement_evaluate.add_argument(
        "--disable-nugget",
        action="store_true",
        help="Sensitivity option: force the variogram nugget to zero.",
    )
    measurement_evaluate.add_argument(
        "--disable-pair-count-weighting",
        action="store_true",
        help="Sensitivity option: fit variogram bins with equal rather than pair-count weights.",
    )
    measurement_evaluate.add_argument(
        "--robustness-upper-k-m-s",
        type=float,
        default=5.0e-2,
        help=(
            "Repeat calibration/CV after excluding measurements above this K_h threshold; "
            "set <=0 to disable. The baseline fit always retains all measurements."
        ),
    )
    measurement_evaluate.add_argument("--output-dir", required=True)
    measurement_evaluate.set_defaults(func=_measurement_evaluate)

    measurement_condition = sub.add_parser(
        "measurement-condition",
        help=(
            "condition a selected covariance model on all real Munich measurements and "
            "write the analytical posterior on the active 100 m reference grid"
        ),
    )
    measurement_condition.add_argument("--measurements", required=True)
    measurement_condition.add_argument("--reference-grid", required=True)
    measurement_condition.add_argument("--calibration", required=True)
    measurement_condition.add_argument(
        "--model", choices=["matern32", "exponential", "radial_exponential"]
    )
    measurement_condition.add_argument("--sheet-name", default="kf_werte_180223")
    measurement_condition.add_argument("--stratigraphy", default="q")
    measurement_condition.add_argument("--groundwater-state", default="ungespannt")
    measurement_condition.add_argument("--expected-cell-size-m", type=float, default=100.0)
    measurement_condition.add_argument("--observation-std-log10-k", type=float, default=0.0)
    measurement_condition.add_argument("--kriging-chunk-size", type=int, default=100000)
    measurement_condition.add_argument("--dynamic-viscosity-pa-s", type=float, default=1.002e-3)
    measurement_condition.add_argument("--density-kg-m3", type=float, default=998.2)
    measurement_condition.add_argument("--gravity-m-s2", type=float, default=9.80665)
    measurement_condition.add_argument("--output", required=True)
    measurement_condition.set_defaults(func=_measurement_condition)

    new_domain = sub.add_parser(
        "new-domain-generate",
        help=(
            "define a new 12.8 km projected Munich domain, condition the calibrated "
            "structured KL field on the real measurements inside it, and generate "
            "5 m intrinsic-permeability realizations for LGCNN input"
        ),
    )
    new_domain.add_argument("--measurements", required=True)
    new_domain.add_argument("--reference-grid", required=True)
    new_domain.add_argument("--calibration", required=True)
    new_domain.add_argument("--model", choices=["matern32", "exponential"])
    new_domain.add_argument("--sheet-name", default="kf_werte_180223")
    new_domain.add_argument("--stratigraphy", default="q")
    new_domain.add_argument("--groundwater-state", default="ungespannt")
    new_domain.add_argument("--expected-reference-cell-size-m", type=float, default=100.0)
    new_domain.add_argument("--domain-size-m", type=float, default=12800.0)
    new_domain.add_argument("--cell-size-m", type=float, default=5.0)
    new_domain.add_argument("--domain-origin-x-m", type=float)
    new_domain.add_argument("--domain-origin-y-m", type=float)
    new_domain.add_argument("--min-conditioning-measurements", type=int, default=10)
    new_domain.add_argument("--n-modes", type=int)
    new_domain.add_argument("--energy-threshold", type=float, default=0.95)
    new_domain.add_argument("--n-samples", type=int, default=8)
    new_domain.add_argument("--batch-size", type=int, default=1)
    new_domain.add_argument("--seed", type=int, default=4901)
    new_domain.add_argument("--observation-std-log10-k", type=float, default=0.0)
    new_domain.add_argument("--diagnostic-grid-stride", type=int, default=128)
    new_domain.add_argument("--training-k-min-m2", type=float, default=1.02e-11)
    new_domain.add_argument("--training-k-max-m2", type=float, default=5.10e-9)
    new_domain.add_argument("--dynamic-viscosity-pa-s", type=float, default=1.002e-3)
    new_domain.add_argument("--density-kg-m3", type=float, default=998.2)
    new_domain.add_argument("--gravity-m-s2", type=float, default=9.80665)
    new_domain.add_argument(
        "--save-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write each generated 2560x2560 intrinsic-permeability field as .npy.",
    )
    new_domain.add_argument("--output-dir", required=True)
    new_domain.set_defaults(func=_new_domain_generate)

    georeference = sub.add_parser(
        "georeference-domain",
        help=(
            "infer the exact projected Munich origin and array orientation of one "
            "release25/DaRUS real-permeability LGCNN domain by matching its raw "
            "permeability fingerprint to the 100 m Munich reference model"
        ),
    )
    georeference.add_argument("--raw-dataset", required=True)
    georeference.add_argument("--run", default="RUN_1")
    georeference.add_argument("--reference-grid", required=True)
    georeference.add_argument(
        "--reference-column",
        default="K_P50",
        choices=["K_P10", "K_P50", "K_P90"],
    )
    georeference.add_argument(
        "--reference-z-mode",
        default="top",
        choices=["top", "bottom", "first", "log_geomean", "nearest"],
    )
    georeference.add_argument("--reference-z-m", type=float)
    georeference.add_argument("--raw-cell-size-m", type=float, default=5.0)
    georeference.add_argument("--coarse-anchor-stride", type=int, default=8)
    georeference.add_argument("--coarse-keep-per-transform", type=int, default=3)
    georeference.add_argument("--refine-radius-m", type=float, default=100.0)
    georeference.add_argument("--refine-step-m", type=float)
    georeference.add_argument("--refine-sample-stride-cells", type=int, default=128)
    georeference.add_argument("--min-reference-coverage", type=float, default=0.80)
    georeference.add_argument("--min-correlation", type=float, default=0.90)
    georeference.add_argument("--max-centered-rmse-log10", type=float, default=0.15)
    georeference.add_argument(
        "--require-validated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail instead of silently accepting a weak spatial match. Disable only "
            "for diagnosis of alternative reference surfaces/z modes."
        ),
    )
    georeference.add_argument("--measurements")
    georeference.add_argument("--sheet-name", default="kf_werte_180223")
    georeference.add_argument("--stratigraphy", default="q")
    georeference.add_argument("--groundwater-state", default="ungespannt")
    georeference.add_argument("--expected-reference-cell-size-m", type=float, default=100.0)
    georeference.add_argument("--dynamic-viscosity-pa-s", type=float, default=1.002e-3)
    georeference.add_argument("--density-kg-m3", type=float, default=998.2)
    georeference.add_argument("--gravity-m-s2", type=float, default=9.80665)
    georeference.add_argument("--plots-dir")
    georeference.add_argument("--output", required=True)
    georeference.set_defaults(func=_georeference_domain)

    sweep = sub.add_parser(
        "georeference-sweep",
        help=(
            "test all nine K_P10/K_P50/K_P90 x top/bottom/log_geomean Munich "
            "reference representations across multiple DaRUS RUN_n fields and "
            "decide whether one mapping is consistent and defensible"
        ),
    )
    sweep.add_argument("--raw-dataset", required=True)
    sweep.add_argument(
        "--runs",
        nargs="+",
        default=["RUN_1", "RUN_2", "RUN_3"],
        help="At least two RUN_n directories; defaults to RUN_1 RUN_2 RUN_3.",
    )
    sweep.add_argument("--reference-grid", required=True)
    sweep.add_argument("--raw-cell-size-m", type=float, default=5.0)
    sweep.add_argument("--coarse-anchor-stride", type=int, default=8)
    sweep.add_argument("--coarse-keep-per-transform", type=int, default=3)
    sweep.add_argument("--refine-radius-m", type=float, default=100.0)
    sweep.add_argument("--refine-step-m", type=float)
    sweep.add_argument("--refine-sample-stride-cells", type=int, default=128)
    sweep.add_argument("--min-reference-coverage", type=float, default=0.80)
    sweep.add_argument("--min-correlation", type=float, default=0.90)
    sweep.add_argument("--max-centered-rmse-log10", type=float, default=0.15)
    sweep.add_argument(
        "--max-unit-shift-spread-log10",
        type=float,
        default=0.15,
        help=(
            "Maximum allowed max-min fitted log10 unit-shift spread across runs "
            "for one representation to be called cross-run consistent."
        ),
    )
    sweep.add_argument(
        "--require-defensible",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Exit with an error after writing outputs when no representation satisfies "
            "all single-run and cross-run consistency criteria."
        ),
    )
    sweep.add_argument("--measurements")
    sweep.add_argument("--sheet-name", default="kf_werte_180223")
    sweep.add_argument("--stratigraphy", default="q")
    sweep.add_argument("--groundwater-state", default="ungespannt")
    sweep.add_argument("--expected-reference-cell-size-m", type=float, default=100.0)
    sweep.add_argument("--dynamic-viscosity-pa-s", type=float, default=1.002e-3)
    sweep.add_argument("--density-kg-m3", type=float, default=998.2)
    sweep.add_argument("--gravity-m-s2", type=float, default=9.80665)
    sweep.add_argument("--plots-dir")
    sweep.add_argument("--output-dir", required=True)
    sweep.set_defaults(func=_georeference_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
