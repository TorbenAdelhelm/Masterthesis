"""Reproducible matched-support RUN/parent comparison and uncertainty scenarios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .reference_field_permeability import coarsen_reference, main as generate
from ..sampling.darus_real import load_release25_raw_permeability_run
from ..sampling.calibration import calibrate_covariance_candidates
from ..sampling.historical_realistic import (
    load_parent_hydraulic_conductivity_tif, reconstruct_historical_training_permeability,
    sample_parent_hydraulic_conductivity_at_projected_points, HISTORICAL_PARENT_SHA256,
)
from ..sampling.training_provenance import load_realistic_run_metadata
from ..sampling.munich_measurements import load_munich_hydraulic_conductivity_measurements
from ..sampling.geospatial import RAW_TO_GEO_TRANSFORMS, orient_raw_field


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--reference-runs", nargs="+", required=True,
                        help="Explicit actual training/support RUN IDs; no automatic split inference")
    parser.add_argument("--parent-tif")
    parser.add_argument("--measurements")
    parser.add_argument("--roi", type=int, nargs=4, default=[640, 1920, 640, 1920])
    parser.add_argument("--cell-size-m", type=float, default=5.)
    parser.add_argument("--coarsen-factor", type=int, default=8)
    parser.add_argument("--residual-stds", type=float, nargs="+", default=[.05, .1])
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4901)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset_root).resolve()
    r0, r1, c0, c1 = args.roi
    reference_fields = []
    report = {"reference_runs": args.reference_runs, "roi": args.roi,
              "cell_size_m": args.cell_size_m * args.coarsen_factor,
              "coarsening": "geometric_block_mean; different support from CNN stride decimation",
              "residual_amplitudes_are_sensitivity_assumptions": True,
              "parent_reconstruction": {}, "scenarios": []}
    parent, parent_metadata = (None, None)
    if args.parent_tif:
        parent, parent_metadata = load_parent_hydraulic_conductivity_tif(args.parent_tif)
        if parent_metadata["sha256"] != HISTORICAL_PARENT_SHA256:
            raise ValueError("validation needs the exact recovered historical parent raster")
        report["parent_source"] = parent_metadata
    for run in args.reference_runs:
        field = load_release25_raw_permeability_run(dataset, run, cell_size_m=args.cell_size_m)
        if not (0 <= r0 < r1 <= field.shape[0] and 0 <= c0 < c1 <= field.shape[1]):
            raise ValueError("ROI must lie inside every reference field")
        reference = coarsen_reference(field[r0:r1, c0:c1], args.coarsen_factor)
        reference_fields.append(reference)
        np.save(root / f"{run}.reference.npy", reference)
        if parent is not None:
            metadata = load_realistic_run_metadata(dataset, run_name=run)
            try:
                reconstructed = reconstruct_historical_training_permeability(
                    parent, metadata, destination_shape_yx=field.shape, destination_resolution_m=args.cell_size_m,
                ).reconstructed_permeability_m2
            except ValueError as error:
                report["parent_reconstruction"][run] = {"status": "unverified", "reason": str(error)}
                print(f"{run}: parent reconstruction unverified: {error}")
            else:
                orientation_rows = []
                for transform in RAW_TO_GEO_TRANSFORMS:
                    oriented = orient_raw_field(reconstructed, transform)
                    parent_reference = coarsen_reference(oriented[r0:r1, c0:c1], args.coarsen_factor)
                    difference = np.log10(reference) - np.log10(parent_reference)
                    orientation_rows.append({
                        "transform": transform, "rmse_log10": float(np.sqrt(np.mean(difference**2))),
                        "max_abs_log10": float(np.max(np.abs(difference))),
                        "correlation_log10": float(np.corrcoef(np.log10(reference).ravel(), np.log10(parent_reference).ravel())[0, 1]),
                    })
                orientation_rows.sort(key=lambda row: row["rmse_log10"])
                report["parent_reconstruction"][run] = {
                    "status": "computed", "selected": orientation_rows[0],
                    "orientation_candidates": orientation_rows,
                    "interpretation": "orientation search on matched ROI support; not proof of exact source lineage",
                }
        del field
    calibrations = calibrate_covariance_candidates(
        np.asarray(reference_fields), cell_size_m=report["cell_size_m"], max_lag_cells=48,
        models=("matern32", "exponential", "radial_exponential"),
    )
    report["reference_texture_covariance_fits"] = [item.to_dict() for item in calibrations]
    chosen = next(item for item in calibrations if item.covariance_model in {"matern32", "exponential"})
    report["covariance_shape_interpretation"] = (
        "Directional variogram fit to reference texture is a covariance-shape proxy only. "
        "It does not identify the covariance or amplitude of interpolation error. No spatial trend is removed."
    )
    for run in args.reference_runs:
        for amplitude in args.residual_stds:
            scenario_dir = root / f"{run}_std_{amplitude:g}"
            generate(["--reference-npy", str(root / f"{run}.reference.npy"),
                      "--permeability-convention", "historical-training",
                      "--cell-size-m", str(report["cell_size_m"]),
                      "--residual-std-log10-k", str(amplitude),
                      "--length-scale-y-m", str(chosen.length_scale_y_m),
                      "--length-scale-x-m", str(chosen.length_scale_x_m),
                      "--covariance-model", chosen.covariance_model.removeprefix("separable_"),
                      "--n-samples", str(args.n_samples), "--seed", str(args.seed),
                      "--output-dir", str(scenario_dir)])
            summary = json.loads((scenario_dir / "summary.json").read_text())
            report["scenarios"].append({"reference_run": run, "residual_std_log10_k": amplitude,
                                        **summary})
    if args.measurements:
        if parent is None:
            raise ValueError("measurement comparison requires --parent-tif")
        measurements, qc = load_munich_hydraulic_conductivity_measurements(args.measurements)
        parent_values, inside = sample_parent_hydraulic_conductivity_at_projected_points(
            parent, parent_metadata, x_m=measurements.x_m, y_m=measurements.y_m,
        )
        valid = inside & np.isfinite(parent_values) & (parent_values > 0)
        residual = measurements.log10_hydraulic_conductivity[valid] - np.log10(parent_values[valid])
        if residual.size < 2:
            raise ValueError("at least two valid parent/measurement comparisons required")
        report["parent_measurement_diagnostic"] = {
            "qc": qc, "valid_pair_count": int(valid.sum()),
            "rmse_log10": float(np.sqrt(np.mean(residual**2))), "mean_log10_residual": float(residual.mean()),
            "std_log10_residual": float(residual.std(ddof=1)),
            "interpretation": "descriptive same-support audit, not independent holdout evidence or calibrated uncertainty amplitude",
        }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved RUN, parent and sensitivity report to {root / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
