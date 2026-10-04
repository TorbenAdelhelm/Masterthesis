from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..sampling.darus_real import load_release25_raw_permeability_run
from ..sampling.geospatial import (
    LGCNNDomainGeoreference,
    infer_lgcnn_domain_georeference,
    load_reference_permeability_surface,
    load_reference_permeability_sweep_surfaces,
    orient_raw_field,
    summarize_georeference_sweep,
)
from ..sampling.munich_measurements import (
    hydraulic_conductivity_to_intrinsic_permeability,
    load_munich_hydraulic_conductivity_measurements,
    load_reference_horizontal_grid,
)
from ..sampling.training_provenance import (
    RealisticRunMetadata,
    discover_realistic_runs,
    load_realistic_run_metadata,
    measurement_training_consistency,
    summarize_metadata_georeference_relation,
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_run_measurement_overlay(
    *,
    field: np.ndarray,
    georef: LGCNNDomainGeoreference,
    measurement_x_m: np.ndarray,
    measurement_y_m: np.ndarray,
    measured_k_m2: np.ndarray,
    sampled_k_m2: np.ndarray,
    valid: np.ndarray,
    destination: Path,
) -> None:
    geo = np.asarray(orient_raw_field(field, georef.transform), dtype=np.float64)
    field_log = np.log10(geo)
    measured_log = np.log10(np.asarray(measured_k_m2, dtype=np.float64))
    values_for_scale = [field_log.reshape(-1)]
    if np.any(valid):
        values_for_scale.append(measured_log[valid])
    pooled = np.concatenate(values_for_scale)
    lower, upper = (float(v) for v in np.quantile(pooled, [0.01, 0.99]))
    extent = (georef.west_edge_m, georef.east_edge_m, georef.south_edge_m, georef.north_edge_m)

    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    image = axes[0].imshow(
        field_log,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )
    if np.any(valid):
        axes[0].scatter(
            measurement_x_m[valid],
            measurement_y_m[valid],
            c=measured_log[valid],
            cmap="viridis",
            vmin=lower,
            vmax=upper,
            s=44,
            edgecolors="white",
            linewidths=0.8,
            label="measured intrinsic k",
        )
        axes[0].legend(loc="best")
    axes[0].set_title(f"{georef.run_name}: training field + measurements")
    axes[0].set_xlabel("projected x [m]")
    axes[0].set_ylabel("projected y [m]")
    figure.colorbar(image, ax=axes[0], label="log10(k / m²)")

    axes[1].imshow(
        field_log,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="Greys",
        alpha=0.35,
    )
    if np.any(valid):
        residual = np.log10(sampled_k_m2[valid]) - measured_log[valid]
        limit = max(float(np.quantile(np.abs(residual), 0.95)), 1.0e-6)
        residual_scatter = axes[1].scatter(
            measurement_x_m[valid],
            measurement_y_m[valid],
            c=residual,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            s=52,
            edgecolors="black",
            linewidths=0.5,
        )
        figure.colorbar(
            residual_scatter,
            ax=axes[1],
            label="training - measurement [log10(k)]",
        )
    axes[1].set_title("Residuals at measurement locations")
    axes[1].set_xlabel("projected x [m]")
    axes[1].set_ylabel("projected y [m]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _infer_georeferences(
    *,
    fields: dict[str, np.ndarray],
    run_names: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[dict[str, LGCNNDomainGeoreference], dict[str, object], object | None]:
    rows: list[dict[str, object]] = []
    resolved: dict[tuple[str, str, str], LGCNNDomainGeoreference] = {}

    if args.reference_mode == "sweep":
        surfaces = load_reference_permeability_sweep_surfaces(args.reference_grid)
        for (column, z_mode), surface in surfaces.items():
            for run_name in run_names:
                try:
                    georef = infer_lgcnn_domain_georeference(
                        fields[run_name],
                        surface,
                        run_name=run_name,
                        raw_cell_size_m=args.cell_size_m,
                        coarse_anchor_stride=args.coarse_anchor_stride,
                        coarse_keep_per_transform=args.coarse_keep_per_transform,
                        refine_radius_m=args.refine_radius_m,
                        refine_step_m=args.refine_step_m,
                        refine_sample_stride_cells=args.refine_sample_stride_cells,
                        min_reference_coverage=args.min_reference_coverage,
                        min_correlation=args.min_correlation,
                        max_centered_rmse_log10=args.max_centered_rmse_log10,
                    )
                    row = georef.to_dict()
                    resolved[(column, z_mode, run_name)] = georef
                except Exception as exc:  # diagnostic sweep must retain failed candidates
                    row = {
                        "run_name": run_name,
                        "reference_column": column,
                        "reference_z_mode": z_mode,
                        "validated": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(row)
        summary = summarize_georeference_sweep(
            rows,
            run_names=run_names,
            max_unit_shift_spread_log10=args.max_unit_shift_spread_log10,
        )
        selected = summary.get("selected_representation")
        if selected is None:
            return {}, summary, None
        column = str(selected["reference_column"])
        z_mode = str(selected["reference_z_mode"])
        georefs = {
            run: resolved[(column, z_mode, run)]
            for run in run_names
            if (column, z_mode, run) in resolved
        }
        return georefs, summary, surfaces[(column, z_mode)]

    surface = load_reference_permeability_surface(
        args.reference_grid,
        column=args.reference_column,
        z_mode=args.reference_z_mode,
        z_value_m=args.reference_z_m,
    )
    georefs: dict[str, LGCNNDomainGeoreference] = {}
    for run_name in run_names:
        try:
            georef = infer_lgcnn_domain_georeference(
                fields[run_name],
                surface,
                run_name=run_name,
                raw_cell_size_m=args.cell_size_m,
                coarse_anchor_stride=args.coarse_anchor_stride,
                coarse_keep_per_transform=args.coarse_keep_per_transform,
                refine_radius_m=args.refine_radius_m,
                refine_step_m=args.refine_step_m,
                refine_sample_stride_cells=args.refine_sample_stride_cells,
                min_reference_coverage=args.min_reference_coverage,
                min_correlation=args.min_correlation,
                max_centered_rmse_log10=args.max_centered_rmse_log10,
            )
            georefs[run_name] = georef
            rows.append(georef.to_dict())
        except Exception as exc:
            rows.append(
                {
                    "run_name": run_name,
                    "reference_column": args.reference_column,
                    "reference_z_mode": args.reference_z_mode,
                    "validated": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    all_valid = len(georefs) == len(run_names) and all(g.validated for g in georefs.values())
    summary = {
        "run_names": list(run_names),
        "reference_mode": "single",
        "consistent_defensible_mapping_exists": bool(all_valid),
        "selected_representation": (
            {
                "reference_column": args.reference_column,
                "reference_z_mode": args.reference_z_mode,
            }
            if all_valid
            else None
        ),
        "rows": rows,
    }
    return (georefs if all_valid else {}), summary, surface


def _audit(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    discovered = discover_realistic_runs(
        root,
        metadata_filename=args.metadata_filename,
        require_h5=True,
    )
    run_names = tuple(args.runs) if args.runs else discovered
    missing = [run for run in run_names if run not in discovered]
    if missing:
        raise ValueError(f"requested runs lack HDF5/metadata below {root}: {missing}")

    metadata: list[RealisticRunMetadata] = []
    fields: dict[str, np.ndarray] = {}
    for run_name in run_names:
        metadata.append(
            load_realistic_run_metadata(
                root,
                run_name,
                filename=args.metadata_filename,
            )
        )
        fields[run_name] = load_release25_raw_permeability_run(
            root,
            run_name,
            cell_size_m=args.cell_size_m,
        )

    georefs, georef_summary, selected_surface = _infer_georeferences(
        fields=fields,
        run_names=run_names,
        args=args,
    )
    absolute_mapping_validated = bool(
        georef_summary.get("consistent_defensible_mapping_exists", False)
        and len(georefs) == len(run_names)
    )

    metadata_relation = (
        summarize_metadata_georeference_relation(
            metadata,
            georefs,
            tolerance_m=args.metadata_translation_tolerance_m,
        )
        if absolute_mapping_validated
        else {
            "rotation_convention_resolved": False,
            "interpretation": (
                "No consistent validated absolute georeference was recovered, so the "
                "historical start-position relation cannot yet be tested."
            ),
        }
    )

    reference_spacing = None
    if selected_surface is not None:
        reference_spacing = {
            "x_m": float(selected_surface.spacing_x_m),
            "y_m": float(selected_surface.spacing_y_m),
        }
    original_resolutions = sorted({float(item.original_resolution_m) for item in metadata})
    historical_parent_resolution_matches_reference = bool(
        selected_surface is not None
        and len(original_resolutions) == 1
        and np.isclose(selected_surface.spacing_x_m, original_resolutions[0])
        and np.isclose(selected_surface.spacing_y_m, original_resolutions[0])
    )

    measurement_rows: list[dict[str, object]] = []
    per_run_measurement: list[dict[str, object]] = []
    measurement_qc: dict[str, object] | None = None
    if absolute_mapping_validated:
        reference_horizontal = load_reference_horizontal_grid(
            args.reference_grid,
            expected_cell_size_m=args.expected_reference_cell_size_m,
        )
        measurements, measurement_qc = load_munich_hydraulic_conductivity_measurements(
            args.measurements,
            sheet_name=args.sheet_name,
            stratigraphy=args.stratigraphy,
            groundwater_state=args.groundwater_state,
            reference_grid=reference_horizontal,
        )
        measured_k = hydraulic_conductivity_to_intrinsic_permeability(
            measurements.hydraulic_conductivity_m_s,
            dynamic_viscosity_pa_s=args.dynamic_viscosity_pa_s,
            density_kg_m3=args.density_kg_m3,
            gravity_m_s2=args.gravity_m_s2,
        )
        for run_name in run_names:
            metrics, sampled_k, valid = measurement_training_consistency(
                fields[run_name],
                georefs[run_name],
                measurements.x_m,
                measurements.y_m,
                measured_k,
            )
            per_run_measurement.append({"run_name": run_name, **metrics})
            for index in np.flatnonzero(valid):
                measured_log = float(np.log10(measured_k[index]))
                training_log = float(np.log10(sampled_k[index]))
                measurement_rows.append(
                    {
                        "run_name": run_name,
                        "measurement_id": str(measurements.measurement_id[index]),
                        "excel_row": int(measurements.excel_row[index]),
                        "x_m": float(measurements.x_m[index]),
                        "y_m": float(measurements.y_m[index]),
                        "hydraulic_conductivity_m_s": float(
                            measurements.hydraulic_conductivity_m_s[index]
                        ),
                        "measured_intrinsic_permeability_m2": float(measured_k[index]),
                        "training_intrinsic_permeability_m2": float(sampled_k[index]),
                        "measured_log10_k_m2": measured_log,
                        "training_log10_k_m2": training_log,
                        "training_minus_measurement_log10": training_log - measured_log,
                    }
                )
            _plot_run_measurement_overlay(
                field=fields[run_name],
                georef=georefs[run_name],
                measurement_x_m=measurements.x_m,
                measurement_y_m=measurements.y_m,
                measured_k_m2=measured_k,
                sampled_k_m2=sampled_k,
                valid=valid,
                destination=output / "overlays" / f"{run_name}_measurement_overlay.png",
            )

    total_overlap = int(
        sum(int(row.get("measurement_count_in_run", 0)) for row in per_run_measurement)
    )
    provenance_evidence_available = bool(absolute_mapping_validated and total_overlap > 0)
    payload = {
        "schema_version": 1,
        "purpose": "darus_5065_training_measurement_provenance_audit",
        "source": {
            "dataset_root": str(root),
            "measurements": str(Path(args.measurements).expanduser().resolve()),
            "reference_grid": str(Path(args.reference_grid).expanduser().resolve()),
            "runs": list(run_names),
            "metadata_filename": args.metadata_filename,
        },
        "run_metadata": [item.to_dict() for item in metadata],
        "metadata_summary": {
            "original_resolutions_m": original_resolutions,
            "rotation_angles_deg": [float(item.rotation_angle_deg) for item in metadata],
            "historical_parent_resolution_matches_reference_surface": (
                historical_parent_resolution_matches_reference
            ),
            "selected_reference_spacing_m": reference_spacing,
        },
        "georeference": georef_summary,
        "metadata_georeference_relation": metadata_relation,
        "measurement_qc": measurement_qc,
        "measurement_vs_training": {
            "total_run_measurement_pairs": total_overlap,
            "per_run": per_run_measurement,
            "detail_csv": str(output / "measurement_training_pairs.csv"),
        },
        "provenance_conclusions": {
            "absolute_training_cutout_mapping_validated": absolute_mapping_validated,
            "spatial_measurement_training_comparison_available": provenance_evidence_available,
            "exact_historical_rotation_crop_resampling_reconstructed": False,
            "exact_measurement_workbook_used_historically_verified": False,
            "safe_for_residual_generator_calibration": False,
        },
        "decision_note": (
            "A validated template match plus agreement with measurements can support "
            "the claim that the DaRUS fields are spatially consistent with the Munich "
            "measurement-derived map. It does not prove that this exact workbook was "
            "the historical input. Residual-model generation must remain disabled until "
            "the rotation/crop/resampling convention or an equivalent exact parent-map "
            "reconstruction is independently validated."
        ),
    }

    _write_rows(output / "measurement_training_pairs.csv", measurement_rows)
    _write_rows(output / "measurement_training_summary.csv", per_run_measurement)
    _write_rows(output / "georeference_rows.csv", list(georef_summary.get("rows", [])))
    with (output / "training_provenance.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    with (output / "training_provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("DaRUS-5065 training/measurement provenance audit")
    print(f"  runs: {', '.join(run_names)}")
    print(f"  absolute mapping validated: {absolute_mapping_validated}")
    print(f"  measurement/run comparison pairs: {total_overlap}")
    print(
        "  exact historical rotation/crop/resampling reconstructed: False "
        "(not guessed from metadata)"
    )
    print(f"Saved provenance artifacts to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether DaRUS-5065 realistic permeability RUNs can be placed back "
            "onto the Munich parent map and compared to the borehole measurements."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--reference-grid", required=True)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--metadata-filename", default="realistic_params.yaml")
    parser.add_argument("--cell-size-m", type=float, default=5.0)
    parser.add_argument(
        "--reference-mode",
        choices=["sweep", "single"],
        default="sweep",
        help=(
            "sweep tests K_P10/P50/P90 x top/bottom/log-geomean and requires one "
            "representation to map consistently across all requested runs."
        ),
    )
    parser.add_argument("--reference-column", default="K_P50")
    parser.add_argument("--reference-z-mode", default="top")
    parser.add_argument("--reference-z-m", type=float)
    parser.add_argument("--sheet-name", default="kf_werte_180223")
    parser.add_argument("--stratigraphy", default="q")
    parser.add_argument("--groundwater-state", default="ungespannt")
    parser.add_argument("--expected-reference-cell-size-m", type=float, default=100.0)
    parser.add_argument("--coarse-anchor-stride", type=int, default=8)
    parser.add_argument("--coarse-keep-per-transform", type=int, default=3)
    parser.add_argument("--refine-radius-m", type=float, default=100.0)
    parser.add_argument("--refine-step-m", type=float, default=5.0)
    parser.add_argument("--refine-sample-stride-cells", type=int, default=128)
    parser.add_argument("--min-reference-coverage", type=float, default=0.80)
    parser.add_argument("--min-correlation", type=float, default=0.90)
    parser.add_argument("--max-centered-rmse-log10", type=float, default=0.15)
    parser.add_argument("--max-unit-shift-spread-log10", type=float, default=0.15)
    parser.add_argument("--metadata-translation-tolerance-m", type=float, default=100.0)
    parser.add_argument("--dynamic-viscosity-pa-s", type=float, default=1.002e-3)
    parser.add_argument("--density-kg-m3", type=float, default=998.2)
    parser.add_argument("--gravity-m-s2", type=float, default=9.80665)
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(func=_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
