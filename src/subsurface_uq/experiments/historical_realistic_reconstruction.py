from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..sampling.darus_real import load_release25_raw_permeability_run
from ..sampling.geospatial import RAW_TO_GEO_TRANSFORMS, orient_raw_field
from ..sampling.historical_realistic import (
    HISTORICAL_GENERATOR_BRANCH,
    HISTORICAL_GENERATOR_COMMIT,
    HISTORICAL_GENERATOR_REPOSITORY,
    HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
    HISTORICAL_PARENT_CRS,
    HISTORICAL_PARENT_SHA256,
    HISTORICAL_PARENT_SHAPE_YX,
    HISTORICAL_SOURCE_RESOLUTION_M,
    load_parent_hydraulic_conductivity_tif,
    reconstruct_historical_training_permeability,
)
from ..sampling.training_provenance import (
    discover_realistic_runs,
    load_realistic_run_metadata,
)


def _compare_fields(
    raw_training: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, object]:
    reference = np.asarray(reconstructed, dtype=np.float64)
    reference_log = np.log10(reference)
    rows: list[dict[str, object]] = []
    for transform in RAW_TO_GEO_TRANSFORMS:
        candidate = np.asarray(orient_raw_field(raw_training, transform), dtype=np.float64)
        if candidate.shape != reference.shape:
            continue
        candidate_log = np.log10(candidate)
        residual = candidate_log - reference_log
        corr = None
        if np.std(candidate_log) > 0.0 and np.std(reference_log) > 0.0:
            corr = float(np.corrcoef(candidate_log.reshape(-1), reference_log.reshape(-1))[0, 1])
        rows.append(
            {
                "transform": transform,
                "rmse_log10": float(np.sqrt(np.mean(residual * residual))),
                "mae_log10": float(np.mean(np.abs(residual))),
                "bias_log10": float(np.mean(residual)),
                "max_abs_log10": float(np.max(np.abs(residual))),
                "correlation_log10": corr,
            }
        )
    if not rows:
        raise ValueError("no raw-array orientation has the reconstructed field shape")
    rows.sort(key=lambda row: float(row["rmse_log10"]))
    return {"selected": rows[0], "candidates": rows}


def _plot_comparison(
    *,
    raw_training: np.ndarray,
    reconstructed: np.ndarray,
    transform: str,
    run_name: str,
    destination: Path,
) -> None:
    training = np.asarray(orient_raw_field(raw_training, transform), dtype=np.float64)
    rebuilt = np.asarray(reconstructed, dtype=np.float64)
    training_log = np.log10(training)
    rebuilt_log = np.log10(rebuilt)
    pooled = np.concatenate((training_log.reshape(-1), rebuilt_log.reshape(-1)))
    lower, upper = (float(v) for v in np.quantile(pooled, [0.01, 0.99]))
    residual = training_log - rebuilt_log
    residual_limit = max(float(np.quantile(np.abs(residual), 0.99)), 1.0e-8)

    figure, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    image = axes[0].imshow(training_log, origin="lower", cmap="viridis", vmin=lower, vmax=upper)
    axes[0].set_title(f"{run_name}: stored training field")
    axes[1].imshow(rebuilt_log, origin="lower", cmap="viridis", vmin=lower, vmax=upper)
    axes[1].set_title("Reconstructed by historical pipeline")
    residual_image = axes[2].imshow(
        residual,
        origin="lower",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    axes[2].set_title("stored - reconstructed [log10(k)]")
    figure.colorbar(image, ax=axes[:2], label="log10(k / m²)", shrink=0.85)
    figure.colorbar(residual_image, ax=axes[2], label="difference [log10(k)]")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def _run(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    discovered = discover_realistic_runs(
        dataset_root,
        metadata_filename=args.metadata_filename,
        require_h5=True,
    )
    run_names = tuple(args.runs) if args.runs else discovered
    missing = [run for run in run_names if run not in discovered]
    if missing:
        raise ValueError(f"requested runs do not have HDF5 + metadata: {missing}")

    parent_kh, parent_metadata = load_parent_hydraulic_conductivity_tif(
        args.parent_hydraulic_conductivity_tif
    )
    parent_crs = str(parent_metadata.get("crs"))
    parent_resolution = tuple(float(v) for v in parent_metadata["resolution_m"])
    parent_shape = tuple(int(v) for v in parent_metadata["shape_yx"])
    parent_sha256 = str(parent_metadata["sha256"])
    crs_matches = parent_crs.upper() == HISTORICAL_PARENT_CRS.upper()
    resolution_matches = all(
        np.isclose(value, HISTORICAL_SOURCE_RESOLUTION_M, atol=1.0e-6)
        for value in parent_resolution
    )
    shape_matches = parent_shape == HISTORICAL_PARENT_SHAPE_YX
    checksum_matches = parent_sha256.lower() == HISTORICAL_PARENT_SHA256.lower()
    if args.require_historical_parent_metadata and not crs_matches:
        raise ValueError(
            f"parent GeoTIFF CRS is {parent_crs!r}; the supplied historical parent "
            f"raster is {HISTORICAL_PARENT_CRS}"
        )
    if args.require_historical_parent_metadata and not resolution_matches:
        raise ValueError(
            f"parent GeoTIFF resolution is {parent_resolution}; historical source "
            f"resolution was {HISTORICAL_SOURCE_RESOLUTION_M:g} m"
        )
    if args.require_historical_parent_metadata and not shape_matches:
        raise ValueError(
            f"parent GeoTIFF shape is {parent_shape}; exact historical source shape "
            f"is {HISTORICAL_PARENT_SHAPE_YX}"
        )
    if args.require_exact_parent_checksum and not checksum_matches:
        raise ValueError(
            "parent GeoTIFF SHA-256 does not match the supplied historical "
            "Hydraulic_conductivity_20m_resolution.tif"
        )

    summary_rows: list[dict[str, object]] = []
    run_payloads: list[dict[str, object]] = []
    for run_name in run_names:
        metadata = load_realistic_run_metadata(
            dataset_root,
            run_name,
            filename=args.metadata_filename,
        )
        if not np.isclose(
            metadata.original_resolution_m,
            HISTORICAL_SOURCE_RESOLUTION_M,
            atol=1.0e-6,
        ):
            raise ValueError(
                f"{run_name} metadata resolution {metadata.original_resolution_m:g} m "
                f"does not match recovered historical source resolution"
            )
        stored = load_release25_raw_permeability_run(
            dataset_root,
            run_name,
            cell_size_m=args.cell_size_m,
        )
        reconstruction = reconstruct_historical_training_permeability(
            parent_kh,
            metadata,
            destination_shape_yx=stored.shape,
            destination_resolution_m=args.cell_size_m,
        )
        comparison = _compare_fields(
            stored,
            reconstruction.reconstructed_permeability_m2,
        )
        selected = comparison["selected"]
        validated = bool(
            float(selected["rmse_log10"]) <= args.max_rmse_log10
            and (
                selected["correlation_log10"] is None
                or float(selected["correlation_log10"]) >= args.min_correlation
            )
        )
        run_row = {
            "run_name": run_name,
            "rotation_angle_deg": metadata.rotation_angle_deg,
            "start_x_m_in_parent_array": metadata.start_position_m[0],
            "start_y_m_in_parent_array": metadata.start_position_m[1],
            "selected_raw_transform": selected["transform"],
            "rmse_log10": selected["rmse_log10"],
            "mae_log10": selected["mae_log10"],
            "bias_log10": selected["bias_log10"],
            "max_abs_log10": selected["max_abs_log10"],
            "correlation_log10": selected["correlation_log10"],
            "validated": validated,
        }
        summary_rows.append(run_row)
        run_payloads.append(
            {
                "metadata": metadata.to_dict(),
                "comparison": comparison,
                "validated": validated,
                "reconstructed_npy": str(output / "reconstructed" / f"{run_name}_permeability_m2.npy"),
                "comparison_png": str(output / "comparisons" / f"{run_name}_historical_reconstruction.png"),
            }
        )
        reconstructed_path = output / "reconstructed" / f"{run_name}_permeability_m2.npy"
        reconstructed_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(
            reconstructed_path,
            reconstruction.reconstructed_permeability_m2.astype(np.float32),
        )
        _plot_comparison(
            raw_training=stored,
            reconstructed=reconstruction.reconstructed_permeability_m2,
            transform=str(selected["transform"]),
            run_name=run_name,
            destination=output / "comparisons" / f"{run_name}_historical_reconstruction.png",
        )

    all_validated = bool(summary_rows) and all(bool(row["validated"]) for row in summary_rows)
    payload = {
        "schema_version": 2,
        "historical_generator": {
            "repository": HISTORICAL_GENERATOR_REPOSITORY,
            "branch": HISTORICAL_GENERATOR_BRANCH,
            "commit": HISTORICAL_GENERATOR_COMMIT,
            "parent_crs": HISTORICAL_PARENT_CRS,
            "source_resolution_m": HISTORICAL_SOURCE_RESOLUTION_M,
            "source_shape_yx": list(HISTORICAL_PARENT_SHAPE_YX),
            "known_parent_sha256": HISTORICAL_PARENT_SHA256,
            "hydraulic_conductivity_to_permeability": (
                f"k = K_h / {HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR:g}"
            ),
            "rotation_rule": "rotation_rad = deg2rad(realistic_params_angle_deg + 90)",
            "source_index_rule": "rotated floating source indices are truncated with astype(int)",
            "resampling_rule": (
                "scipy RegularGridInterpolator on the extracted 20 m window, queried "
                "at PFLOTRAN 5 m cell centres; historical TODO confirms no +0.5 source-pixel correction"
            ),
        },
        "parent_tif": parent_metadata,
        "parent_metadata_checks": {
            "crs_matches_supplied_historical_parent": crs_matches,
            "resolution_matches_historical_generator": resolution_matches,
            "shape_matches_supplied_historical_parent": shape_matches,
            "sha256_matches_supplied_historical_parent": checksum_matches,
        },
        "runs": run_payloads,
        "validation_thresholds": {
            "max_rmse_log10": float(args.max_rmse_log10),
            "min_correlation": float(args.min_correlation),
        },
        "historical_training_pipeline_reconstruction_validated": all_validated,
        "interpretation": (
            "If all requested runs validate, the stored DaRUS permeability fields are "
            "numerically reproducible from the supplied 20 m parent hydraulic-conductivity "
            "GeoTIFF using the recovered historical extraction/rotation/interpolation code. "
            "This establishes field provenance independently of any stochastic model."
        ),
    }
    _write_csv(output / "historical_reconstruction_summary.csv", summary_rows)
    with (output / "historical_reconstruction.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    with (output / "historical_reconstruction.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("Historical realistic-permeability reconstruction")
    print(f"  recovered generator commit: {HISTORICAL_GENERATOR_COMMIT}")
    print(
        "  parent CRS/resolution/shape/checksum match: "
        f"{crs_matches} / {resolution_matches} / {shape_matches} / {checksum_matches}"
    )
    for row in summary_rows:
        print(
            f"  {row['run_name']}: RMSE={float(row['rmse_log10']):.6g}, "
            f"corr={row['correlation_log10']}, validated={row['validated']}"
        )
    print(f"  all requested runs validated: {all_validated}")
    print(f"Saved historical reconstruction artifacts to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce DaRUS-5065 realistic permeability fields with the exact "
            "historical extraction/rotation/interpolation algorithm recovered from "
            "JuliaPelzer/Dataset-generation-with-Pflotran."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--parent-hydraulic-conductivity-tif", required=True)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--metadata-filename", default="realistic_params.yaml")
    parser.add_argument("--cell-size-m", type=float, default=5.0)
    parser.add_argument("--max-rmse-log10", type=float, default=1.0e-3)
    parser.add_argument("--min-correlation", type=float, default=0.999)
    parser.add_argument(
        "--require-historical-parent-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the supplied parent raster to match the observed historical "
            "20 m EPSG:5678 raster shape/CRS metadata."
        ),
    )
    parser.add_argument(
        "--require-exact-parent-checksum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Additionally require the byte-for-byte SHA-256 of the supplied "
            "Hydraulic_conductivity_20m_resolution.tif. Metadata matching remains "
            "the default so lossless GeoTIFF rewrites do not block reconstruction."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(func=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
