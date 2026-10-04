from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..sampling.historical_realistic import (
    HISTORICAL_PARENT_CRS,
    load_parent_hydraulic_conductivity_tif,
    sample_parent_hydraulic_conductivity_at_projected_points,
)
from ..sampling.munich_measurements import (
    load_munich_hydraulic_conductivity_measurements,
)


def _metrics(
    measured_kh: np.ndarray,
    sampled_kh: np.ndarray,
    valid: np.ndarray,
) -> dict[str, object]:
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {
            "n": 0,
            "rmse_log10": None,
            "mae_log10": None,
            "bias_parent_minus_measurement_log10": None,
            "correlation_log10": None,
            "fraction_within_0_05_log10": None,
            "fraction_within_0_10_log10": None,
            "fraction_within_0_30_log10": None,
            "median_parent_to_measurement_ratio": None,
        }
    measured_log = np.log10(measured_kh[valid])
    sampled_log = np.log10(sampled_kh[valid])
    residual = sampled_log - measured_log
    correlation = None
    if count >= 2 and np.std(measured_log) > 0.0 and np.std(sampled_log) > 0.0:
        correlation = float(np.corrcoef(sampled_log, measured_log)[0, 1])
    return {
        "n": count,
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(np.abs(residual))),
        "bias_parent_minus_measurement_log10": float(np.mean(residual)),
        "correlation_log10": correlation,
        "fraction_within_0_05_log10": float(np.mean(np.abs(residual) <= 0.05)),
        "fraction_within_0_10_log10": float(np.mean(np.abs(residual) <= 0.10)),
        "fraction_within_0_30_log10": float(np.mean(np.abs(residual) <= 0.30)),
        "median_parent_to_measurement_ratio": float(
            np.median(sampled_kh[valid] / measured_kh[valid])
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    measured: np.ndarray,
    sampled: np.ndarray,
    valid: np.ndarray,
    output: Path,
) -> None:
    if not np.any(valid):
        return
    measured_log = np.log10(measured[valid])
    sampled_log = np.log10(sampled[valid])
    residual = sampled_log - measured_log

    lo = float(min(np.min(measured_log), np.min(sampled_log)))
    hi = float(max(np.max(measured_log), np.max(sampled_log)))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    axes[0].scatter(measured_log, sampled_log, s=16, alpha=0.7)
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("measured log10(Kh / m s^-1)")
    axes[0].set_ylabel("parent-raster log10(Kh / m s^-1)")
    axes[0].set_title("Parent map versus borehole measurements")

    limit = max(float(np.quantile(np.abs(residual), 0.95)), 1.0e-6)
    scatter = axes[1].scatter(
        x_m[valid],
        y_m[valid],
        c=residual,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        s=18,
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Easting [m]")
    axes[1].set_ylabel("Northing [m]")
    axes[1].set_title("Parent - measurement residual")
    figure.colorbar(scatter, ax=axes[1], label="log10(Kh) residual")
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    parent, parent_metadata = load_parent_hydraulic_conductivity_tif(
        args.parent_hydraulic_conductivity_tif
    )
    measurements, measurement_qc = load_munich_hydraulic_conductivity_measurements(
        args.measurements,
        sheet_name=args.sheet_name,
        stratigraphy=args.stratigraphy,
        groundwater_state=args.groundwater_state,
        reference_grid=None,
    )
    measured = np.asarray(
        measurements.hydraulic_conductivity_m_s, dtype=np.float64
    )

    sampled_linear, valid_linear = sample_parent_hydraulic_conductivity_at_projected_points(
        parent,
        parent_metadata,
        measurements.x_m,
        measurements.y_m,
        method="linear",
    )
    sampled_nearest, valid_nearest = sample_parent_hydraulic_conductivity_at_projected_points(
        parent,
        parent_metadata,
        measurements.x_m,
        measurements.y_m,
        method="nearest",
    )
    linear_metrics = _metrics(measured, sampled_linear, valid_linear)
    nearest_metrics = _metrics(measured, sampled_nearest, valid_nearest)

    rows: list[dict[str, object]] = []
    for index in range(len(measurements)):
        rows.append(
            {
                "measurement_id": str(measurements.measurement_id[index]),
                "excel_row": int(measurements.excel_row[index]),
                "x_m": float(measurements.x_m[index]),
                "y_m": float(measurements.y_m[index]),
                "measured_hydraulic_conductivity_m_s": float(measured[index]),
                "parent_linear_hydraulic_conductivity_m_s": (
                    None if not np.isfinite(sampled_linear[index])
                    else float(sampled_linear[index])
                ),
                "parent_nearest_hydraulic_conductivity_m_s": (
                    None if not np.isfinite(sampled_nearest[index])
                    else float(sampled_nearest[index])
                ),
                "linear_parent_minus_measurement_log10": (
                    None if not valid_linear[index]
                    else float(np.log10(sampled_linear[index]) - np.log10(measured[index]))
                ),
                "nearest_parent_minus_measurement_log10": (
                    None if not valid_nearest[index]
                    else float(np.log10(sampled_nearest[index]) - np.log10(measured[index]))
                ),
            }
        )
    _write_rows(output / "parent_measurement_pairs.csv", rows)
    _plot(
        x_m=measurements.x_m,
        y_m=measurements.y_m,
        measured=measured,
        sampled=sampled_linear,
        valid=valid_linear,
        output=output / "parent_measurement_comparison.png",
    )

    payload = {
        "schema_version": 1,
        "purpose": "historical_parent_map_vs_munich_measurement_consistency",
        "parent_raster": parent_metadata,
        "working_crs": HISTORICAL_PARENT_CRS,
        "measurement_source": str(Path(args.measurements).expanduser().resolve()),
        "measurement_qc": measurement_qc,
        "comparison": {
            "bilinear_parent_sampling": linear_metrics,
            "nearest_parent_sampling": nearest_metrics,
        },
        "interpretation": (
            "Agreement establishes spatial/value consistency between the supplied "
            "historical parent hydraulic-conductivity raster and the filtered workbook "
            "measurements. It does not prove that this exact workbook was the sole or "
            "original source used to construct the parent raster."
        ),
    }
    with (output / "parent_measurement_audit.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    with (output / "parent_measurement_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print("Historical parent-map / measurement audit")
    print(f"  filtered measurements: {len(measurements)}")
    print(
        "  bilinear: "
        f"n={linear_metrics['n']}, RMSE={linear_metrics['rmse_log10']}, "
        f"corr={linear_metrics['correlation_log10']}"
    )
    print(
        "  nearest: "
        f"n={nearest_metrics['n']}, RMSE={nearest_metrics['rmse_log10']}, "
        f"corr={nearest_metrics['correlation_log10']}"
    )
    print(f"Saved parent-map measurement audit to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the supplied historical 20 m Munich hydraulic-conductivity "
            "parent raster directly with the filtered borehole measurements."
        )
    )
    parser.add_argument("--parent-hydraulic-conductivity-tif", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--sheet-name", default="kf_werte_180223")
    parser.add_argument("--stratigraphy", default="q")
    parser.add_argument("--groundwater-state", default="ungespannt")
    parser.add_argument("--output-dir", required=True)
    parser.set_defaults(func=_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
