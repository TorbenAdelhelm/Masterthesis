from __future__ import annotations

import argparse
from pathlib import Path

from .diagnostics import save_temperature_diagnostic_plots, summarize_temperature_errors
from .overlays import (
    build_release25_validation_overlay_context,
    save_validation_overlay_plots,
)
from .temperature import (
    compare_temperature_fields,
    load_prediction_field,
    load_prepared_temperature_label,
    save_temperature_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one physical release25 temperature prediction against a "
            "prepared DaRUS temperature label for the same run id."
        )
    )
    parser.add_argument(
        "--prediction",
        required=True,
        help="Prediction file (.npz from release25 runner or a 2-D .npy field).",
    )
    parser.add_argument(
        "--prediction-key",
        default="mean",
        help="Key to read from --prediction when it is an .npz archive (default: mean).",
    )
    parser.add_argument(
        "--reference-dir",
        required=True,
        help="Prepared temperature dataset containing info.yaml and Labels/.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--alignment",
        choices=("center-crop-reference", "strict"),
        default="center-crop-reference",
        help=(
            "Spatial alignment policy. The default center-crops the reference "
            "to the smaller UNetNoPad2 prediction field."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="run_output/release25_temperature_validation.npz",
    )
    parser.add_argument(
        "--plots-dir",
        help=(
            "Optional directory for prediction, reference, signed-error and "
            "absolute-error PNG maps."
        ),
    )
    parser.add_argument(
        "--overlay-plots-dir",
        help=(
            "Optional directory for prediction/reference/absolute-error maps "
            "overlaid with regenerated release25 streamlines and heat-pump locations."
        ),
    )
    parser.add_argument(
        "--release25-repo",
        help="External release25 checkout; required together with --overlay-plots-dir.",
    )
    parser.add_argument(
        "--cnn1-dir",
        help="Pretrained release25 Step-1 model directory for overlay regeneration.",
    )
    parser.add_argument(
        "--cnn2-dir",
        help="Pretrained release25 Step-3 model directory for overlay regeneration.",
    )
    parser.add_argument(
        "--prepared-pki-dir",
        help="Prepared p,k,i dataset used by the deterministic release25 run.",
    )
    parser.add_argument(
        "--random-k",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--streamline-method",
        choices=("RK45", "RK23", "Radau"),
        default="RK45",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=5.0,
        help="Grid-cell size in metres for validation overlay axes (default: 5).",
    )
    parser.add_argument(
        "--streamline-plot-threshold",
        type=float,
        default=0.05,
        help="Threshold for the central streamline contour overlay (default: 0.05).",
    )
    return parser


def _require_overlay_arguments(args: argparse.Namespace) -> None:
    required = {
        "--release25-repo": args.release25_repo,
        "--cnn1-dir": args.cnn1_dir,
        "--cnn2-dir": args.cnn2_dir,
        "--prepared-pki-dir": args.prepared_pki_dir,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "--overlay-plots-dir requires " + ", ".join(missing)
        )


def main() -> None:
    args = build_parser().parse_args()
    prediction = load_prediction_field(args.prediction, key=args.prediction_key)
    reference = load_prepared_temperature_label(
        args.reference_dir,
        args.run_id,
        device=args.device,
    )
    original_reference_shape = reference.shape
    comparison = compare_temperature_fields(
        prediction,
        reference,
        alignment=args.alignment,
    )
    diagnostics = summarize_temperature_errors(comparison)
    destination = save_temperature_comparison(
        comparison,
        args.output,
        run_id=args.run_id,
    )

    print(f"Run id: {args.run_id}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Reference shape: {original_reference_shape}")
    if original_reference_shape != comparison.reference.shape:
        print(f"Aligned reference shape: {comparison.reference.shape}")
    print(f"Prediction min/max: {comparison.prediction.min():.6f} / {comparison.prediction.max():.6f} degC")
    print(f"Reference min/max: {comparison.reference.min():.6f} / {comparison.reference.max():.6f} degC")
    print(f"MAE: {comparison.mae:.6f} degC")
    print(f"MSE: {comparison.mse:.6f} degC^2")
    print(f"RMSE: {comparison.rmse:.6f} degC")
    print(f"Max absolute error: {comparison.max_absolute_error:.6f} degC")
    print(f"Mean bias: {comparison.bias:.6f} degC")
    print("Absolute-error percentiles:")
    print(f"  p50:  {diagnostics.p50_absolute_error:.6f} degC")
    print(f"  p90:  {diagnostics.p90_absolute_error:.6f} degC")
    print(f"  p95:  {diagnostics.p95_absolute_error:.6f} degC")
    print(f"  p99:  {diagnostics.p99_absolute_error:.6f} degC")
    print(f"  p99.9:{diagnostics.p999_absolute_error:.6f} degC")
    print("Spatial fraction above absolute-error thresholds:")
    print(f"  > 0.1 degC: {100.0 * diagnostics.fraction_above_0p1:.4f}%")
    print(f"  > 0.5 degC: {100.0 * diagnostics.fraction_above_0p5:.4f}%")
    print(f"  > 1.0 degC: {100.0 * diagnostics.fraction_above_1p0:.4f}%")
    print(f"Saved validation result to {destination}")

    if args.plots_dir:
        prefix = f"release25_{args.run_id}"
        paths = save_temperature_diagnostic_plots(
            comparison,
            Path(args.plots_dir),
            prefix=prefix,
        )
        print("Saved diagnostic plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    if args.overlay_plots_dir:
        _require_overlay_arguments(args)
        context = build_release25_validation_overlay_context(
            release25_repo=args.release25_repo,
            cnn1_dir=args.cnn1_dir,
            cnn2_dir=args.cnn2_dir,
            prepared_pki_dir=args.prepared_pki_dir,
            run_id=args.run_id,
            target_shape=comparison.shape,
            device=args.device,
            random_k=args.random_k,
            streamline_method=args.streamline_method,
        )
        paths = save_validation_overlay_plots(
            prediction=comparison.prediction,
            reference=comparison.reference,
            absolute_error=comparison.absolute_error,
            overlay_context=context,
            directory=args.overlay_plots_dir,
            prefix=f"release25_{args.run_id}",
            cell_size_m=args.cell_size_m,
            streamline_threshold=args.streamline_plot_threshold,
        )
        print("Saved validation overlay plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
