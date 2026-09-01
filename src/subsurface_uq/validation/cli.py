from __future__ import annotations

import argparse

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
    return parser


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
    print(f"Saved validation result to {destination}")


if __name__ == "__main__":
    main()
