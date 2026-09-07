from __future__ import annotations

import argparse

from .uq import plot_monte_carlo_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create spatial Monte Carlo/UQ plots from a saved .npz result."
    )
    parser.add_argument("--input", required=True, help="Saved Monte Carlo .npz archive.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prefix",
        help="Filename prefix. Defaults to the input archive stem.",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=5.0,
        help="Spatial grid-cell size in metres (default: 5.0 for release25).",
    )
    parser.add_argument(
        "--background-temperature",
        type=float,
        help=(
            "Optional background temperature in degC for Delta-T plots. "
            "If omitted, the value stored in result metadata is used when available."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = plot_monte_carlo_archive(
        args.input,
        args.output_dir,
        prefix=args.prefix,
        cell_size_m=args.cell_size_m,
        background_temperature=args.background_temperature,
    )
    print(f"Saved {len(paths)} Monte Carlo/UQ plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
