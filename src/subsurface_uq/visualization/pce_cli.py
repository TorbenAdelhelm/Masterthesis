from __future__ import annotations

import argparse

from .pce import plot_pce_archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create compact validation/diagnostic plots from a saved PCE .npz result."
    )
    parser.add_argument("--input", required=True, help="Saved PCE .npz archive.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prefix",
        help="Filename prefix. Defaults to the input archive stem.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = plot_pce_archive(
        args.input,
        args.output_dir,
        prefix=args.prefix,
    )
    print(f"Saved {len(paths)} PCE plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
