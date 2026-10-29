from __future__ import annotations

import argparse

from ..rq1 import load_rq1_config, run_rq1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the finalized RQ1 permeability-input UQ experiment: "
            "Perlin MC baseline, unconditional GRF/KL MC, conditional GRF/KL MC, "
            "and repeated scrambled-Sobol randomized QMC comparison."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="RQ1 YAML configuration. A normalized copy is written to the output root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_rq1_config(args.config)
    print(f"RQ1 experiment: {config.experiment_id}")
    print(f"Main budgets: {config.budgets}")
    print(
        "MC/RQMC comparison: "
        f"budgets={config.comparison_budgets}, repetitions={config.repetitions}"
    )
    print(f"Output root: {config.output_root}")
    artifacts = run_rq1(config)
    print("RQ1 artifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
