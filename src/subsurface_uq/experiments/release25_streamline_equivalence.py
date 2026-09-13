from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from ..io import git_head, runtime_versions, sha256_file
from ..pce import latin_hypercube_uniform_design
from ..sampling import (
    RELEASE25_PERLIN_DEFAULT_SEED,
    RELEASE25_PERLIN_FREQUENCY,
    RELEASE25_PERLIN_K_MAX,
    RELEASE25_PERLIN_K_MIN,
    PerlinCoordinatePermeabilityMap,
)
from ..surrogates.bounded_streamlines import BoundedStreamlineFactory
from ..surrogates.release25_runtime import Release25Runtime

Array = np.ndarray
EQUIVALENCE_SCHEMA_VERSION = 1


def difference_metrics(reference: Array, candidate: Array) -> dict[str, float]:
    """Return scale-aware differences between two equal-shaped finite arrays."""

    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape:
        raise ValueError(
            f"reference and candidate shapes differ: {reference.shape} != {candidate.shape}"
        )
    if not (np.all(np.isfinite(reference)) and np.all(np.isfinite(candidate))):
        raise ValueError("equivalence comparison requires finite arrays")

    error = candidate - reference
    abs_error = np.abs(error)
    reference_norm = float(np.linalg.norm(reference.ravel()))
    error_norm = float(np.linalg.norm(error.ravel()))
    relative_l2 = (
        error_norm / reference_norm
        if reference_norm > 0.0
        else (0.0 if error_norm == 0.0 else float("inf"))
    )
    return {
        "max_abs": float(np.max(abs_error)) if abs_error.size else 0.0,
        "mean_abs": float(np.mean(abs_error)) if abs_error.size else 0.0,
        "rmse": float(np.sqrt(np.mean(error**2))) if error.size else 0.0,
        "relative_l2": float(relative_l2),
    }


def select_lhs_coordinates(
    *,
    seed: int,
    design_size: int,
    sample_indices: tuple[int, ...] | list[int],
) -> Array:
    """Recreate one LHS design and select one-based sample indices from it."""

    design_size = int(design_size)
    if design_size <= 0:
        raise ValueError("design_size must be positive")
    indices = tuple(int(value) for value in sample_indices)
    if not indices:
        raise ValueError("at least one sample index is required")
    if len(set(indices)) != len(indices):
        raise ValueError("sample indices must be unique")
    if any(index < 1 or index > design_size for index in indices):
        raise ValueError(
            f"sample indices must lie in [1,{design_size}], got {indices}"
        )

    design = latin_hypercube_uniform_design(
        design_size,
        2,
        seed=int(seed),
    )
    return np.asarray(design[np.asarray(indices) - 1], dtype=np.float64)


def _to_numpy(value: torch.Tensor | Array) -> Array:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _metric_names() -> tuple[str, ...]:
    return ("max_abs", "mean_abs", "rmse", "relative_l2")


def _save_result(
    output: str | Path,
    *,
    sample_indices: Array,
    coordinates: Array,
    perlin_offsets: Array,
    release25_seconds: Array,
    bounded_seconds: Array,
    metrics: Mapping[str, Mapping[str, Array]],
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Array] = {
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "coordinates": np.asarray(coordinates, dtype=np.float64),
        "perlin_offsets": np.asarray(perlin_offsets, dtype=np.float64),
        "release25_seconds": np.asarray(release25_seconds, dtype=np.float64),
        "bounded_seconds": np.asarray(bounded_seconds, dtype=np.float64),
    }
    for quantity, quantity_metrics in metrics.items():
        for name, values in quantity_metrics.items():
            payload[f"{quantity}_{name}"] = np.asarray(values, dtype=np.float64)

    np.savez_compressed(destination, **payload)
    metadata_path = destination.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return destination, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original release25 and bounded Step-2 streamline behavior on "
            "the same reproducible non-pathological Perlin-coordinate realizations."
        )
    )
    parser.add_argument("--release25-repo", required=True)
    parser.add_argument("--cnn1-dir", required=True)
    parser.add_argument(
        "--cnn2-dir",
        required=True,
        help="LGCNN Step 3 / CNN3 model folder; legacy argument name retained.",
    )
    parser.add_argument("--prepared-pki-dir", required=True)
    parser.add_argument("--fixed-run-id", required=True)
    parser.add_argument(
        "--design-seed",
        type=int,
        default=RELEASE25_PERLIN_DEFAULT_SEED,
        help="Seed for the reconstructed two-dimensional Latin-hypercube design.",
    )
    parser.add_argument(
        "--design-size",
        type=int,
        default=4,
        help=(
            "Size of the reconstructed LHS design. Default 4 reproduces the smoke-test "
            "training design used to identify the pathological fourth sample."
        ),
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        metavar="INDEX",
        help=(
            "One-based design indices to compare. Default: 1 2 3, the known "
            "non-pathological samples from the 4-point seed-2907 smoke-test design."
        ),
    )
    parser.add_argument(
        "--frequency",
        type=float,
        nargs=2,
        metavar=("FX", "FY"),
        default=RELEASE25_PERLIN_FREQUENCY,
    )
    parser.add_argument("--k-min", type=float, default=RELEASE25_PERLIN_K_MIN)
    parser.add_argument("--k-max", type=float, default=RELEASE25_PERLIN_K_MAX)
    parser.add_argument("--x-base-shift", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--random-k", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--streamline-method", choices=("RK45", "RK23", "Radau"), default="RK45"
    )
    parser.add_argument(
        "--streamline-max-nfev",
        type=int,
        default=100_000,
        help="Per-streamline RHS watchdog for bounded mode; 0 disables it.",
    )
    parser.add_argument(
        "--streamline-diagnostics",
        action="store_true",
        help="Print bounded streamline timing and context diagnostics.",
    )
    parser.add_argument(
        "--streamline-slow-seconds",
        type=float,
        default=2.0,
        help="Threshold for bounded slow-streamline diagnostics.",
    )
    parser.add_argument(
        "--output",
        default="run_output/release25_streamline_equivalence.npz",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    runtime = Release25Runtime.from_paths(
        release25_repo=args.release25_repo,
        cnn1_dir=args.cnn1_dir,
        cnn2_dir=args.cnn2_dir,
        prepared_pki_dir=args.prepared_pki_dir,
        run_id=args.fixed_run_id,
        device=args.device,
        random_k=args.random_k,
        streamline_method=args.streamline_method,
    )
    shape = tuple(int(value) for value in runtime.scenario.shape)
    domain_size_m = (shape[0] * 5.0, shape[1] * 5.0)
    field_map = PerlinCoordinatePermeabilityMap(
        shape=shape,
        domain_size_m=domain_size_m,
        frequency=(float(args.frequency[0]), float(args.frequency[1])),
        k_min=float(args.k_min),
        k_max=float(args.k_max),
        x_base_shift=float(args.x_base_shift),
    )

    selected_indices = tuple(int(value) for value in args.sample_indices)
    coordinates = select_lhs_coordinates(
        seed=args.design_seed,
        design_size=args.design_size,
        sample_indices=selected_indices,
    )
    perlin_offsets = np.asarray(field_map.coordinates_to_offsets(coordinates), dtype=np.float64)

    original_make_streamlines = runtime.adapter._make_streamlines
    bounded_factory = BoundedStreamlineFactory(
        max_nfev=args.streamline_max_nfev,
        diagnostics=bool(args.streamline_diagnostics),
        slow_streamline_seconds=args.streamline_slow_seconds,
    )

    quantities = ("velocity", "streamline_center", "streamline_outer", "temperature")
    metrics: dict[str, dict[str, list[float]]] = {
        quantity: {name: [] for name in _metric_names()} for quantity in quantities
    }
    release25_seconds: list[float] = []
    bounded_seconds: list[float] = []

    try:
        for local_index, (design_index, coordinate, offset) in enumerate(
            zip(selected_indices, coordinates, perlin_offsets), start=1
        ):
            permeability = np.asarray(field_map.map_coordinates(coordinate), dtype=np.float32)
            k = torch.as_tensor(
                permeability,
                dtype=torch.float32,
                device=runtime.adapter.cnn1.device,
            )

            print(
                f"Equivalence sample {local_index}/{len(selected_indices)}: "
                f"design_index={design_index}, xi={np.array2string(coordinate, precision=6)}, "
                f"perlin_offset={np.array2string(offset, precision=6)}"
            )

            runtime.adapter._make_streamlines = original_make_streamlines
            started = perf_counter()
            reference = runtime.adapter.predict(k, runtime.scenario.fixed)
            release25_seconds.append(perf_counter() - started)

            bounded_factory.set_batch_context(
                [
                    {
                        "phase": "equivalence",
                        "design_index": int(design_index),
                        "xi": coordinate.tolist(),
                        "perlin_offset": offset.tolist(),
                    }
                ]
            )
            runtime.adapter._make_streamlines = bounded_factory
            started = perf_counter()
            bounded = runtime.adapter.predict(k, runtime.scenario.fixed)
            bounded_seconds.append(perf_counter() - started)

            reference_velocity = _to_numpy(reference["velocity"])
            bounded_velocity = _to_numpy(bounded["velocity"])
            reference_streamlines = _to_numpy(reference["streamlines"])
            bounded_streamlines = _to_numpy(bounded["streamlines"])
            reference_temperature = _to_numpy(reference["temperature"])
            bounded_temperature = _to_numpy(bounded["temperature"])

            comparisons = {
                "velocity": difference_metrics(reference_velocity, bounded_velocity),
                "streamline_center": difference_metrics(
                    reference_streamlines[0], bounded_streamlines[0]
                ),
                "streamline_outer": difference_metrics(
                    reference_streamlines[1], bounded_streamlines[1]
                ),
                "temperature": difference_metrics(
                    reference_temperature, bounded_temperature
                ),
            }
            for quantity, quantity_metrics in comparisons.items():
                for name, value in quantity_metrics.items():
                    metrics[quantity][name].append(float(value))

            print(
                "  temperature: "
                f"rel_L2={comparisons['temperature']['relative_l2']:.6g}, "
                f"RMSE={comparisons['temperature']['rmse']:.6g}, "
                f"max_abs={comparisons['temperature']['max_abs']:.6g}"
            )
            print(
                "  streamlines: "
                f"center_rel_L2={comparisons['streamline_center']['relative_l2']:.6g}, "
                f"outer_rel_L2={comparisons['streamline_outer']['relative_l2']:.6g}"
            )
    finally:
        runtime.adapter._make_streamlines = original_make_streamlines

    metric_arrays: dict[str, dict[str, Array]] = {
        quantity: {
            name: np.asarray(values, dtype=np.float64)
            for name, values in quantity_metrics.items()
        }
        for quantity, quantity_metrics in metrics.items()
    }

    project_root = Path(__file__).resolve().parents[3]
    metadata = {
        "schema_version": EQUIVALENCE_SCHEMA_VERSION,
        "experiment": "release25_bounded_streamline_equivalence",
        "interpretation": (
            "Paired comparison of original release25 and bounded Step-2 behavior on "
            "identical permeability fields. No tolerance is imposed automatically; "
            "the recorded differences are intended to justify or reject bounded-mode "
            "equivalence empirically."
        ),
        "design": {
            "type": "randomized_latin_hypercube_uniform_minus1_1",
            "seed": int(args.design_seed),
            "design_size": int(args.design_size),
            "selected_one_based_indices": [int(value) for value in selected_indices],
        },
        "field_map": field_map.metadata,
        "fixed_run_id": args.fixed_run_id,
        "release25_repo": str(Path(args.release25_repo)),
        "cnn1_dir": str(Path(args.cnn1_dir)),
        "cnn2_dir": str(Path(args.cnn2_dir)),
        "prepared_pki_dir": str(Path(args.prepared_pki_dir)),
        "device": args.device,
        "random_k": bool(args.random_k),
        "streamline_method": args.streamline_method,
        "bounded": {
            "max_nfev": int(args.streamline_max_nfev),
            "diagnostics": bool(args.streamline_diagnostics),
            "slow_streamline_seconds": float(args.streamline_slow_seconds),
        },
        "runtime": runtime_versions(),
        "provenance": {
            "masterthesis_git_sha": git_head(project_root),
            "release25_git_sha": git_head(args.release25_repo),
            "cnn1_checkpoint_sha256": sha256_file(runtime.adapter.cnn1.checkpoint),
            "cnn3_checkpoint_sha256": sha256_file(runtime.adapter.cnn2.checkpoint),
        },
    }

    destination, metadata_path = _save_result(
        args.output,
        sample_indices=np.asarray(selected_indices, dtype=np.int64),
        coordinates=coordinates,
        perlin_offsets=perlin_offsets,
        release25_seconds=np.asarray(release25_seconds),
        bounded_seconds=np.asarray(bounded_seconds),
        metrics=metric_arrays,
        metadata=metadata,
    )

    print(f"Saved equivalence result to {destination}")
    print(f"Saved reproducibility metadata to {metadata_path}")
    print(
        "Summary: "
        f"max temperature relative_L2={float(np.max(metric_arrays['temperature']['relative_l2'])):.6g}, "
        f"max temperature max_abs={float(np.max(metric_arrays['temperature']['max_abs'])):.6g}, "
        f"max center-streamline relative_L2={float(np.max(metric_arrays['streamline_center']['relative_l2'])):.6g}, "
        f"max outer-streamline relative_L2={float(np.max(metric_arrays['streamline_outer']['relative_l2'])):.6g}"
    )
    print(
        "Runtime means: "
        f"release25={float(np.mean(release25_seconds)):.3f}s, "
        f"bounded={float(np.mean(bounded_seconds)):.3f}s"
    )


if __name__ == "__main__":
    main()
