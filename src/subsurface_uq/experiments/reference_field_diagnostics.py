"""Validate generated permeability against one fixed reference and its declared law.

Read existing generated NPY fields, or draw an explicit MC/RQMC validation design
from a portable reference-field artifact. No temperature prediction is executed.
Every supplied/generated field contributes; diagnostics never select realizations.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.input_model import load_conditional_kl_input_model
from ..sampling.reference_field import ReferenceFieldLogGaussianPermeabilityMap
from ..sampling.reference_coherence import (
    reference_ensemble_coherence, reference_probe_law_diagnostics,
)
from ..visualization.reference_fields import reference_field_agreement
from ..visualization.reference_fields_cli import _coordinates
from ..visualization.reference_validation import write_reference_validation_report
from .reference_field_permeability import coarsen_reference


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _native_marginal(field):
    logs = np.log10(np.asarray(field, dtype=np.float64))
    scale = float(np.max(field))
    normalized = np.asarray(field, dtype=float) / scale
    return {
        "log10_mean": float(logs.mean()),
        "log10_std": float(logs.std(ddof=1)) if logs.size > 1 else 0.,
        "log10_variance": float(logs.var(ddof=1)) if logs.size > 1 else 0.,
        "arithmetic_mean_m2": float(normalized.mean() * scale),
        "arithmetic_std_m2": float(normalized.std(ddof=1) * scale) if logs.size > 1 else 0.,
        "range_m2": [float(np.min(field)), float(np.max(field))],
        "quantile_probabilities": [.01, .05, .5, .95, .99],
        "quantiles_m2": np.quantile(field, [.01, .05, .5, .95, .99]).tolist(),
        "pixel_count": int(logs.size),
    }


def _probe_indices(shape, size, *, length_scale_m=None, cell_size_m=None):
    """Preselect global coverage plus native cells at declared correlation scales."""
    axes = [np.unique(np.linspace(0, n - 1, min(size, n), dtype=int)) for n in shape]
    rows, cols = np.meshgrid(*axes, indexing="ij")
    indices = np.column_stack((rows.ravel(), cols.ravel()))
    if length_scale_m is None and cell_size_m is None:
        return indices
    lengths = np.asarray(length_scale_m, dtype=float)
    if (lengths.shape != (2,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0)
            or cell_size_m is None or not np.isfinite(cell_size_m) or cell_size_m <= 0):
        raise ValueError("probe stencil requires positive finite row/column lengths and cell_size_m")
    center = np.asarray(shape, dtype=int) // 2
    stencil = [center]
    for fraction in (.5, 1., 2.):
        offsets = np.maximum(1, np.rint(fraction * lengths / cell_size_m).astype(int))
        for axis in range(2):
            for sign in (-1, 1):
                point = center.copy()
                point[axis] += sign * offsets[axis]
                stencil.append(point)
        if fraction <= 1.:
            for row_sign in (-1, 1):
                for col_sign in (-1, 1):
                    stencil.append(center + offsets * [row_sign, col_sign])
    local = np.clip(np.asarray(stencil), 0, np.asarray(shape) - 1)
    return np.unique(np.concatenate((indices, local)), axis=0)


def _load_criteria(criteria):
    if isinstance(criteria, (str, Path)):
        criteria = yaml.safe_load(Path(criteria).read_text(encoding="utf-8"))
    if criteria is None:
        return {}
    allowed = {
        "max_abs_log10_bias", "max_log10_rmse", "min_log10_spatial_correlation",
        "max_marginal_wasserstein_log10", "max_directional_variogram_relative_rmse",
        "max_radial_power_relative_l2", "max_angular_power_relative_l2",
    }
    if not isinstance(criteria, dict) or not set(criteria) <= allowed:
        raise ValueError("unknown reference coherence criteria; use documented metric names")
    result = {}
    for key, value in criteria.items():
        if isinstance(value, bool) or value is None:
            raise ValueError("coherence criteria must contain finite numeric thresholds")
        value = float(value)
        if not np.isfinite(value) or (key.startswith("max_") and value < 0):
            raise ValueError("coherence criteria must be finite and non-negative for maximum limits")
        if key.startswith("min_") and not -1 <= value <= 1:
            raise ValueError("minimum correlation must be between -1 and 1")
        result[key] = value
    return result


def _assess_criteria(criteria, coherence, agreement):
    fidelity = coherence["reference_fidelity"]
    correlations = [r["log10_spatial_correlation"] for r in agreement]
    directional = [row["relative_rmse"] for row in fidelity["directional_variograms"].values()]
    observed = {
        "max_abs_log10_bias": max(abs(r["log10_bias"]) for r in agreement),
        "max_log10_rmse": max(r["log10_rmse"] for r in agreement),
        "min_log10_spatial_correlation": min(correlations) if all(v is not None for v in correlations) else None,
        "max_marginal_wasserstein_log10": fidelity["marginal"]["wasserstein_distance_log10"],
        "max_directional_variogram_relative_rmse": max(directional) if all(v is not None for v in directional) else None,
        "max_radial_power_relative_l2": fidelity["spectra"]["radial_power_relative_l2"],
        "max_angular_power_relative_l2": fidelity["spectra"]["angular_power_relative_l2"],
    }
    checks = {}
    for name, threshold in criteria.items():
        value = observed[name]
        satisfied = None if value is None else bool(value >= threshold if name.startswith("min_") else value <= threshold)
        checks[name] = {"observed": value, "threshold": threshold, "satisfied": satisfied}
    flags = [row["satisfied"] for row in checks.values()]
    status = ("not_configured" if not flags else
              "deviations_observed" if False in flags else
              "partly_unassessed" if None in flags else "configured_checks_satisfied")
    return {"status": status, "checks": checks,
            "interpretation": "User-specified scenario review assumptions; no statistical calibration, sample filtering or automatic UQ approval."}


def validate_reference_fields(input_model_path, output_dir, *, generated_fields=None,
                              n_samples=None, seed=None, method=None,
                              diagnostic_factor=8, probe_grid_size=4,
                              max_lag_cells=24, criteria=None):
    """Stream native fields; retain only matched-support diagnostics and fixed probes."""
    for value, name in ((diagnostic_factor, "diagnostic_factor"),
                        (probe_grid_size, "probe_grid_size"), (max_lag_cells, "max_lag_cells")):
        _positive_integer(value, name)
    if probe_grid_size > 16:
        raise ValueError("probe_grid_size must not exceed 16 (256 global grid probes plus the local stencil)")
    criteria = _load_criteria(criteria)
    if method not in (None, "MC", "RQMC"):
        raise ValueError("method must be MC or RQMC")
    artifact = Path(input_model_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("validation output is nonempty; choose a fresh directory")
    if generated_fields is None:
        _positive_integer(n_samples, "n_samples")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or method is None:
            raise ValueError("new validation designs require explicit non-negative seed and MC/RQMC method")
        if method == "RQMC" and n_samples & (n_samples - 1):
            raise ValueError("RQMC requires a power-of-two design size")
    elif n_samples is not None or seed is not None:
        raise ValueError("existing fields determine their sample count; do not override n_samples or seed")
    model = load_conditional_kl_input_model(artifact)
    field_map = model.conditional
    if not isinstance(field_map, ReferenceFieldLogGaussianPermeabilityMap):
        raise ValueError("validation requires a reference-centered permeability artifact")
    reference = field_map.reference_permeability
    spacing = np.asarray(field_map.domain_size_m) / np.asarray(reference.shape)
    if not np.allclose(spacing, spacing[0]):
        raise ValueError("validation requires square cells on the declared reference grid")
    cell_size = float(spacing[0])
    coarse_reference = coarsen_reference(reference, diagnostic_factor)
    if min(coarse_reference.shape) < 3:
        raise ValueError("diagnostic grid must have at least three cells per direction; reduce diagnostic_factor")
    indices = _probe_indices(reference.shape, probe_grid_size,
                             length_scale_m=field_map.prior.length_scale_m, cell_size_m=cell_size)
    source = {"input_model": str(artifact), "input_model_sha256": sha256_file(artifact),
              "reference_sha256": sha256(np.ascontiguousarray(reference).tobytes()).hexdigest(),
              "scenario": model.payload.get("scenario"),
              "source_metadata": model.payload.get("source_metadata", {}),
              "center": field_map.center, "dimension": field_map.dimension,
              "reference_run": (model.payload.get("source_metadata", {}).get("reference_run")
                                or (model.payload.get("scenario") or {}).get("config", {}).get("reference_run")),
              "sampling_method": method or "unspecified", "seed": seed,
              "coordinate_distribution": "standard_normal",
              "sample_filtering": None, "scenario_pooling": False}
    if generated_fields is not None:
        fields_path = Path(generated_fields).expanduser().resolve()
        fields = np.load(fields_path, mmap_mode="r", allow_pickle=False)
        if fields.ndim != 3 or fields.shape[0] < 1 or fields.shape[1:] != reference.shape:
            raise ValueError("generated_fields must have nonempty shape [N,H,W] matching the artifact reference")
        source.update(mode="existing_fields", generated_fields=str(fields_path),
                      generated_fields_sha256=sha256_file(fields_path),
                      design_provenance="Sampling method, if supplied, is a user assertion; field provenance is not inferred from array shape.")
        total_fields = len(fields)
    else:
        coordinates = _coordinates(field_map.dimension, n_samples, seed, method)
        fields = (field_map.map_coordinates(row) for row in coordinates)
        source.update(mode="validation_design", coordinate_design_sha256=sha256(coordinates.tobytes()).hexdigest(),
                      design_provenance="Explicit reproducible validation design, not a replay of previously queried LGCNN inputs.")
        total_fields = n_samples
    agreement, marginals, coarse, probes, hashes = [], [], [], [], []
    for index, field in enumerate(fields):
        if not np.isrealobj(field) or not np.issubdtype(field.dtype, np.number):
            raise ValueError("generated permeability fields must contain real numeric values")
        # This existing full-grid comparison rejects invalid numerical inputs,
        # never statistically atypical draws from a valid declared law.
        agreement.append({"sample_index": index, **reference_field_agreement(reference, field)})
        marginals.append({"sample_index": index, **_native_marginal(field)})
        hashes.append({"sample_index": index, "dtype": str(field.dtype),
                       "physical_field_sha256": sha256(np.ascontiguousarray(field).tobytes()).hexdigest()})
        coarse.append(coarsen_reference(field, diagnostic_factor))
        probes.append(np.log10(np.asarray(field[indices[:, 0], indices[:, 1]], dtype=float)))
        if (index + 1) % 8 == 0 or index + 1 == total_fields:
            print(f"Processed permeability field {index + 1}/{total_fields}", flush=True)
    count = len(coarse)
    source.update(n_samples=count, fields=hashes)
    coherence = reference_ensemble_coherence(coarse_reference, np.asarray(coarse),
        cell_size_m=cell_size * diagnostic_factor, max_lag_cells=max_lag_cells)
    probe_law = reference_probe_law_diagnostics(field_map, np.asarray(probes), indices,
                                                sampling_method=method or "unspecified")
    report = {
        "schema_version": 1, "output_type": "permeability input coherence diagnostics",
        "source": source,
        "support": {"native_shape": list(reference.shape), "native_cell_size_m": cell_size,
                    "diagnostic_shape": list(coarse_reference.shape),
                    "diagnostic_cell_size_m": cell_size * diagnostic_factor,
                    "diagnostic_factor": diagnostic_factor, "coarsening": "geometric_block_mean",
                    "probe_support": "predetermined native cells; no block averaging",
                    "probe_selection": {
                        "strategy": "global_uniform_grid_plus_declared_length_scale_stencil",
                        "selected_before_seeing_fields": True,
                        "global_grid_size_per_axis": probe_grid_size,
                        "local_center_yx": (np.asarray(reference.shape) // 2).tolist(),
                        "declared_row_column_length_scales_m": list(field_map.prior.length_scale_m),
                        "axial_length_scale_fractions": [.5, 1., 2.],
                        "diagonal_length_scale_fractions": [.5, 1.],
                        "offset_rule": "round to nearest native cell, minimum one; clip to grid; deduplicate",
                        "probe_count": len(indices),
                        "interpretation": "Global coverage and local dependence scales are checked together; stencil uses declared lengths, not fitted correlations.",
                    },
                    "probe_indices_yx": indices.tolist(),
                    "axes": "array row/column, not inferred geographic orientation"},
        "native_marginals": {"reference": _native_marginal(reference), "generated_per_field": marginals},
        "native_agreement": agreement, "coherence": coherence, "probe_law": probe_law,
        "assessment": {"status": "reference_review_required",
            "configured_reference_checks": _assess_criteria(criteria, coherence, agreement),
            "ensemble_resolution": "single_realization_only" if count == 1 else "finite_ensemble_descriptive",
            "training_support": {"status": "unassessed_by_reference_comparison",
                "explicit_profile_present": model.training_k_range is not None,
                "interpretation": "Reference resemblance and agreement with an assumed law do not establish LGCNN training support; inspect proven training-patch diagnostics separately."},
            "temperature_qoi_convergence": "not_evaluated",
            "interpretation": "Inspect all diagnostic dimensions and scenario assumptions before propagation; no universal pass/fail threshold or calibrated geological truth is inferred."},
        "no_filtering": True, "runtime": runtime_versions(("numpy", "scipy", "matplotlib")),
        "implementation": {"git_head": git_head(Path(__file__).resolve().parents[3]),
                           "command_source_sha256": sha256_file(Path(__file__)),
                           "coherence_source_sha256": sha256_file(Path(__file__).parents[1] / "sampling/reference_coherence.py")},
    }
    # All scientific metrics must be finite or explicitly unassessed before
    # materializing any output. Keep supplied fields and the artifact untouched.
    json.dumps(report, allow_nan=False)
    root.mkdir(parents=True, exist_ok=True)
    report["outputs"] = write_reference_validation_report(report, root)
    (root / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True, help="Portable artifact containing the exact reference and retained KL law")
    parser.add_argument("--generated-fields", help="Existing positive [N,H,W] permeability NPY; all fields are diagnosed")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=("MC", "RQMC"))
    parser.add_argument("--diagnostic-factor", type=int, default=8)
    parser.add_argument("--probe-grid-size", type=int, default=4,
                        help="Global native probe rows/columns (at most 16 each), augmented by up to 21 declared-length-scale stencil cells")
    parser.add_argument("--max-lag-cells", type=int, default=24)
    parser.add_argument("--criteria", help="Optional explicit reference-review thresholds in YAML")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = validate_reference_fields(args.input_model, args.output_dir,
        generated_fields=args.generated_fields, n_samples=args.n_samples, seed=args.seed,
        method=args.method, diagnostic_factor=args.diagnostic_factor,
        probe_grid_size=args.probe_grid_size, max_lag_cells=args.max_lag_cells, criteria=args.criteria)
    print(f"Diagnosed all {report['source']['n_samples']} permeability fields; report: {Path(args.output_dir).resolve() / 'validation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
