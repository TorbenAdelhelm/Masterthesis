"""Propagate a reduced reference-scenario study through the frozen real-K LGCNN.

The primary matrix remains separate by scenario. Three nested KL maps share
Gaussian coordinates; their 0.95 anchor is reused in the primary matrix.
Checkpoints load once, samples run individually, and completed QoIs are cached.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import torch
import yaml

import subsurface_uq
from subsurface_uq.experiments.reference_kl_sensitivity import compare_qoi_truncations
from subsurface_uq.io import git_head, runtime_versions, sha256_file
from subsurface_uq.rq1.config import load_rq1_config
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.training_compatibility import (
    characterize_training_patch_distribution, TrainingPatchCompatibilityDiagnostics,
)
from subsurface_uq.surrogates import Release25Surrogate
from subsurface_uq.surrogates.bounded_streamlines import configure_release25_streamlines
from subsurface_uq.surrogates.release25_runtime import Release25Runtime


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _write_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def _manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"scenario manifest has no scenarios: {path}")
    result = []
    for row in rows:
        artifact = Path(row["input_model"])
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        row = {**row, "input_model": str(artifact.resolve())}
        result.append(row)
    ids = [row["scenario_id"] for row in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate scenario IDs in {path}")
    return path, result


def _anchor(row):
    config = row["config"]
    return (config["reference_run"] == "RUN_2"
            and np.isclose(config["sigma_R"], .05, rtol=0, atol=1e-12)
            and config["covariance_family"] == "matern32"
            and config["n_modes"] is None
            and np.isclose(config["energy_threshold"], .95, rtol=0, atol=1e-12))


def _artifact_context(row, expected_shape):
    artifact = Path(row["input_model"])
    digest = sha256_file(artifact)
    if row.get("input_model_sha256") not in (None, digest):
        raise ValueError(f"manifest/artifact checksum mismatch for {row['scenario_id']}")
    payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
    if (payload.get("input_law") not in {"reference-centered-lognormal", "reference-centered-lognormal-candidate"}
            or payload.get("coordinate_distribution") != "iid_standard_normal"
            or payload.get("conditioning")):
        raise ValueError("pilot requires unconditioned Gaussian reference-field artifacts")
    if (payload["scenario"]["scenario_id"] != row["scenario_id"]
            or payload["scenario"]["config"] != row["config"]):
        raise ValueError("scenario manifest disagrees with its input artifact")
    shape = tuple(payload["field_shape"])
    domain = np.asarray(payload["prior"]["domain_size_m"], dtype=float)
    if (shape != expected_shape or not np.allclose(domain, np.asarray(shape) * 5., rtol=0, atol=1e-8)):
        raise ValueError("all artifacts must match the fixed full-resolution 5 m grid and domain")
    reference = artifact.parent / payload["reference_field"]["path"]
    reference_context = _file(reference)
    if reference_context["sha256"] != payload["reference_field"]["sha256"]:
        raise ValueError("reference companion checksum mismatch")
    return {"scenario_id": row["scenario_id"], "config": row["config"],
            "input_model": {"path": str(artifact), "sha256": digest},
            "reference_companion": reference_context}


def _coordinates(dimension, count, seed, method):
    if method == "MC":
        return np.random.default_rng(seed).normal(size=(count, dimension))
    uniforms = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return ndtri(np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.)))


def _moments(values):
    return {"sample_count": len(values), "mean": values.mean(axis=0).tolist(),
            "std_ddof1": values.std(axis=0, ddof=1).tolist(),
            "quantiles_05_50_95": np.quantile(values, [.05, .5, .95], axis=0).tolist()}


def _budgets(values):
    result = {str(n): _moments(values[:n]) for n in sorted({len(values), *[n for n in (4, 8) if n <= len(values)]})}
    if len(values) >= 8:
        smaller, larger = values[:4], values[:8]
        result["budget_4_vs_8"] = {
            "mean_difference_4_minus_8": (smaller.mean(axis=0) - larger.mean(axis=0)).tolist(),
            "std_difference_4_minus_8": (smaller.std(axis=0, ddof=1) - larger.std(axis=0, ddof=1)).tolist(),
            "quantile_difference_4_minus_8": (np.quantile(smaller, [.05, .5, .95], axis=0)
                                             - np.quantile(larger, [.05, .5, .95], axis=0)).tolist(),
            "interpretation": "small nested pilot budgets; neither budget establishes converged uncertainty",
        }
    else:
        result["budget_4_vs_8"] = {"status": "unavailable: fewer than eight samples"}
    return result


def _reference_labels(value):
    if isinstance(value, dict):
        return {str(key).replace("training", "reference"): _reference_labels(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_reference_labels(item) for item in value]
    if isinstance(value, str):
        return value.replace("training", "reference")
    return value


def _reference_profile(profile):
    # This reused diagnostic class carries DOI/split assertions for actual
    # training profiles. These three nominal references have no such verified
    # membership, so preserve only their descriptive geometry and statistics.
    raw = profile.to_dict()
    keys = ("field_count", "field_names", "field_shape", "cell_size_m",
            "patch_extraction", "minimum_k_m2", "maximum_k_m2", "metrics")
    return {**{key: _reference_labels(raw[key]) for key in keys},
            "interpretation": (
                "Supplied nominal-reference patch envelope; actual training membership is unverified. "
                "Patches are native 1280 x 1280 cells at 5 m, with overlapping starts every 8 cells. "
                "Feature stride 8 decimates only the descriptive metrics, not the CNN input. "
                "Correlated patches are not independent geological draws; no sample acceptance rule.")}


class _CaptureVelocityGrid:
    """Observe the grid used to crop physical permeability for CNN3."""
    def __init__(self, adapter):
        self.adapter = adapter
        self.velocity_shape = None

    def predict(self, permeability, fixed):
        result = self.adapter.predict(permeability, fixed)
        self.velocity_shape = tuple(int(n) for n in result["velocity"].shape[-2:])
        return result


def _normalization_range(field, low, high):
    return {"stored_min_m2": low, "stored_max_m2": high,
            "fraction_pixels_below_min": float(np.mean(field < low)),
            "fraction_pixels_above_max": float(np.mean(field > high)),
            "fraction_pixels_outside_minmax": float(np.mean((field < low) | (field > high))),
            "evaluated_grid_shape": list(field.shape)}


def run_pilot(args):
    if args.n_samples < 2 or args.seed < 0:
        raise ValueError("at least two samples and a non-negative seed are required")
    if args.method == "RQMC" and args.n_samples & (args.n_samples - 1):
        raise ValueError("RQMC sample count must be a power of two")
    matrix_path, primary = _manifest(args.matrix_manifest)
    kl_path, kl_rows = _manifest(args.kl_manifest)
    anchors = [row for row in primary if _anchor(row)]
    if len(primary) != 12 or len(anchors) != 1:
        raise ValueError("this reduced study expects twelve primary scenarios and one RUN_2/.05/matern32/.95 anchor")
    anchor = anchors[0]
    extras = [row for row in kl_rows if not _anchor(row)]
    if len(extras) != 2:
        raise ValueError("KL manifest must supply exactly two additional truncations")
    extras.sort(key=lambda row: row["config"]["energy_threshold"] or 0.)
    if not np.allclose([row["config"]["energy_threshold"] for row in extras], [.99, .999], rtol=0, atol=1e-12):
        raise ValueError("additional KL energy targets must be 0.99 and 0.999")
    paired_rows = [anchor, *extras]
    all_rows = [*primary, *extras]
    if len({row["scenario_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("additional KL scenarios must have distinct IDs from the primary matrix")
    config = load_rq1_config(args.rq1_config, input_model_override=anchor["input_model"])
    if config.random_k or config.cell_size_m != 5.:
        raise ValueError("pilot requires real-K streamline semantics and the original 5 m grid")
    qoi_names = [f"temperature_{r}_{c}" for r, c in config.receptors]
    if config.mean_anomaly_roi is not None:
        qoi_names.append("mean_temperature_anomaly")
    if not qoi_names:
        raise ValueError("configure at least one temperature QoI")
    prepared_info = config.prepared_pki_dir / "info.yaml"
    metadata = yaml.safe_load(prepared_info.read_text(encoding="utf-8"))
    expected_shape = tuple(metadata["CellsNumber"])
    if not np.allclose(metadata["CellsSize"][:2], (5., 5.), rtol=0, atol=0):
        raise ValueError("fixed prepared inputs must be on the original 5 m grid")
    prepared_candidates = [config.prepared_pki_dir / "Inputs" / config.fixed_run_id,
                           config.prepared_pki_dir / "Inputs" / f"{config.fixed_run_id}.pt"]
    prepared_tensors = [path for path in prepared_candidates if path.is_file()]
    if len(prepared_tensors) != 1:
        raise ValueError("expected one prepared tensor for the selected fixed RUN")
    artifacts = [_artifact_context(row, expected_shape) for row in all_rows]
    model_files = []
    for name, directory in (("cnn1", config.cnn1_dir), ("cnn3", config.cnn2_dir)):
        for filename in ("model.pt", "info.yaml", "HPS_options.yaml"):
            model_files.append({"role": name, **_file(directory / filename)})
    code = config.release25_repo / "code"
    if not code.is_dir():
        code = config.release25_repo
    package = Path(subsurface_uq.__file__).resolve().parent
    reference_paths = [Path(path).resolve() for path in (args.reference_arrays or [])]
    context = {
        "schema_version": 1, "script": _file(Path(__file__)),
        "matrix_manifest": _file(matrix_path), "kl_manifest": _file(kl_path),
        "rq1_config_source": _file(args.rq1_config), "rq1_config_resolved": config.to_dict(),
        "model_files": model_files, "artifacts": artifacts,
        "prepared_fixed_inputs": [_file(prepared_info), _file(prepared_tensors[0])],
        "upstream_git_sha": git_head(config.release25_repo),
        "upstream_python_sha256": {str(path.relative_to(code)): sha256_file(path) for path in sorted(code.rglob("*.py"))},
        "package_python_sha256": {str(path.relative_to(package)): sha256_file(path) for path in sorted(package.rglob("*.py"))},
        "masterthesis_git_sha": git_head(Path(__file__).resolve().parents[2]),
        "runtime": runtime_versions(("numpy", "torch", "scipy", "PyYAML")),
        "torch_threads": 8, "batch_size": 1, "n_samples": args.n_samples,
        "seed": args.seed, "method": args.method, "qoi_names": qoi_names,
        "reference_patch_comparison": {
            "sources": [_file(path) for path in reference_paths],
            "box_size": 1280, "skip_per_dir": 8, "feature_stride": 8, "max_patches": 192,
            "native_cell_size_m": 5., "patch_start_stride_cells": 8,
            "interpretation": "supplied native 1280-cell reference patch envelope only; start stride 8, diagnostic feature stride 8; actual training membership is unverified; no acceptance/filtering",
        },
        "scenario_policy": "separate results; no scenario pooling, weights or sample rejection",
    }
    context_id = sha256(_canonical(context).encode()).hexdigest()
    root = Path(args.output_dir).resolve()
    context_path = root / "run_context.json"
    if context_path.exists():
        previous = json.loads(context_path.read_text(encoding="utf-8"))
        if previous != {"context_id": context_id, "context": context}:
            raise ValueError("output directory belongs to a different immutable run context; choose a new output directory")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("nonempty output directory has no run context; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    if not context_path.exists():
        _write_json(context_path, {"context_id": context_id, "context": context})
    cache_root = root / "sample_cache" / context_id
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    runtime = Release25Runtime.from_paths(
        release25_repo=config.release25_repo, cnn1_dir=config.cnn1_dir, cnn2_dir=config.cnn2_dir,
        prepared_pki_dir=config.prepared_pki_dir, run_id=config.fixed_run_id,
        device=config.device, random_k=config.random_k, streamline_method=config.streamline_method)
    if tuple(runtime.scenario.shape) != expected_shape:
        raise ValueError("prepared tensor does not match its declared full grid")
    configure_release25_streamlines(runtime.adapter, mode=config.streamline_mode,
                                    max_nfev=config.streamline_max_nfev, diagnostics=config.streamline_diagnostics,
                                    slow_streamline_seconds=config.streamline_slow_seconds)
    observer = _CaptureVelocityGrid(runtime.adapter)
    surrogate = Release25Surrogate(observer, runtime.scenario.fixed, device=config.device)
    bounds = {}
    for role, bundle in (("cnn1", runtime.adapter.cnn1), ("cnn3", runtime.adapter.cnn2)):
        candidates = [value for key, value in bundle.info["Inputs"].items() if "permeability" in key.lower()]
        if len(candidates) != 1:
            raise ValueError(f"{role} must have one permeability normalization channel")
        bounds[role] = (float(candidates[0]["min"]), float(candidates[0]["max"]))
    profile = None
    if reference_paths:
        fields = np.stack([np.load(path, allow_pickle=False) for path in reference_paths])
        if fields.shape[1:] != expected_shape:
            raise ValueError("reference patch arrays must match the full model grid")
        profile = characterize_training_patch_distribution(
            fields, cell_size_m=5., box_size=1280, skip_per_dir=8, feature_stride=8,
            max_patches=192, field_names=[str(path) for path in reference_paths])
        del fields
        _write_json(root / "reference_patch_profile.json", {
            "profile": _reference_profile(profile),
            "interpretation": context["reference_patch_comparison"]["interpretation"]})
    patches = {row["scenario_id"]: TrainingPatchCompatibilityDiagnostics(profile) for row in all_rows} if profile else {}
    records = {row["scenario_id"]: [] for row in all_rows}
    counters = {"completed": 0, "predicted": 0, "cache_hits": 0}
    active = {"scenario_id": None, "sample_index": 0}
    total = len(all_rows) * args.n_samples

    def evaluate(field):
        scenario_id = active["scenario_id"]
        field = np.ascontiguousarray(field, dtype=np.float32)
        field_digest = sha256(field.tobytes(order="C")).hexdigest()
        cache_path = cache_root / f"{field_digest}.json"
        if cache_path.exists():
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            checksum = record.pop("result_sha256", None)
            if (checksum != sha256(_canonical(record).encode()).hexdigest()
                    or record.get("context_id") != context_id or record.get("physical_field_sha256") != field_digest
                    or record.get("physical_field_shape") != list(field.shape)
                    or len(record.get("qoi", [])) != len(qoi_names)
                    or not np.all(np.isfinite(record["qoi"]))):
                raise ValueError(f"invalid cached QoI record: {cache_path}")
            counters["cache_hits"] += 1
            status = "cached"
        else:
            started = time.perf_counter()
            temperature = surrogate.predict_temperature(field)
            if any(not (0 <= row < temperature.shape[0] and 0 <= col < temperature.shape[1])
                   for row, col in config.receptors):
                raise ValueError("configured temperature receptor lies outside the actual output grid")
            values = [float(temperature[row, col]) for row, col in config.receptors]
            if config.mean_anomaly_roi is not None:
                r0, r1, c0, c1 = config.mean_anomaly_roi
                if not (0 <= r0 < r1 <= temperature.shape[0] and 0 <= c0 < c1 <= temperature.shape[1]):
                    raise ValueError("configured temperature ROI lies outside the actual output grid")
                values.append(float(temperature[r0:r1, c0:c1].mean() - config.background_temperature))
            height, width = observer.velocity_shape
            y0, x0 = (field.shape[0] - height) // 2, (field.shape[1] - width) // 2
            cnn3_field = field[y0:y0 + height, x0:x0 + width]
            record = {
                "context_id": context_id, "physical_field_sha256": field_digest,
                "physical_field_shape": list(field.shape), "physical_field_dtype": "float32",
                "temperature_grid_shape": list(temperature.shape), "qoi": values,
                "prediction_seconds": time.perf_counter() - started,
                "permeability_spatial_descriptors": {
                    "range_m2": [float(field.min()), float(field.max())],
                    "quantiles_01_50_99_m2": np.quantile(field, [.01, .5, .99]).tolist(),
                    "interpretation": "spatial descriptors of one heterogeneous realization; not pooled RUN probabilities",
                },
                "normalization_range_diagnostic": {
                    "cnn1": _normalization_range(field, *bounds["cnn1"]),
                    "cnn3": _normalization_range(cnn3_field, *bounds["cnn3"]),
                    "interpretation": "outside stored checkpoint normalization min/max; not a full training-support test or rejection rule",
                },
            }
            _write_json(cache_path, {**record, "result_sha256": sha256(_canonical(record).encode()).hexdigest()})
            counters["predicted"] += 1
            status = "predicted"
        if patches:
            patches[scenario_id].update(field[None, ...])
        records[scenario_id].append({"sample_index": active["sample_index"], **record})
        active["sample_index"] += 1
        counters["completed"] += 1
        print(f"[{counters['completed']}/{total}] {scenario_id} sample {active['sample_index']}/{args.n_samples} {status}", flush=True)
        return np.asarray(record["qoi"], dtype=float)

    paired_models = [load_conditional_kl_input_model(row["input_model"]) for row in paired_rows]
    for model in paired_models:
        if model.prior.field_shape != expected_shape:
            raise ValueError("reconstructed KL grid mismatch")
    paired_index = 0
    def evaluate_paired(field):
        nonlocal paired_index
        active["scenario_id"] = paired_rows[paired_index // args.n_samples]["scenario_id"]
        active["sample_index"] = paired_index % args.n_samples
        result = evaluate(field)
        paired_index += 1
        return result
    truncation, paired_values = compare_qoi_truncations(
        [model.unconditional for model in paired_models], evaluate_paired,
        n_samples=args.n_samples, seed=args.seed, method=args.method)
    truncation.update(context_id=context_id, qoi_names=qoi_names,
                      scenario_ids=[row["scenario_id"] for row in paired_rows],
                      input_models=[row["input_model"] for row in paired_rows])
    _write_array(root / "paired_temperature_qoi.npy", paired_values)
    _write_json(root / "qoi_truncation.json", truncation)
    samples = {row["scenario_id"]: values for row, values in zip(paired_rows, paired_values)}
    del paired_models
    gc.collect()
    for row in primary:
        scenario_id = row["scenario_id"]
        if scenario_id in samples:
            continue
        model = load_conditional_kl_input_model(row["input_model"])
        if model.prior.field_shape != expected_shape:
            raise ValueError("reconstructed scenario grid mismatch")
        coordinates = _coordinates(model.unconditional.dimension, args.n_samples, args.seed, args.method)
        active.update(scenario_id=scenario_id, sample_index=0)
        samples[scenario_id] = np.asarray([evaluate(model.unconditional.map_coordinates(xi)) for xi in coordinates])
        del model, coordinates
        gc.collect()
    summaries = []
    for row in all_rows:
        scenario_id = row["scenario_id"]
        scenario_output = root / scenario_id
        scenario_output.mkdir(exist_ok=True)
        values = samples[scenario_id]
        _write_array(scenario_output / "qoi_samples.npy", values)
        _write_json(scenario_output / "sample_diagnostics.json", records[scenario_id])
        patch = None
        if patches:
            raw_patch = patches[scenario_id].finalize()
            patch = {"generated_field_count": raw_patch["generated_field_count"],
                     "generated_patch_count": raw_patch["generated_patch_count"],
                     "generated_patches_per_field": raw_patch["generated_patches_per_field"],
                     "reference_profile": _reference_profile(profile),
                     "metrics": _reference_labels(raw_patch["metrics"]),
                     "interpretation": context["reference_patch_comparison"]["interpretation"],
                     "sample_filtering": None}
            _write_json(scenario_output / "reference_patch_comparison.json", patch)
        summaries.append({"scenario_id": scenario_id, "config": row["config"],
                          "role": "primary" if row in primary else "KL representation sensitivity",
                          "dimension": row["dimension"], "retained_energy": row["retained_energy"],
                          "temperature_qoi": _moments(values), "budget_sensitivity": _budgets(values),
                          "reference_patch_comparison": patch,
                          "normalization_range_diagnostics": [record["normalization_range_diagnostic"] for record in records[scenario_id]],
                          "permeability_spatial_descriptors": [record["permeability_spatial_descriptors"] for record in records[scenario_id]],
                          "qoi_samples": str(scenario_output / "qoi_samples.npy")})
    summary = {
        "context_id": context_id, "qoi_names": qoi_names,
        "method": args.method, "seed": args.seed, "n_samples_per_scenario": args.n_samples,
        "primary_scenario_count": len(primary), "additional_KL_scenario_count": len(extras),
        "total_evaluated_sample_occurrences": counters["completed"],
        "new_model_predictions_this_invocation": counters["predicted"],
        "cache_hits_this_invocation": counters["cache_hits"],
        "fixed_run_id": config.fixed_run_id, "background_temperature_C": config.background_temperature,
        "scenario_pooling": False, "scenario_weights": None, "sample_filtering": None,
        "uncertainty_status": "explicit model-form assumptions; small pilot budgets; no calibrated sigma/covariance or established QoI convergence",
        "paired_anchor_reuse": {"scenario_id": anchor["scenario_id"],
                               "coordinate_dimension": max(row["dimension"] for row in paired_rows),
                               "interpretation": "anchor uses prefix coordinates of the paired highest-dimensional design; no repeated anchor predictions"},
        "scenarios": summaries,
    }
    _write_json(root / "summary.json", summary)
    print(f"Saved {len(summaries)} separate scenario summaries to {root}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--kl-manifest", required=True)
    parser.add_argument("--rq1-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=5901)
    parser.add_argument("--method", choices=("MC", "RQMC"), default="RQMC")
    parser.add_argument("--reference-arrays", nargs="+", help="Optional full-grid reference envelope; actual training membership is unverified")
    args = parser.parse_args(argv)
    run_pilot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
