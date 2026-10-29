"""Matched GSTools-fitted radial-kernel selection on the original native grid.

Stages prepare/inputs/select/temperatures are resumable and checksum checked.
Covariance fits characterize separate references, not an uncertainty posterior.
All draws are retained in summaries; only fixed picture indices are exported.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtri

from subsurface_uq.io import git_head, runtime_versions, sha256_file
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec, GSToolsReferenceFFTSampler
from subsurface_uq.sampling.reference_coherence import (
    native_short_range_features, reference_ensemble_coherence, _correlation,
    _relative_error, _iid_covariance_relative_rmse, _prefix_convergence, _safe)
from subsurface_uq.sampling.training_compatibility import permeability_field_features
from subsurface_uq.experiments.reference_field_diagnostics import _native_marginal, _probe_indices
from subsurface_uq.experiments.reference_field_permeability import coarsen_reference
from subsurface_uq.visualization.reference_fields import reference_field_agreement, plot_reference_field_comparison


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_hash(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def code_identity():
    import subsurface_uq
    paths = sorted(Path(subsurface_uq.__file__).parent.rglob("*.py"))
    paths += [Path(__file__), Path(__file__).with_name("run_temperature_pilot.py")]
    return {str(p.resolve()): sha256_file(p) for p in paths}


def load_study(root):
    manifest = read(root / "manifest.json")
    context = manifest["context"]
    if sha256(canonical(context).encode()).hexdigest() != manifest["context_id"]:
        raise ValueError("study context checksum changed")
    if context["code"] != code_identity():
        raise ValueError("study implementation changed; use a fresh study directory")
    for source in context["references"].values():
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("reference bytes changed")
    if sha256_file(context["fits"]["path"]) != context["fits"]["sha256"]:
        raise ValueError("fit source changed")
    return manifest


def verify_completed(directory, context_id):
    completed = read(directory / "complete.json")
    if completed["context_id"] != context_id or any(sha256_file(directory / p) != h for p,h in completed["files"].items()):
        raise ValueError("completed scenario output changed")
    return completed


def prepare(root, fits_path, n_inputs, n_temperatures):
    if any(n < 4 or n % 2 for n in (n_inputs, n_temperatures)):
        raise ValueError("even budgets >=4 required for two independent seed batches")
    fits = read(fits_path)
    refs = {}
    cases = []
    for run, item in fits["references"].items():
        path = Path(item["reference_path"]).resolve()
        if sha256_file(path) != item["reference_sha256"]:
            raise ValueError("fit/reference checksum mismatch")
        ref = np.load(path, allow_pickle=False)
        if ref.ndim != 2 or min(ref.shape) < 16 or not np.all(np.isfinite(ref)) or np.any(ref <= 0):
            raise ValueError("positive finite native reference required")
        refs[run] = {"path": str(path), "sha256": item["reference_sha256"], "shape": list(ref.shape)}
        for family in ("gaussian", "matern32_radial", "exponential_radial"):
            for window in (1280., 2560.):
                fit = next(r for r in item["fits"] if r["family"] == family and r["fit_max_lag_m"] == window)
                spec = GSToolsReferenceSpec(family, fit["ell_row_m"], fit["ell_col_m"])
                identity = {"reference_run": run, "reference_sha256": refs[run]["sha256"],
                            "spec": asdict(spec), "fit_max_lag_m": window, "cell_size_m": 5.}
                sid = "matched-" + sha256(canonical(identity).encode()).hexdigest()[:20]
                cases.append({"scenario_id": sid, **identity, "fit": fit})
    context = {"schema_version": 1, "fits": {"path": str(fits_path), "sha256": sha256_file(fits_path)},
        "references": refs, "cases": cases, "cell_size_m": 5., "diagnostic_factor": 8,
        "diagnostic_max_lag_cells": 64, "n_input_draws": n_inputs, "n_temperature_draws": n_temperatures,
        "input_batch_seeds": [194901, 294901], "temperature_batch_seeds": [195901, 295901],
        "code": code_identity(), "source_git_head": git_head(Path(__file__).parents[2]),
        "runtime": runtime_versions(("numpy", "scipy", "gstools", "matplotlib")),
        "sampler": "common native-grid FFT embedding of GSTools-matched radial covariances; no KL truncation",
        "assumptions": {"sigma_R_log10": .05, "center": "median", "rotation_rad": 0., "nugget": 0.,
                        "permeability_convention": "historical-training", "scenario_weights": None},
        "selection_protocol": {
            "within_family_window": "lexicographic: maximum across directions of mean absolute per-field native-gradient deviation, maximum directional variogram relative RMSE, angular spectrum error, radial spectrum error; report all tradeoffs and both seed batches",
            "temperature_finalists": "one input-preferred fit window for EACH kernel family and RUN (nine cases); retain the exponential comparator even if it is not among the two closest-texture cases",
            "final_recommendation": "structural preservation first; assess local and global model responses and numerical stability; no minimum-temperature-variance objective or invented physical pass/fail cutoff"},
        "scenario_pooling": False, "sample_filtering": None,
        "training_membership": "unverified; nominal patch and checkpoint range evidence only"}
    manifest = {"context_id": sha256(canonical(context).encode()).hexdigest(), "context": context}
    if (root / "manifest.json").exists():
        if read(root / "manifest.json") != manifest:
            raise ValueError("existing study has a different context")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("prepare requires an empty output directory")
    write(root / "manifest.json", manifest)
    print(f"Prepared {len(cases)} separate matched configurations", flush=True)


def seeds(context, case, stage):
    count = context["n_input_draws" if stage == "input" else "n_temperature_draws"]
    batch_seeds = context["input_batch_seeds" if stage == "input" else "temperature_batch_seeds"]
    identity = int(case["scenario_id"].split("-")[1][:8], 16)
    return [int(np.random.SeedSequence([batch_seeds[batch], identity, index]).generate_state(1)[0])
            for batch in range(2) for index in range(count // 2)]


def sampler_for(case, reference):
    started = time.perf_counter()
    sampler = GSToolsReferenceFFTSampler(GSToolsReferenceSpec(**case["spec"]), reference.shape)
    sampler.metadata["setup_seconds"] = time.perf_counter() - started
    return sampler


def probe_report(log_samples, reference, indices, spec):
    log_reference = np.log10(reference[tuple(indices.T)])
    delta = (indices[:, None] - indices[None, :]) * 5.
    covariance = spec.sigma_R**2 * spec.correlation(delta[..., 0], delta[..., 1])
    samples = np.asarray(log_samples)
    empirical_cov = np.cov(samples, rowvar=False)
    count = len(samples)
    probabilities = np.array([.01, .05, .25, .5, .75, .95, .99])
    standardized = (samples - log_reference) / spec.sigma_R
    return _safe({"law": "full declared radial Gaussian residual; numerical covariance bound recorded separately",
        "probe_count": len(indices), "field_count": count, "indices_yx": indices,
        "expected_log10_mean": log_reference, "empirical_log10_mean": samples.mean(axis=0),
        "expected_log10_variance": np.diag(covariance), "empirical_log10_variance": np.diag(empirical_cov),
        "expected_correlation": _correlation(covariance), "empirical_correlation": _correlation(empirical_cov),
        "expected_covariance_log10": covariance, "empirical_covariance_log10": empirical_cov,
        "covariance_relative_frobenius_error": _relative_error(empirical_cov, covariance),
        "iid_gaussian_expected_covariance_relative_frobenius_rmse": _iid_covariance_relative_rmse(covariance, count),
        "standardized_quantile_probabilities": probabilities, "standardized_quantiles": np.quantile(standardized, probabilities),
        "standard_normal_quantiles": ndtri(probabilities),
        "batch_probe_mean_errors": [batch.mean(axis=0) - log_reference for batch in np.array_split(samples, 2)],
        "prefix_convergence": _prefix_convergence(samples, log_reference, covariance, iid_mc=True, variance_zero_tolerance=0.),
        "interpretation": "ensemble draws are iid; dependent spatial probes/pixels are not iid replicates; no acceptance test"})


def descriptor_row(case, records, coherence, native_ref):
    result = {"scenario_id": case["scenario_id"], "reference_run": case["reference_run"],
              "family": case["spec"]["family"], "fit_max_lag_m": case["fit_max_lag_m"],
              "ell_row_m": case["spec"]["ell_row_m"], "ell_col_m": case["spec"]["ell_col_m"]}
    fidelity = coherence["reference_fidelity"]
    for name, key in (("row", "gradient_rms_row_log10_per_m"), ("column", "gradient_rms_column_log10_per_m")):
        values = [r["native_short_range"][key] / native_ref[key] for r in records]
        result[f"gradient_{name}_ratio_mean"] = float(np.mean(values))
        result[f"gradient_{name}_mean_absolute_deviation"] = float(np.mean(np.abs(np.asarray(values)-1)))
        result[f"gradient_{name}_batch_means"] = [float(np.mean(v)) for v in np.array_split(values, 2)]
    result["maximum_native_gradient_deviation"] = max(result[f"gradient_{name}_mean_absolute_deviation"] for name in ("row", "column"))
    result["max_variogram_relative_rmse"] = max(v["relative_rmse"] for v in fidelity["directional_variograms"].values())
    result["angular_power_relative_l2"] = fidelity["spectra"]["angular_power_relative_l2"]
    result["radial_power_relative_l2"] = fidelity["spectra"]["radial_power_relative_l2"]
    result["marginal_wasserstein_log10"] = fidelity["marginal"]["wasserstein_distance_log10"]
    result["log10_rmse_mean"] = float(np.mean([r["agreement"]["log10_rmse"] for r in records]))
    result["log10_spatial_correlation_mean"] = float(np.mean([r["agreement"]["log10_spatial_correlation"] for r in records]))
    result["generation_seconds_mean"] = float(np.mean([r["generation_seconds"] for r in records]))
    return result


def input_stage(root, ids=None):
    manifest = load_study(root)
    context = manifest["context"]
    if ids and (len(ids) != len(set(ids)) or set(ids) - {c["scenario_id"] for c in context["cases"]}):
        raise ValueError("select distinct known scenario IDs")
    for case in context["cases"]:
        sid = case["scenario_id"]
        if ids and sid not in ids:
            continue
        directory = root / "inputs" / sid
        if (directory / "complete.json").exists():
            verify_completed(directory, manifest["context_id"])
            print(f"Reused {sid}", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        reference = np.load(context["references"][case["reference_run"]]["path"], allow_pickle=False)
        reference_coarse = coarsen_reference(reference, 8)
        native_ref = native_short_range_features(reference, cell_size_m=5.)
        spec = GSToolsReferenceSpec(**case["spec"])
        probes = _probe_indices(reference.shape, 4, length_scale_m=(spec.ell_row_m, spec.ell_col_m), cell_size_m=5.)
        sampler = None
        records, coarse = [], []
        patch_size = min(1280, *reference.shape)
        patch_starts = sorted({(0, 0), (reference.shape[0] - patch_size, 0),
                               (0, reference.shape[1] - patch_size),
                               (reference.shape[0] - patch_size, reference.shape[1] - patch_size),
                               ((reference.shape[0] - patch_size) // 2, (reference.shape[1] - patch_size) // 2)})
        reference_patches = [permeability_field_features(reference[r:r+patch_size, c:c+patch_size], cell_size_m=5., spatial_stride=8)
                             for r,c in patch_starts]
        for index, seed in enumerate(seeds(context, case, "input")):
            record_path = directory / f"sample-{index:03d}.json"
            coarse_path = directory / f"sample-{index:03d}.coarse.npy"
            if record_path.exists():
                record = read(record_path)
                record_checksum = record.pop("record_sha256")
                if (record["seed"] != seed or record["context_id"] != manifest["context_id"]
                        or sha256(canonical(record).encode()).hexdigest() != record_checksum
                        or sha256_file(coarse_path) != record["coarse_sha256"]):
                    raise ValueError("input sample cache changed")
                current_coarse = np.load(coarse_path, allow_pickle=False)
            else:
                if sampler is None:
                    sampler = sampler_for(case, reference)
                    write(directory / "sampler_accuracy.json", sampler.metadata)
                started = time.perf_counter()
                field = sampler.sample_permeability(reference, seed)
                seconds = time.perf_counter() - started
                residual = np.log10(field.astype(float) / reference)
                increments = {name: float(np.mean(np.diff(residual, axis=axis)**2))
                              for name, axis in (("row", 0), ("column", 1))}
                current_coarse = coarsen_reference(field, 8)
                np.save(coarse_path, current_coarse, allow_pickle=False)
                record = {"context_id": manifest["context_id"], "sample_index": index,
                    "batch": index // (context["n_input_draws"] // 2), "seed": seed,
                    "physical_field_sha256": array_hash(field), "coarse_sha256": sha256_file(coarse_path),
                    "generation_seconds": seconds, "agreement": reference_field_agreement(reference, field),
                    "native_marginal": _native_marginal(field),
                    "native_short_range": native_short_range_features(field, cell_size_m=5.),
                    "residual_native_increment_mean_squares": increments,
                    "log10_probes": np.log10(field[tuple(probes.T)].astype(float)),
                    "patch_features": [permeability_field_features(field[r:r+patch_size,c:c+patch_size], cell_size_m=5., spatial_stride=8)
                                       for r,c in patch_starts]}
                if index == 0:
                    np.save(directory / "permeability_sample_000.npy", field, allow_pickle=False)
                record = _safe(record)
                write(record_path, {**record, "record_sha256": sha256(canonical(record).encode()).hexdigest()})
                del field, residual
            records.append(record)
            coarse.append(current_coarse)
            print(f"{case['reference_run']} {spec.family} w{int(case['fit_max_lag_m'])} input {index+1}/{context['n_input_draws']}", flush=True)
        del sampler
        gc.collect()
        coherence = reference_ensemble_coherence(reference_coarse, np.array(coarse), cell_size_m=40., max_lag_cells=context["diagnostic_max_lag_cells"])
        law = probe_report([r["log10_probes"] for r in records], reference, probes, spec)
        expected_increments = {"row": float(2 * spec.sigma_R**2 * (1-spec.correlation(5.,0.))),
                               "column": float(2 * spec.sigma_R**2 * (1-spec.correlation(0.,5.)))}
        report = {"schema_version": 1, "context_id": manifest["context_id"], "case": case,
            "support": {"native_shape": list(reference.shape), "native_cell_size_m": 5.,
                        "diagnostic_shape": list(reference_coarse.shape), "diagnostic_cell_size_m": 40.,
                        "coarsening": "geometric_block_mean", "native_increment_support": "5 m adjacent cells"},
            "native_reference": {"marginal": _native_marginal(reference), "short_range": native_ref},
            "records": records, "coherence": coherence, "probe_law": law,
            "residual_expected_native_increment_mean_squares": expected_increments,
            "patch_support": {"starts_yx": patch_starts, "size": patch_size, "feature_stride": 8,
                              "reference": reference_patches, "membership": "nominal reference only, training membership unverified"},
            "sample_filtering": None, "scenario_pooling": False}
        summary = descriptor_row(case, records, coherence, native_ref)
        write(directory / "validation.json", report)
        write(directory / "summary.json", summary)
        from subsurface_uq.visualization.reference_validation import _distribution_figure, _correlation_figure, _spectra_figure
        for name, function in (("distribution", _distribution_figure), ("correlation", _correlation_figure), ("spectra", _spectra_figure)):
            function(coherence, directory / f"{name}.png")
        field = np.load(directory / "permeability_sample_000.npy", allow_pickle=False)
        if array_hash(field) != records[0]["physical_field_sha256"]:
            raise ValueError("stored permeability picture array changed")
        plot_reference_field_comparison(reference, field, directory / "permeability_sample_000.png", cell_size_m=5.,
            title=f"{case['reference_run']} | {spec.family} | fitted window {int(case['fit_max_lag_m'])} m | fixed input index 0")
        write(directory / "complete.json", {"context_id": manifest["context_id"],
              "files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name != "complete.json"}})
        print("Completed input diagnostics", sid, flush=True)
        del reference, field, records, coarse, coherence, report
        gc.collect()


def selection_stage(root):
    manifest = load_study(root)
    cases = manifest["context"]["cases"]
    for case in cases:
        verify_completed(root / "inputs" / case["scenario_id"], manifest["context_id"])
    summaries = [read(root / "inputs" / c["scenario_id"] / "summary.json") for c in cases]
    fields = ("maximum_native_gradient_deviation", "max_variogram_relative_rmse",
              "angular_power_relative_l2", "radial_power_relative_l2")
    selections = []
    for run in manifest["context"]["references"]:
        for family in ("gaussian", "matern32_radial", "exponential_radial"):
            candidates = sorted([r for r in summaries if r["reference_run"] == run and r["family"] == family],
                                key=lambda r: tuple(r[k] for k in fields))
            selections.append({"scenario_id": candidates[0]["scenario_id"], "reference_run": run, "family": family,
                               "ranked_fit_windows": candidates, "ranking_priority": fields})
    decision = {"context_id": manifest["context_id"], "selections": selections,
        "interpretation": "one fit window per family; all three families receive model evaluations; input-only ranking precedes model outputs; not model probabilities or an acceptance test"}
    if (root / "selection.json").exists() and read(root / "selection.json") != _safe(decision):
        raise ValueError("existing selection changed")
    write(root / "selection.json", decision)
    write(root / "input_comparison.json", {"context_id": manifest["context_id"], "rows": summaries})
    print(f"Recorded {len(selections)} finalists, including each fitted kernel family", flush=True)


def temperature_stage(root, rq1_path, ids=None):
    # Reuse the verified release25 adapter, observer and descriptor calculations.
    import run_temperature_pilot as pilot
    import torch
    import yaml
    from subsurface_uq.rq1.config import load_rq1_config
    from subsurface_uq.surrogates import Release25Surrogate
    from subsurface_uq.surrogates.release25_runtime import Release25Runtime
    from subsurface_uq.surrogates.bounded_streamlines import configure_release25_streamlines
    from subsurface_uq.sampling.training_compatibility import characterize_training_patch_distribution, TrainingPatchCompatibilityDiagnostics

    manifest = load_study(root)
    context = manifest["context"]
    selection = read(root / "selection.json")
    if selection["context_id"] != manifest["context_id"]:
        raise ValueError("selection context mismatch")
    chosen = {r["scenario_id"] for r in selection["selections"]}
    if ids and (len(ids) != len(set(ids)) or set(ids) - chosen):
        raise ValueError("temperature case IDs must be distinct recorded finalists")
    cases = [c for c in context["cases"] if c["scenario_id"] in chosen and (not ids or c["scenario_id"] in ids)]
    # This runner owns the matched-law manifest; it never calls the KL artifact loader.
    config = load_rq1_config(rq1_path, input_model_override=root / "manifest.json")
    if config.random_k or config.cell_size_m != 5.:
        raise ValueError("real-K 5 m fixed context required")
    info = config.prepared_pki_dir / "info.yaml"
    prepared = yaml.safe_load(info.read_text(encoding="utf-8"))
    shape = tuple(prepared["CellsNumber"])
    if any(tuple(r["shape"]) != shape for r in context["references"].values()) or not np.allclose(prepared["CellsSize"][:2], [5., 5.], rtol=0, atol=0):
        raise ValueError("native field/fixed model grid mismatch")
    tensors = [p for p in (config.prepared_pki_dir / "Inputs" / config.fixed_run_id,
                           config.prepared_pki_dir / "Inputs" / f"{config.fixed_run_id}.pt") if p.is_file()]
    if len(tensors) != 1:
        raise ValueError("one fixed input tensor required")
    files = [info, tensors[0], Path(rq1_path)] + [d / f for d in (config.cnn1_dir, config.cnn2_dir) for f in ("model.pt", "info.yaml", "HPS_options.yaml")]
    model_context = {"study_context_id": manifest["context_id"], "selection_sha256": sha256_file(root / "selection.json"),
        "files": {str(p.resolve()): sha256_file(p) for p in files}, "config": config.to_dict(),
        "upstream_python": {str(p): sha256_file(p) for p in sorted((config.release25_repo / "code").rglob("*.py"))},
        "runtime": runtime_versions(("numpy", "scipy", "torch", "gstools")), "torch_threads": 8,
        "input_law": "actual matched GSTools radial specification from manifest; no KL map",
        "temperature_seed_batches": context["temperature_batch_seeds"], "sample_filtering": None}
    model_id = sha256(canonical(model_context).encode()).hexdigest()
    output = root / "temperatures"
    if (output / "context.json").exists() and read(output / "context.json") != {"context_id": model_id, "context": model_context}:
        raise ValueError("model context changed; use a fresh study directory")
    write(output / "context.json", {"context_id": model_id, "context": model_context})
    torch.set_num_threads(8)
    runtime = Release25Runtime.from_paths(release25_repo=config.release25_repo, cnn1_dir=config.cnn1_dir,
        cnn2_dir=config.cnn2_dir, prepared_pki_dir=config.prepared_pki_dir, run_id=config.fixed_run_id,
        device=config.device, random_k=config.random_k, streamline_method=config.streamline_method)
    configure_release25_streamlines(runtime.adapter, mode=config.streamline_mode, max_nfev=config.streamline_max_nfev,
        diagnostics=config.streamline_diagnostics, slow_streamline_seconds=config.streamline_slow_seconds)
    observer = pilot._CaptureVelocityGrid(runtime.adapter)
    surrogate = Release25Surrogate(observer, runtime.scenario.fixed, device=config.device)
    bounds = {}
    for role, bundle in (("cnn1", runtime.adapter.cnn1), ("cnn3", runtime.adapter.cnn2)):
        channel = [value for key,value in bundle.info["Inputs"].items() if "permeability" in key.lower()]
        if len(channel) != 1:
            raise ValueError("one permeability normalization channel required")
        bounds[role] = (float(channel[0]["min"]), float(channel[0]["max"]))
    reference_fields = [np.load(r["path"], allow_pickle=False) for r in context["references"].values()]
    profile = characterize_training_patch_distribution(np.stack(reference_fields), cell_size_m=5., box_size=1280,
        skip_per_dir=8, feature_stride=8, max_patches=192, field_names=list(context["references"]))
    del reference_fields
    gc.collect()
    cache = output / "predictions"
    cache.mkdir(exist_ok=True)

    def evaluate(field):
        digest = array_hash(field)
        record_path, array_path = cache / f"{digest}.json", cache / f"{digest}.npy"
        if record_path.exists():
            record = read(record_path)
            expected = record.pop("record_sha256")
            if (record["context_id"] != model_id or record["field_sha256"] != digest
                    or sha256(canonical(record).encode()).hexdigest() != expected
                    or sha256_file(array_path) != record["temperature_sha256"]):
                raise ValueError("prediction cache changed")
            return np.load(array_path, allow_pickle=False), record
        started = time.perf_counter()
        temperature = surrogate.predict_temperature(field)
        qoi = [float(temperature[r,c]) for r,c in config.receptors]
        if config.mean_anomaly_roi:
            r0,r1,c0,c1 = config.mean_anomaly_roi
            qoi.append(float(temperature[r0:r1,c0:c1].mean()-config.background_temperature))
        flow = pilot._flow_descriptors(observer.last_result, field.shape)
        h,w = flow["velocity_grid_shape"][-2:]
        y,x = (field.shape[0]-h)//2, (field.shape[1]-w)//2
        np.save(array_path, temperature, allow_pickle=False)
        record = {"context_id": model_id, "field_sha256": digest, "qoi": qoi,
            "temperature_sha256": sha256_file(array_path), "prediction_seconds": time.perf_counter()-started,
            "temperature_descriptors": pilot._temperature_descriptors(temperature, config.background_temperature, .5),
            "flow_descriptors": flow, "normalization": {
                "cnn1": pilot._normalization_range(field, *bounds["cnn1"]),
                "cnn3": pilot._normalization_range(field[y:y+h,x:x+w], *bounds["cnn3"])}}
        write(record_path, {**record, "record_sha256": sha256(canonical(record).encode()).hexdigest()})
        return temperature, record

    nominal = {}
    for run, source in context["references"].items():
        field = np.load(source["path"], allow_pickle=False)
        temperature, record = evaluate(field)
        nominal[run] = (temperature, record)
        directory = output / "nominal" / run
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "temperature.npy", temperature, allow_pickle=False)
        write(directory / "record.json", record)
    for case in cases:
        directory = output / case["scenario_id"]
        directory.mkdir(exist_ok=True)
        if (directory / "complete.json").exists():
            completed = verify_completed(directory, model_id)
            for digest, hashes in completed["prediction_cache"].items():
                if sha256_file(cache / f"{digest}.json") != hashes["record"] or sha256_file(cache / f"{digest}.npy") != hashes["temperature"]:
                    raise ValueError("completed prediction cache changed")
            print("Reused complete temperature screen", case["scenario_id"], flush=True)
            continue
        reference = np.load(context["references"][case["reference_run"]]["path"], allow_pickle=False)
        sampler = sampler_for(case, reference)
        write(directory / "sampler_accuracy.json", sampler.metadata)
        baseline, nominal_record = nominal[case["reference_run"]]
        patch = TrainingPatchCompatibilityDiagnostics(profile)
        records, qois, deltas, mean, m2 = [], [], [], np.zeros_like(baseline,dtype=float), np.zeros_like(baseline,dtype=float)
        for index, seed in enumerate(seeds(context, case, "temperature")):
            field = sampler.sample_permeability(reference, seed)
            temperature, record = evaluate(field)
            patch.update(field[None,...])
            records.append({"sample_index": index, "seed": seed, "batch": index//(context["n_temperature_draws"]//2), **record})
            qois.append(record["qoi"])
            difference = temperature.astype(float)-mean
            mean += difference/(index+1)
            m2 += difference*(temperature-mean)
            delta = temperature-baseline
            deltas.append({"rmse_C": float(np.sqrt(np.mean(delta.astype(float)**2))), "max_absolute_C": float(np.max(np.abs(delta)))})
            if index < 2:
                np.save(directory / f"permeability_{index:03d}.npy", field, allow_pickle=False)
                np.save(directory / f"temperature_{index:03d}.npy", temperature, allow_pickle=False)
                plot_reference_field_comparison(reference, field, directory / f"permeability_{index:03d}.png", cell_size_m=5.,
                    title=f"{case['reference_run']} | {case['spec']['family']} | exact model input {index}")
                pilot._temperature_png(directory / f"temperature_{index:03d}.png", [baseline,temperature],
                    [f"{case['reference_run']} nominal", f"{case['spec']['family']} sample {index}"])
            print(f"{case['reference_run']} {case['spec']['family']} temperature {index+1}/{context['n_temperature_draws']}", flush=True)
        qois = np.array(qois)
        std = np.sqrt(np.maximum(m2/(len(records)-1),0))
        np.save(directory / "qoi_samples.npy", qois, allow_pickle=False)
        np.save(directory / "temperature_mean.npy", mean.astype(np.float32), allow_pickle=False)
        np.save(directory / "temperature_std.npy", std.astype(np.float32), allow_pickle=False)
        pilot._temperature_png(directory / "temperature_mean.png", [baseline,mean], ["Nominal","Ensemble mean"])
        pilot._temperature_png(directory / "temperature_std.png", [std], ["Field-to-field temperature SD"], colorbar_label="Temperature standard deviation [C]")
        pilot._temperature_png(directory / "temperature_mean_delta.png", [mean-baseline], ["Ensemble mean minus nominal"], diverging=True)
        summary = {"context_id": model_id, "case": case, "n_samples": len(records), "records": records,
            "nominal_qoi": nominal_record["qoi"], "qoi_names": [f"temperature_{r}_{c}" for r,c in config.receptors]+(["mean_temperature_anomaly"] if config.mean_anomaly_roi else []),
            "qoi_mean": qois.mean(axis=0), "qoi_std": qois.std(axis=0,ddof=1),
            "qoi_mean_shift": qois.mean(axis=0)-np.array(nominal_record["qoi"]),
            "independent_seed_batches": [{"mean":v.mean(axis=0),"std":v.std(axis=0,ddof=1)} for v in np.array_split(qois,2)],
            "map_deviations": deltas, "temperature_std_spatial_mean_C": float(std.mean()), "temperature_std_spatial_max_C": float(std.max()),
            "temperature_mean_descriptors": pilot._temperature_descriptors(mean,config.background_temperature,.5),
            "nominal_patch_comparison": patch.finalize(), "sample_filtering": None,
            "training_support": "normalization and nominal patch evidence only; actual membership unverified"}
        write(directory / "summary.json", summary)
        write(directory / "complete.json", {"context_id": model_id,
              "prediction_cache": {record["field_sha256"]: {"record": sha256_file(cache / (record["field_sha256"]+".json")),
                                    "temperature": sha256_file(cache / (record["field_sha256"]+".npy"))} for record in records},
              "files": {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file() and p.name != "complete.json"}})
        del sampler, field, patch, reference
        gc.collect()
    all_summaries = [read(output / c["scenario_id"] / "summary.json") for c in context["cases"] if (output / c["scenario_id"] / "summary.json").exists()]
    write(output / "summary.json", {"context_id": model_id, "scenarios": all_summaries,
         "expected_scenario_count": len(chosen), "completed_scenario_count": len(all_summaries),
         "fixed_run_id": config.fixed_run_id, "scenario_pooling": False, "sample_filtering": None})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare","inputs","select","temperatures"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fits")
    parser.add_argument("--rq1-config")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--n-inputs", type=int, default=16)
    parser.add_argument("--n-temperatures", type=int, default=16)
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    if args.stage == "prepare":
        if not args.fits:
            parser.error("prepare requires --fits")
        prepare(root, Path(args.fits).resolve(), args.n_inputs, args.n_temperatures)
    elif args.stage == "inputs":
        input_stage(root, args.case_ids)
    elif args.stage == "select":
        selection_stage(root)
    elif args.stage == "temperatures":
        if not args.rq1_config:
            parser.error("temperatures requires --rq1-config")
        temperature_stage(root, Path(args.rq1_config).resolve(), args.case_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
