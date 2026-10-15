"""Prepare/resume separate kernel scenarios and compare input and frozen-model evidence.

No scores, automatic shortlist, sample rejection, model queries or pooling are
performed here. The temperature screen is a separate, explicitly selected step.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import qmc
import yaml

from ..io import sha256_file
from .reference_scenarios import run_matrix
from .reference_field_diagnostics import validate_reference_fields, _load_criteria


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path)}


def _artifact(row, manifest_path):
    sid = row["scenario_id"]
    if not sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in sid):
        raise ValueError("unsafe scenario ID")
    path = Path(row["input_model"])
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    artifact = _file(path)
    if row.get("input_model_sha256") != artifact["sha256"]:
        raise ValueError(f"manifest/artifact checksum mismatch: {sid}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (payload.get("input_law") not in {"reference-centered-lognormal", "reference-centered-lognormal-candidate"}
            or payload.get("coordinate_distribution") != "iid_standard_normal"
            or payload.get("conditioning")):
        raise ValueError("kernel study requires unconditioned reference-centered Gaussian-coordinate laws")
    if payload.get("scenario") is None or any(payload["scenario"][k] != row[k] for k in ("scenario_id", "config")):
        raise ValueError(f"manifest scenario/config disagrees with artifact: {sid}")
    dimension = payload["coordinate_dimension"]
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1 or dimension != row["dimension"]:
        raise ValueError("manifest/artifact dimension mismatch")
    companion = _file(path.parent / payload["reference_field"]["path"])
    if companion["sha256"] != payload["reference_field"]["sha256"]:
        raise ValueError(f"reference companion checksum mismatch: {sid}")
    return {"scenario_id": sid, "config": row["config"], "input_model": artifact,
            "reference_companion": companion, "dimension": dimension,
            "reference_grid": {"field_shape": payload["field_shape"],
                "domain_size_m": payload["prior"]["domain_size_m"],
                "permeability_convention": payload.get("source_metadata", {}).get("permeability_convention")}}


def _summary(values):
    present = [v for v in values if v is not None]
    return {"min": float(min(present)), "mean": float(np.mean(present)), "max": float(max(present))} if present else None


def _comparison(row, validation=None):
    cfg = row["config"]
    dimension = row["dimension"]
    result = {"scenario_id": row["scenario_id"], "reference_run": cfg["reference_run"],
        "kernel": cfg["covariance_family"], "parameter_set": row.get("parameter_set"),
        "sigma_R_log10": cfg["sigma_R"], "center": cfg["center"],
        "ell_row_m": cfg["ell_y"], "ell_col_m": cfg["ell_x"],
        "n_modes_requested": cfg["n_modes"], "energy_threshold_requested": cfg["energy_threshold"],
        "dimension": dimension, "retained_energy": row["retained_energy"],
        "retained_log10_variance": row.get("retained_log10_variance"),
        "sobol_dimension_supported": dimension <= qmc.Sobol.MAXDIM,
        "sobol_max_dimension": qmc.Sobol.MAXDIM,
        "dense_total_degree_2_terms": (dimension + 1) * (dimension + 2) // 2,
        "input_status": "prepared_not_generated", "temperature_status": "not_evaluated",
        "training_support_status": "unassessed_by_reference_comparison"}
    if validation is None:
        return result
    fidelity = validation["coherence"]["reference_fidelity"]
    probe = validation["probe_law"]
    result.update(input_status="diagnosed_all_fields", n_samples=validation["source"]["n_samples"],
        log10_bias=_summary([v["log10_bias"] for v in validation["native_agreement"]]),
        log10_rmse=_summary([v["log10_rmse"] for v in validation["native_agreement"]]),
        log10_spatial_correlation=_summary([v["log10_spatial_correlation"] for v in validation["native_agreement"]]),
        native_marginals=validation["native_marginals"], native_short_range=validation["native_short_range"],
        marginal_wasserstein_log10=fidelity["marginal"]["wasserstein_distance_log10"],
        directional_variogram_relative_rmse={k: v["relative_rmse"] for k, v in fidelity["directional_variograms"].items()},
        radial_power_relative_l2=fidelity["spectra"]["radial_power_relative_l2"],
        angular_power_relative_l2=fidelity["spectra"]["angular_power_relative_l2"],
        probe_covariance_relative_error=probe["covariance_relative_frobenius_error"],
        probe_correlation_rmse=probe["correlation_rmse"],
        probe_mean_error_log10=probe["log10_mean_error"],
        finite_sample_covariance_reference=probe["mean_sampling_error"],
        support=validation["support"],
        configured_reference_checks=validation["assessment"]["configured_reference_checks"],
        training_support=validation["assessment"]["training_support"])
    return result


def _temperature_attachment(path, artifacts):
    path = Path(path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    record = json.loads((path.parent / "run_context.json").read_text(encoding="utf-8"))
    context = record["context"]
    if (sha256(_canonical(context).encode()).hexdigest() != record["context_id"]
            or summary["context_id"] != record["context_id"] or summary.get("study_mode") != "kernel-screen"):
        raise ValueError("temperature summary/context checksum or study-mode mismatch")
    known = {a["scenario_id"]: a for a in artifacts}
    recorded = {a["scenario_id"]: a for a in context["artifacts"]}
    rows = summary["scenarios"]
    ids = [r["scenario_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids) or set(ids) - known.keys() or set(ids) != recorded.keys():
        raise ValueError("temperature summary must contain distinct scenarios from this study")
    for row in rows:
        sid = row["scenario_id"]
        expected, actual = known[sid], recorded.get(sid)
        if (actual is None or row["config"] != expected["config"] or actual["config"] != expected["config"]
                or any(actual[k]["sha256"] != expected[k]["sha256"] for k in ("input_model", "reference_companion"))):
            raise ValueError(f"temperature scenario provenance disagrees with input study: {sid}")
    # A context checksum binds inputs and options, not the subsequently written
    # summary. Check output claims against that context and durable predictions.
    count, seed, method = context.get("n_samples"), context.get("seed"), context.get("method")
    qoi_names = context.get("qoi_names")
    fixed = context.get("rq1_config_resolved", {}).get("release25", {}).get("fixed_run_id")
    if (context.get("study_mode") != "kernel-screen" or fixed is None
            or isinstance(count, bool) or not isinstance(count, int) or count < 2
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or method not in {"MC", "RQMC"} or (method == "RQMC" and count & (count - 1))
            or not isinstance(qoi_names, list) or not qoi_names
            or any(not isinstance(name, str) or not name for name in qoi_names)
            or len(set(qoi_names)) != len(qoi_names)
            or any(summary.get(key) != value for key, value in (
                ("n_samples_per_scenario", count), ("seed", seed), ("method", method),
                ("qoi_names", qoi_names), ("fixed_run_id", fixed)))):
        raise ValueError("temperature summary sampling/QoI/fixed context disagrees with immutable context")
    root = path.parent
    cache_root = root / "sample_cache" / record["context_id"]
    evidence = {}

    def verified_file(file_path):
        resolved = str(Path(file_path).resolve())
        if resolved in evidence:
            return evidence[resolved]
        entry = _file(file_path)
        evidence[entry["path"]] = entry
        return entry

    def verify_record(diagnostic):
        values = np.asarray(diagnostic.get("qoi", []), dtype=float)
        digest = diagnostic.get("physical_field_sha256", "")
        if (values.shape != (len(qoi_names),) or not np.all(np.isfinite(values))
                or diagnostic.get("context_id") != record["context_id"]
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("temperature diagnostics contain invalid context, field hash or QoIs")
        cache_path = cache_root / f"{digest}.json"
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        checksum = cached.pop("result_sha256", None)
        if (checksum != sha256(_canonical(cached).encode()).hexdigest()
                or cached != {key: value for key, value in diagnostic.items() if key != "sample_index"}):
            raise ValueError("temperature diagnostic disagrees with checksummed prediction cache")
        cached_temperature = cache_root / f"{digest}.temperature.npy"
        if verified_file(cached_temperature)["sha256"] != cached.get("temperature_sha256"):
            raise ValueError("temperature array disagrees with checksummed prediction cache")
        verified_file(cache_path)
        return values

    def moments(values):
        return {"sample_count": len(values), "mean": values.mean(axis=0).tolist(),
                "std_ddof1": values.std(axis=0, ddof=1).tolist(),
                "quantiles_05_50_95": np.quantile(values, [.05, .5, .95], axis=0).tolist()}

    def verify_moments(claim, values):
        expected = moments(values)
        if not isinstance(claim, dict) or claim.get("sample_count") != count:
            raise ValueError("temperature summary moments disagree with saved QoI samples")
        for key in ("mean", "std_ddof1", "quantiles_05_50_95"):
            actual, target = np.asarray(claim.get(key, []), dtype=float), np.asarray(expected[key])
            if (actual.shape != target.shape or not np.all(np.isfinite(actual))
                    or not np.allclose(actual, target, rtol=1.e-12, atol=1.e-12)):
                raise ValueError("temperature summary moments disagree with saved QoI samples")

    nominal = summary.get("nominal_references")
    runs = {row["config"]["reference_run"] for row in rows}
    if not isinstance(nominal, dict) or set(nominal) != runs or summary.get("nominal_reference_count") != len(runs):
        raise ValueError("temperature nominal references disagree with selected scenarios")
    for run in runs:
        baseline = nominal[run]
        reference_checksums = {known[row["scenario_id"]]["reference_companion"]["sha256"]
                               for row in rows if row["config"]["reference_run"] == run}
        if (len(reference_checksums) != 1 or baseline.get("reference_run") != run
                or baseline.get("source", {}).get("sha256") not in reference_checksums):
            raise ValueError("temperature nominal reference source disagrees with input study")
        values = verify_record(baseline["diagnostics"])
        if not np.array_equal(np.asarray(baseline.get("qoi", []), dtype=float), values):
            raise ValueError("temperature nominal QoIs disagree with checksummed prediction cache")
        baseline_path = root / "nominal_references" / run / "temperature.npy"
        if verified_file(baseline_path)["sha256"] != baseline["diagnostics"]["temperature_sha256"]:
            raise ValueError("temperature nominal array disagrees with checksummed prediction cache")
    for row in rows:
        sid = row["scenario_id"]
        if not isinstance(sid, str) or not sid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in sid):
            raise ValueError("unsafe temperature scenario ID")
        qoi_path, diagnostics_path = root / sid / "qoi_samples.npy", root / sid / "sample_diagnostics.json"
        if row.get("dimension") != known[sid]["dimension"] or Path(row.get("qoi_samples", "")).resolve() != qoi_path.resolve():
            raise ValueError("temperature scenario dimension or QoI path disagrees with saved study outputs")
        values = np.load(qoi_path, allow_pickle=False)
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if (values.shape != (count, len(qoi_names)) or not np.all(np.isfinite(values))
                or not isinstance(diagnostics, list) or len(diagnostics) != count
                or [entry.get("sample_index") for entry in diagnostics] != list(range(count))):
            raise ValueError("temperature saved QoI samples/indices disagree with immutable design")
        verified = np.asarray([verify_record(entry) for entry in diagnostics])
        if not np.array_equal(values, verified):
            raise ValueError("temperature saved QoI samples disagree with checksummed prediction cache")
        verify_moments(row.get("temperature_qoi"), values)
        baseline_values = np.asarray(nominal[row["config"]["reference_run"]]["qoi"])
        if not np.array_equal(np.asarray(row.get("nominal_reference_qoi", [])), baseline_values):
            raise ValueError("temperature scenario nominal QoIs disagree with verified reference")
        verify_moments(row.get("qoi_delta_to_nominal"), values - baseline_values)
        for key, record_key in (("temperature_descriptors", "temperature_descriptors"),
                                ("flow_descriptors", "flow_descriptors"),
                                ("normalization_range_diagnostics", "normalization_range_diagnostic")):
            if row.get(key) != [entry[record_key] for entry in diagnostics]:
                raise ValueError("temperature summary descriptors disagree with checksummed prediction cache")
        verified_file(qoi_path)
        verified_file(diagnostics_path)
    return summary, {"summary": _file(path), "context": _file(path.parent / "run_context.json"),
                     "context_id": record["context_id"], "fixed_run_id": summary["fixed_run_id"],
                     "fixed_context": summary["fixed_context"], "training_support_status": summary["training_support_status"],
                     "sampling": {"method": method, "seed": seed, "n_samples_per_scenario": count,
                                  "qoi_names": qoi_names},
                     "design_association": "same serialized scenario laws; input diagnostics and temperature screen can use different samples, seeds, budgets or integration methods; no sample-wise pairing is inferred",
                     "verification": "input identities, immutable output design, checksummed prediction records/temperature arrays, saved QoI samples, moments, nominal deltas and per-prediction descriptors",
                     "verified_output_files": list(evidence.values())}


def _write_comparison(root, report):
    _write(root / "comparison.json", report)
    columns = ["scenario_id", "reference_run", "kernel", "setting", "sigma_R_log10", "ell_row_m", "ell_col_m",
               "dimension", "retained_energy", "sobol_dimension_supported", "input_status", "temperature_status",
               "mean_log10_rmse", "mean_log10_correlation", "marginal_wasserstein_log10", "probe_covariance_relative_error"]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["scenarios"]:
            flat = {k: row.get(k) for k in columns}
            flat.update(setting=(row.get("parameter_set") or {}).get("label"),
                mean_log10_rmse=(row.get("log10_rmse") or {}).get("mean"),
                mean_log10_correlation=(row.get("log10_spatial_correlation") or {}).get("mean"))
            writer.writerow(flat)
    lines = ["# Separate reference/kernel comparisons", "", report["interpretation"], "",
        "All scenarios remain explicit assumptions. No automatic ranking, sample filtering or scenario pooling is performed.", "",
        "| Reference | Kernel | Setting | ID | KL dimension | Retained energy | Input status | Temperature |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |"]
    for row in report["scenarios"]:
        lines.append(f"| {row['reference_run']} | {row['kernel']} | {(row.get('parameter_set') or {}).get('label', '')} | {row['scenario_id']} | {row['dimension']} | {row['retained_energy']:.5f} | {row['input_status']} | {row['temperature_status']} |")
    lines += ["", "Use comparison.csv for scalar metrics and comparison.json for native quantiles, short-range descriptors, directional variograms, spectra and retained-law checks.",
        "", "Per-scenario diagnostics/ID/validation_report.md contains five PNGs. Reference coherence and plausible outputs do not establish LGCNN training membership or predictive accuracy.",
        "", "KL energy thresholds are representation sensitivities. Check downstream temperature-QoI convergence before a full UQ run. Dense degree-2 term counts are computational estimates, not PCE recommendations."]
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    evaluated = [row for row in report["scenarios"] if row["input_status"] == "diagnosed_all_fields"]
    if not evaluated:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"gaussian": "tab:blue", "matern32": "tab:orange", "exponential": "tab:green"}
    for run in sorted({r["reference_run"] for r in evaluated}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for row in evaluated:
            if row["reference_run"] != run:
                continue
            label = f"{row['kernel']} {(row.get('parameter_set') or {}).get('label', row['scenario_id'][-6:])}"
            color = colors.get(row["kernel"], "black")
            directional = [v for v in row["directional_variogram_relative_rmse"].values() if v is not None]
            axes[0].scatter(row["log10_rmse"]["mean"], max(directional) if directional else np.nan, color=color)
            axes[0].annotate(label, (row["log10_rmse"]["mean"], max(directional) if directional else 0.), fontsize=7)
            if row["probe_covariance_relative_error"] is not None:
                axes[1].scatter(row["dimension"], row["probe_covariance_relative_error"], label=label, color=color)
        axes[0].set(xlabel="Mean native log10 RMSE to reference", ylabel="Maximum directional variogram relative RMSE")
        axes[1].set(xlabel="Retained KL dimension", ylabel="Probe covariance error to retained law", xscale="log")
        axes[1].legend(fontsize=6)
        figure.suptitle(f"{run}: separate assumptions; no automatic ranking or ID guarantee")
        figure.savefig(root / f"comparison_{run}.png", dpi=160)
        plt.close(figure)


def run_study(output_dir, *, config_path=None, matrix_manifest=None, scenario_ids=None,
              n_samples=None, seed=None, method="MC", diagnostic_factor=None,
              probe_grid_size=4, max_lag_cells=24, criteria=None, prepare_only=False,
              shortlist_ids=None, temperature_summary=None):
    if (config_path is None) == (matrix_manifest is None):
        raise ValueError("provide exactly one config_path or matrix_manifest")
    if prepare_only and (shortlist_ids or temperature_summary):
        raise ValueError("shortlist/output review requires completed input diagnostics")
    config, source = {}, None
    if config_path is not None:
        source = Path(config_path).resolve()
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("matrix config must be a mapping")
        if config.get("distribution_target") is not None:
            raise ValueError("kernel coherence study does not assess a global marginal target; use reference_scenarios separately")
    external = (json.loads(Path(matrix_manifest).read_text(encoding="utf-8")) if matrix_manifest else {})
    defaults = config if config_path is not None else external
    count = n_samples if n_samples is not None else defaults.get("n_samples", 32)
    design_seed = seed if seed is not None else defaults.get("seed", 4901)
    factor = diagnostic_factor if diagnostic_factor is not None else defaults.get("diagnostic_factor", 8)
    if (isinstance(count, bool) or not isinstance(count, int) or count < 2
            or isinstance(design_seed, bool) or not isinstance(design_seed, int) or design_seed < 0
            or method not in {"MC", "RQMC"} or (method == "RQMC" and count & (count - 1))):
        raise ValueError("require >=2 samples, nonnegative integer seed, MC or power-of-two RQMC design")
    for value in (factor, probe_grid_size, max_lag_cells):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("diagnostic supports/probe size/lag must be positive integers")
    if probe_grid_size > 16:
        raise ValueError("probe_grid_size must not exceed 16")
    criteria = _load_criteria(criteria)
    root = Path(output_dir).resolve()
    context_path = root / "study_context.json"
    if root.exists() and not root.is_dir():
        raise ValueError("study output must be a directory")
    if root.exists() and any(root.iterdir()) and not context_path.is_file():
        raise ValueError("nonempty study directory has no context; choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(matrix_manifest).resolve() if matrix_manifest else root / "scenario_matrix/manifest.json"
    if source is not None and not manifest_path.exists():
        run_matrix(source, manifest_path.parent, prepare_only=True)
    matrix = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source is not None:
        if matrix["config_sha256"] != sha256_file(source):
            raise ValueError("prepared matrix config changed; choose a fresh study directory")
        for row in matrix["scenarios"]:
            reference_path = (source.parent / config["references"][row["config"]["reference_run"]]).resolve()
            if sha256_file(reference_path) != row["source_metadata"]["reference_npy_sha256"]:
                raise ValueError("source reference changed after matrix preparation")
    rows = matrix.get("scenarios", [])
    ids = [r["scenario_id"] for r in rows]
    selected = list(scenario_ids) if scenario_ids else ids
    if not rows or len(set(ids)) != len(ids) or len(set(selected)) != len(selected) or set(selected) - set(ids):
        raise ValueError("select distinct scenario IDs present in a nonempty manifest")
    rows = [r for r in rows if r["scenario_id"] in selected]
    artifacts = [_artifact(row, manifest_path) for row in rows]
    reference_definitions = {}
    for artifact in artifacts:
        run = artifact["config"]["reference_run"]
        definition = _canonical({"sha256": artifact["reference_companion"]["sha256"], **artifact["reference_grid"]})
        if run in reference_definitions and definition != reference_definitions[run]:
            raise ValueError("one reference RUN must identify the same reference array, grid and permeability convention across scenarios")
        reference_definitions[run] = definition
    if method == "RQMC" and any(a["dimension"] > qmc.Sobol.MAXDIM for a in artifacts):
        raise ValueError("scenario dimension exceeds scrambled Sobol limit; use MC or an explicit representation sensitivity")
    if shortlist_ids and (len(set(shortlist_ids)) != len(shortlist_ids) or set(shortlist_ids) - set(selected)):
        raise ValueError("shortlist IDs must be distinct IDs from the selected study")
    package = Path(__file__).parents[1]
    context = {"schema_version": 1, "source": _file(source or manifest_path),
        "matrix_manifest": _file(manifest_path), "artifacts": artifacts,
        "n_samples": count, "seed": design_seed, "method": method,
        "diagnostic_factor": factor, "probe_grid_size": probe_grid_size, "max_lag_cells": max_lag_cells,
        "criteria": criteria,
        "implementation": [_file(path) for path in sorted(package.rglob("*.py"))]}
    context_id = sha256(_canonical(context).encode()).hexdigest()
    record = {"context_id": context_id, "context": context}
    if context_path.exists():
        if json.loads(context_path.read_text(encoding="utf-8")) != record:
            raise ValueError("study context changed; use a new output directory")
    else:
        _write(context_path, record)
    comparisons = []
    for row, artifact in zip(rows, artifacts):
        sid = row["scenario_id"]
        directory = root / "diagnostics" / sid
        validation = None
        if not prepare_only:
            cache_path = directory / "study_record.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached["context_id"] != context_id or any(sha256_file(directory / name) != digest for name, digest in cached["outputs"].items()):
                    raise ValueError(f"cached diagnostics changed: {sid}")
                validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise ValueError(f"incomplete diagnostic directory has no completed cache record: {sid}; use a fresh study directory")
                directory.parent.mkdir(parents=True, exist_ok=True)
                # Publish a scenario only after every report/checksum is complete.
                # An interrupted PNG render cannot poison the resumable final path.
                with TemporaryDirectory(prefix=".diagnostics-", dir=root) as temporary:
                    stage = Path(temporary) / sid
                    validation = validate_reference_fields(artifact["input_model"]["path"], stage,
                        n_samples=count, seed=design_seed, method=method, diagnostic_factor=factor,
                        probe_grid_size=probe_grid_size, max_lag_cells=max_lag_cells, criteria=criteria)
                    names = ["validation.json", *validation["outputs"].values()]
                    _write(stage / "study_record.json", {"context_id": context_id,
                        "outputs": {name: sha256_file(stage / name) for name in names}})
                    if directory.exists():
                        directory.rmdir()  # Empty directories only; fail if contents appeared.
                    stage.replace(directory)
            print(f"Diagnosed {sid} ({count} fields, {row['dimension']} coordinates)", flush=True)
        comparisons.append(_comparison(row, validation))
    report = {"schema_version": 1, "context_id": context_id, "matrix_manifest": str(manifest_path),
        "stage": "prepared_not_generated" if prepare_only else "input_diagnostics_completed",
        "sampling": {"n_samples": count, "seed": design_seed, "method": method,
                     "cross_kernel_pairing": "none; shared seed does not align different KL laws or dimensions"},
        "scenario_pooling": False, "scenario_weights": matrix.get("scenario_weights"),
        "weight_status": "explicit scenario assumptions only; never inferred probabilities",
        "sample_filtering": None, "automatic_selection": None,
        "interpretation": "Compare all evidence separately per RUN. Reference coherence and retained-law consistency do not establish LGCNN training support, geological calibration or downstream QoI convergence.",
        "scenarios": comparisons}
    if temperature_summary is not None:
        output, attachment = _temperature_attachment(temperature_summary, artifacts)
        by_id = {r["scenario_id"]: r for r in output["scenarios"]}
        for row in comparisons:
            if row["scenario_id"] in by_id:
                row.update(temperature_status="frozen_model_evaluated", temperature=by_id[row["scenario_id"]])
        report.update(temperature_attachment=attachment, nominal_references=output["nominal_references"],
                      temperature_qoi_names=output["qoi_names"])
    if shortlist_ids:
        _write(root / "shortlist_manifest.json", {**matrix,
            "scenarios": [{**row, "input_model": artifact["input_model"]["path"]} for row, artifact in zip(rows, artifacts) if row["scenario_id"] in shortlist_ids],
            "scenario_weights": None, "source_scenario_weights": matrix.get("scenario_weights"),
            "selection": {"mode": "explicit_user_ids", "scenario_ids": list(shortlist_ids), "study_context_id": context_id,
                          "interpretation": "human-selected model-form shortlist; no inferred probabilities or automatic ranking"}})
    _write_comparison(root, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", dest="config_path")
    source.add_argument("--matrix-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--scenario-ids", nargs="+")
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=("MC", "RQMC"), default="MC")
    parser.add_argument("--diagnostic-factor", type=int)
    parser.add_argument("--probe-grid-size", type=int, default=4)
    parser.add_argument("--max-lag-cells", type=int, default=24)
    parser.add_argument("--criteria")
    parser.add_argument("--shortlist-ids", nargs="+")
    parser.add_argument("--temperature-summary")
    args = vars(parser.parse_args(argv))
    report = run_study(args.pop("output_dir"), **args)
    print(f"Saved {len(report['scenarios'])} separate scenarios ({report['stage']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
