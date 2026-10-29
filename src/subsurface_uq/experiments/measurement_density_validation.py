"""Observation-only model selection and an untouched spatial reconstruction test."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.gstools_reference import GSToolsReferenceSpec
from ..sampling.measurement_density import coverage_locations
from ..validation.reconstruction import spatial_test_partition, select_covariance, predict_points, withheld_diagnostics
from .reference_conditioning import _identity, _positive_integer, _seed, _write


def validate_settings(cfg):
    v = cfg["validation"]
    if not isinstance(v, dict) or set(v) != {"block_cells", "test_fraction", "buffer_cells", "selection_folds", "n_designs", "rules"}:
        raise ValueError("validation requires block_cells, test_fraction, buffer_cells, selection_folds, n_designs and rules")
    for key in ("block_cells", "selection_folds", "n_designs"):
        _positive_integer(v[key], key)
    _positive_integer(v["buffer_cells"], "buffer_cells", zero=True)
    if v["selection_folds"] < 2 or isinstance(v["test_fraction"], bool) or not 0 < v["test_fraction"] < 1:
        raise ValueError("at least two selection folds and test fraction in (0,1) required")
    rules = v["rules"]
    keys = {"minimum_test_blocks", "coverage90_min", "coverage90_max", "standardized_rmse_min",
            "standardized_rmse_max", "absolute_standardized_bias_max"}
    if not isinstance(rules, dict) or set(rules) != keys:
        raise ValueError(f"declare exactly these diagnostic rules: {sorted(keys)}")
    _positive_integer(rules["minimum_test_blocks"], "minimum_test_blocks")
    for k in keys-{"minimum_test_blocks"}:
        if isinstance(rules[k], bool) or not isinstance(rules[k], (int, float)) or not np.isfinite(rules[k]):
            raise ValueError("finite numerical diagnostic rules required")
    if not (0 <= rules["coverage90_min"] < rules["coverage90_max"] <= 1
            and 0 <= rules["standardized_rmse_min"] < rules["standardized_rmse_max"]
            and rules["absolute_standardized_bias_max"] > 0):
        raise ValueError("invalid diagnostic rule intervals")
    raw = cfg["covariance_candidates"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("declare an external, reference-independent covariance candidate list")
    try:
        specs = [GSToolsReferenceSpec(**r) for r in raw]
    except TypeError as exc:
        raise ValueError("each covariance candidate must declare known covariance parameters") from exc
    if any(isinstance(x, bool) for s in specs for x in (s.ell_row_m, s.ell_col_m, s.sigma_R, s.angle_deg)):
        raise ValueError("covariance lengths, amplitude and rotation must be numerical, not boolean")
    if any(s.center != "median" for s in specs) or len(set(specs)) != len(specs):
        raise ValueError("distinct median-transformed Gaussian log-field candidates required")


def write_validation_report(output, manifest, cases):
    rows = []
    refs = list(manifest["configuration"]["references"])
    fig, axes = plt.subplots(len(refs), 3, figsize=(15, 4*len(refs)), squeeze=False, layout="constrained")
    for name in refs:
        ax = axes[refs.index(name)]
        selected = [c for c in cases if c["reference"] == name]
        for key in sorted({(c["coverage"], c["design_index"]) for c in selected}):
            entries = sorted([c for c in selected if (c["coverage"], c["design_index"]) == key], key=lambda c:c["n_measurements"])
            x = [c["n_measurements"] for c in entries]
            label = f"{key[0]}, design {key[1]+1}"
            color = ax[0].plot(x, [c["diagnostics"]["aggregate"]["rmse_log10"] for c in entries], "o-", label=label)[0].get_color()
            ax[0].plot(x, [c["diagnostics"]["aggregate"]["rms_predictive_sd_log10"] for c in entries], "--", color=color)
            ax[1].plot(x, [c["diagnostics"]["equal_block_coverage90"] for c in entries], "o-", color=color)
            ax[2].plot(x, [c["diagnostics"]["aggregate"]["standardized_rmse"] for c in entries], "o-", color=color)
        rules = manifest["configuration"]["validation"]["rules"]
        ax[1].axhline(.9, color="black", ls=":", label="nominal 90%")
        ax[1].axhspan(rules["coverage90_min"], rules["coverage90_max"], color="grey", alpha=.1)
        ax[1].set_ylim(0, 1.02)
        ax[2].axhline(1., color="black", ls=":")
        ax[2].axhspan(rules["standardized_rmse_min"], rules["standardized_rmse_max"], color="grey", alpha=.1)
        for a, title, unit in zip(ax, ["Withheld error (solid) and predictive SD (dashed)", "Withheld 90% coverage", "Standardized reconstruction error"],
                                  ["log10 permeability", "equal-block fraction", "RMS(error / predictive SD)"]):
            a.set(title=f"{name}: {title}", xlabel="Known exact readings", ylabel=unit, xscale="log")
            a.grid(alpha=.2)
        ax[0].legend(fontsize=7)
        ax[1].legend(fontsize=8)
    for c in cases:
        d = c["diagnostics"]
        rows.append({"reference": c["reference"], "coverage": c["coverage"], "design": c["design_index"],
            "n_measurements": c["n_measurements"], **c["covariance"],
            "rmse_log10": d["aggregate"]["rmse_log10"], "rms_predictive_sd_log10": d["aggregate"]["rms_predictive_sd_log10"],
            "coverage90": d["equal_block_coverage90"], "coverage95": d["aggregate"]["reliability"]["0.95"],
            "standardized_rmse": d["aggregate"]["standardized_rmse"], "standardized_bias": d["aggregate"]["standardized_bias"],
            "crps_log10": d["aggregate"]["mean_crps_log10"], "adequate": d["adequate_for_exploratory_propagation"],
            "failed_checks": ",".join(k for k, value in d["checks"].items() if not value)})
    fig.suptitle("Spatial reconstruction test: covariance chosen from revealed readings only\nShared test blocks excluded from observations; parameters frozen at the smallest reading count")
    fig.savefig(output/"reconstruction_validation.png", dpi=160)
    plt.close(fig)
    with (output/"reconstruction_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    passed = sum(r["adequate"] for r in rows)
    assessment = {"n_cases": len(rows), "cases_passing_declared_checks": passed,
        "adequate_for_exploratory_propagation": passed == len(rows),
        "interpretation": "all configured cases must pass before propagation; test scores never tune the selected covariance; no calibrated geological probability claim",
        "rows": rows, "cases": cases}
    _write(output/"reconstruction_validation.json", assessment)
    lines = ["# Withheld reconstruction uncertainty assessment", "",
        f"**{passed} of {len(rows)} cases pass the predeclared diagnostic checks.**", "",
        "The candidate covariance family, length and amplitude are selected using spatial-block cross-validation of only the smallest exact-reading prefix. They are frozen across larger counts. Test values and future readings do not enter selection.", "",
        "Complete randomly selected spatial blocks and their declared observation buffer are reserved before generating any layout. Scores use point samples on the declared evaluation grid, not block averages. Pixels and adjacent blocks are spatially dependent; no binomial significance test or independent-pixel confidence interval is used.", "",
        "The configured checks are exploratory adequacy tolerances, not physical safety limits or a calibration certificate. Covariance-parameter uncertainty is not included in conditional variance. Passing cases do not justify a pooled geological probability or an optimal measurement count.", "",
        "| Reference | Coverage | Design | Readings | RMSE | Predictive RMS SD | 90% coverage | Standardized RMSE | Failed checks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        z = "undefined" if r["standardized_rmse"] is None else f"{r['standardized_rmse']:.3f}"
        fmt = lambda value: "unavailable" if value is None else f"{value:.5f}"
        lines.append(f"| {r['reference']} | {r['coverage']} | {r['design']+1} | {r['n_measurements']} | {fmt(r['rmse_log10'])} | {fmt(r['rms_predictive_sd_log10'])} | {fmt(r['coverage90'])} | {z} | {r['failed_checks'] or 'none'} |")
    lines += ["", "![Withheld error and uncertainty](reconstruction_validation.png)", "",
        "Each design retains its observation sequence, selection scores, test predictions, per-block coverage, reliability levels and uncertainty-bin diagnostics. Reusing these test results to modify candidates requires a fresh independent assessment."]
    (output/"reconstruction_validation.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return assessment


def screen_validation(source, cfg, refs, output, *, source_hashes):
    if output.exists():
        raise ValueError("validation requires a fresh output directory")
    output.mkdir(parents=True)
    settings = cfg["validation"]
    manifest = {"status": "validating", "configuration": cfg, "configuration_path": str(source),
        "source_sha256": source_hashes, "base_git_commit": git_head(Path(__file__).parents[3]),
        "runtime": runtime_versions(("numpy", "scipy", "gstools", "torch")),
        "mode": "observations_only_spatial_validation", "designs": [],
        "information_state": "unknown constant mean estimated from the current exact-reading prefix",
        "covariance_information": "declared external candidates selected by spatial CV of the smallest prefix only",
        "reference_pooling": None, "scenario_weights": None, "sample_filtering": None,
        "measurement_error_variance": 0., "test_values_used_for_selection": False,
        "pairing": "shared prior draws across counts within a fixed design and selected covariance; layouts remain separate"}
    manifest["study_id"] = _identity({"configuration": cfg, "sources": source_hashes}, "density-validation-")
    _write(output/"manifest.json", manifest)
    candidates = [GSToolsReferenceSpec(**s) for s in cfg["covariance_candidates"]]
    cases = []
    for name, record in refs.items():
        root = output/name; root.mkdir()
        truth = np.load(record["path"], mmap_mode="r")
        test, excluded, blocks, metadata = spatial_test_partition(record["shape"], block_cells=settings["block_cells"],
            fraction=settings["test_fraction"], buffer_cells=settings["buffer_cells"], seed=_seed(cfg["seed"], name, "test-blocks"))
        np.savez_compressed(root/"test_partition.npz", test_mask=test, excluded_observation_mask=excluded,
                            block_ids=blocks)
        _write(root/"test_partition.json", metadata)
        stride = cfg["screen_stride_cells"]
        queries = np.column_stack(np.where(test[::stride, ::stride]))*stride
        if not len(queries) or len(np.unique(blocks[tuple(queries.T)])) < settings["rules"]["minimum_test_blocks"]:
            raise ValueError("evaluation grid does not cover enough withheld blocks")
        # This vector is used only below, after each selected law has been saved.
        targets = np.log10(truth[tuple(queries.T)].astype(float))
        for coverage in cfg["coverage_scenarios"]:
            for design in range(settings["n_designs"]):
                directory = root/coverage/f"design-{design:03d}"; directory.mkdir(parents=True)
                indices, design_meta = coverage_locations(record["shape"], max(cfg["measurement_counts"]),
                    stride=cfg["candidate_stride_cells"], seed=_seed(cfg["seed"], name, coverage, design, "locations"),
                    kind=coverage, cluster_scale=cfg["cluster_scale_fraction"], excluded_mask=excluded)
                readings = np.log10(truth[tuple(indices.T)].astype(float))
                np.save(directory/"measurement_indices.npy", indices)
                _write(directory/"design.json", design_meta)
                np.savetxt(directory/"measurements.csv", np.column_stack((np.arange(1,len(indices)+1), indices, readings)),
                    delimiter=",", header="order,native_row,native_column,log10_K", comments="", fmt=["%d","%d","%d","%.17g"])
                n0 = min(cfg["measurement_counts"])
                selection = select_covariance(record["shape"], indices[:n0], readings[:n0], candidates,
                    cell_size_m=cfg["cell_size_m"], block_cells=settings["block_cells"], n_folds=settings["selection_folds"],
                    seed=_seed(cfg["seed"], name, coverage, design, "selection-folds"))
                _write(directory/"covariance_selection.json", selection)
                spec = GSToolsReferenceSpec(**selection["selected"]["covariance"])
                identity = {"reference": name, "coverage": coverage, "design_index": design,
                    "directory": str(directory.relative_to(output)), "covariance": asdict(spec)}
                manifest["designs"].append(identity)
                for n in cfg["measurement_counts"]:
                    print(f"withheld reconstruction: {name}/{coverage}/design-{design:03d}/n={n}", flush=True)
                    target = directory/f"n-{n:04d}"; target.mkdir()
                    try:
                        mean, std, condition = predict_points(record["shape"], spec, indices[:n], readings[:n], queries,
                            cell_size_m=cfg["cell_size_m"], chunk_size=cfg["chunk_size"])
                        diagnostics = withheld_diagnostics(targets, mean, std, blocks[tuple(queries.T)], rules=settings["rules"])
                        np.savez(target/"withheld_predictions.npz", indices=queries, truth_log10=targets,
                                 mean_log10=mean, std_log10=std, block_ids=blocks[tuple(queries.T)])
                    except (ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
                        # A selected model that fails at a larger count is not
                        # replaced using test performance or later readings.
                        diagnostics = {"aggregate": {"rmse_log10": None, "rms_predictive_sd_log10": None,
                            "standardized_rmse": None, "standardized_bias": None, "mean_crps_log10": None,
                            "reliability": {"0.95": None}}, "equal_block_coverage90": None,
                            "checks": {"numerically_available": False}, "adequate_for_exploratory_propagation": False,
                            "reason": str(exc)}
                        condition = None
                    case = {**identity, "n_measurements": n, "diagnostics": diagnostics, "condition_number": condition}
                    _write(target/"withheld_diagnostics.json", case)
                    cases.append(case)
    assessment = write_validation_report(output, manifest, cases)
    for path, digest in source_hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"frozen validation source changed: {path}")
    manifest.update(status="validated" if assessment["adequate_for_exploratory_propagation"] else "validation_failed",
        validation_status=assessment["interpretation"],
        adequate_for_exploratory_propagation=assessment["adequate_for_exploratory_propagation"],
        validation_artifact_sha256={str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*")
                                    if p.is_file() and p.name != "manifest.json"})
    _write(output/"manifest.json", manifest)
    return manifest
