"""Assess a KL/Gaussian approximation of completed native temperature ensembles.

This consumes retained outputs, not weights or input previews. It does not fit
an input-to-output emulator. Output scores retain sample identities for that
subsequent, separately validated task.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from ..statistics.output_field import (
    FieldKL, assess_fields, fit_kl, gaussian_scores, normality_screen,
    project, score_law, split_indices,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(study, manifest, profiles=None, counts=None):
    """Discover completed cases without relying on dashboard-generated files."""
    cases = []
    seen = set()
    scenarios = manifest["configuration"]["scenarios"]
    if profiles and set(profiles) - set(scenarios):
        raise ValueError("profile filter contains a name absent from this study")
    if counts and set(counts) - set(manifest["configuration"]["measurement_counts"]):
        raise ValueError("observation filter contains a count absent from this study")
    for profile in sorted(scenarios):
        # A manifest is external data: never allow its path fragments to escape.
        if Path(profile).name != profile or profile in (".", "..") or "\\" in profile:
            raise ValueError("unsafe profile path in study manifest")
        if profiles and profile not in profiles:
            continue
        for path in sorted((study / profile).glob("truth-*/**/summary.json")):
            if not path.resolve().is_relative_to(study):
                raise ValueError("case path escapes study directory")
            summary = read_json(path)
            required = ("scenario", "trial", "design", "n_measurements", "n_samples")
            if not all(key in summary for key in required):
                continue
            if summary["scenario"] != profile:
                raise ValueError("summary scenario differs from its profile directory")
            if counts and summary["n_measurements"] not in counts:
                continue
            key = tuple(summary[k] for k in required[:-1])
            if key in seen:
                raise ValueError("duplicate case identity")
            seen.add(key)
            cases.append((path.parent, summary))
    if not cases:
        raise ValueError("no completed native cases match the selected filters")
    # Do not silently omit a requested profile/count pair because its files are missing.
    expected = {(p, n) for p in (profiles or scenarios) for n in (counts or manifest["configuration"]["measurement_counts"])}
    present = {(s["scenario"], s["n_measurements"]) for _, s in cases}
    if expected - present:
        raise ValueError(f"missing selected case combinations: {sorted(expected - present)}")
    cfg = manifest["configuration"]
    if all(k in cfg for k in ("n_truths", "n_designs", "coverage_scenarios")):
        expected_ids = set()
        for profile, count in expected:
            for trial in range(cfg["n_truths"]):
                designs = ["prior"] if count == 0 else [
                    f"{layout}-{i:03d}" for layout in cfg["coverage_scenarios"] for i in range(cfg["n_designs"])]
                expected_ids.update((profile, trial, design, count) for design in designs)
        if seen != expected_ids:
            raise ValueError("selected cases do not match all declared truths and observation designs")
    return cases


def verified_source(case, summary):
    """Verify just the consumed native artifacts, not every unrelated preview."""
    ledger = read_json(case / "result_files.json")
    hashes = {}
    for name in ("summary.json", "prediction_records.json", "temperature_ensemble.npy"):
        path = case / name
        if name not in ledger or not path.resolve().is_relative_to(case.resolve()):
            raise ValueError(f"missing or unsafe completed artifact: {name}")
        hashes[name] = file_hash(path)
        if hashes[name] != ledger[name]:
            raise ValueError(f"source checksum mismatch: {path}")
    fields = np.load(case / "temperature_ensemble.npy", mmap_mode="r", allow_pickle=False)
    records = read_json(case / "prediction_records.json")
    if fields.ndim != 3 or len(fields) != summary["n_samples"] or len(records) != len(fields):
        raise ValueError("native array, summary and sample record counts disagree")
    for i, record in enumerate(records):
        if record.get("sample_index") != i or not isinstance(record.get("input_sha256"), str):
            raise ValueError("sample records must preserve ordered global input identities")
    return fields, records, hashes


def figures(directory, model, metrics):
    """Static scientific figures; numerical arrays remain available at native size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    root = Path(directory)
    areas = np.load(root / "area_samples.npz")
    thresholds = areas["thresholds_C"]
    colors = {"direct": "#252525", "reconstruction": "#2166ac", "gaussian": "#b35806"}
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 4), squeeze=False)
    for i, threshold in enumerate(thresholds):
        ax = axes[0, i]
        for name, color in colors.items():
            values = np.sort(areas[f"{name}_m2"][:, i]) / 1e6
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=name, color=color)
        ax.set(xlabel="Affected area [km²]", ylabel="Cumulative probability", title=f"Warming > {threshold:g} °C", ylim=(0, 1))
        ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(root / "area_distributions.png", dpi=160); plt.close(fig)

    direct = np.load(root / "probability_direct.npy", mmap_mode="r")
    recon = np.load(root / "probability_reconstruction.npy", mmap_mode="r")
    analytic = np.load(root / "probability_gaussian_analytic.npy", mmap_mode="r")
    fig, axes = plt.subplots(len(thresholds), 4, figsize=(16, 4 * len(thresholds)), squeeze=False)
    for i, threshold in enumerate(thresholds):
        # Rendering may be downsampled; metrics and downloadable arrays are native.
        delta = analytic[i] - direct[i]
        bound = max(float(np.max(np.abs(delta))), .01)
        for j, (data, title) in enumerate(((direct[i], "Held-out empirical probability"),
                                         (recon[i], "KL reconstruction probability"),
                                         (analytic[i], "Gaussian probability (analytic)"),
                                         (delta, "Gaussian minus held-out"))):
            im = axes[i, j].imshow(data, origin="upper", cmap="RdBu_r" if j == 3 else "viridis",
                                   vmin=-bound if j == 3 else 0, vmax=bound if j == 3 else 1)
            axes[i, j].set(title=f"{title}\nWarming > {threshold:g} °C", xlabel="Output column", ylabel="Output row")
            fig.colorbar(im, ax=axes[i, j], shrink=.75, label="Probability difference" if j == 3 else "Probability")
    fig.tight_layout(); fig.savefig(root / "spatial_probabilities.png", dpi=140); plt.close(fig)

    latent = np.load(root / "latent_scores.npz")
    scores = latent["scores"][~latent["is_train"]]
    n_panels = min(4, model.rank)
    if n_panels:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), squeeze=False)
        mu, cov, _ = score_law(latent["scores"][latent["is_train"]])
        reference = norm.ppf((np.arange(len(scores)) + .5) / len(scores))
        for j in range(n_panels):
            ax = axes[0, j]
            standardized = (np.sort(scores[:, j]) - mu[j]) / np.sqrt(cov[j, j])
            ax.plot(reference, standardized, ".", color="#2166ac")
            bounds = [min(reference.min(), standardized.min()), max(reference.max(), standardized.max())]
            ax.plot(bounds, bounds, color="gray", linewidth=1)
            ax.set(title=f"Held-out KL coefficient {j + 1}", xlabel="Standard-normal quantile", ylabel="Training-standardized coefficient")
        fig.tight_layout(); fig.savefig(root / "latent_qq.png", dpi=160); plt.close(fig)


def case_report(directory, result):
    root = Path(directory)
    fit, metrics = result["fit"], result["metrics"]
    rows = []
    for item in metrics["thresholds"]:
        for name in ("direct", "reconstruction", "gaussian"):
            stats = item["area_m2"][name]
            rows.append(f"<tr><td>{item['threshold_C']:g}</td><td>{name}</td><td>{stats['n']}</td>"
                        f"<td>{stats['mean']/1e6:.5g}</td><td>{stats['sd']/1e6:.4g}</td>"
                        f"<td>{stats['q05_q50_q95'][0]/1e6:.5g} / {stats['q05_q50_q95'][2]/1e6:.5g}</td></tr>")
    errors = []
    decisions = []
    for item in metrics["thresholds"]:
        for name in ("reconstruction", "gaussian"):
            comparison = item[name]
            errors.append(f"<tr><td>{item['threshold_C']:g}</td><td>{name}</td>"
                          f"<td>{comparison['area_wasserstein_m2']/1e6:.5g}</td>"
                          f"<td>{comparison['probability_mae']:.4g}</td><td>{comparison['probability_max_abs_error']:.4g}</td></tr>")
        for event in item["area_events"]:
            for name, risk in event["risk"].items():
                decisions.append(f"<tr><td>{item['threshold_C']:g}</td><td>{event['budget_m2']/1e6:g}</td>"
                                 f"<td>{name}</td><td>{risk['probability']:.3f}</td>"
                                 f"<td>{risk['ci95'][0]:.3f}–{risk['ci95'][1]:.3f}</td></tr>")
    decision_table = ('<h2>Declared decision events</h2><p>Probability that affected area strictly exceeds the declared budget. '
                      'Exact binomial 95% intervals condition on each ensemble law.</p><table><tr><th>Warming [°C]</th>'
                      '<th>Area budget [km²]</th><th>Ensemble</th><th>Probability</th><th>95% interval</th></tr>'
                      + ''.join(decisions) + '</table>') if decisions else ''
    text = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Output-field assessment</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#222}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}p{{line-height:1.5}}</style>
<h1>{html.escape(result['case']['scenario'])}: {result['case']['n_measurements']} observations</h1>
<p>Training draws: {fit['n_train']}; held-out draws: {result['n_heldout']}. Retained KL rank: {fit['rank']};
training variance retained: {fit['variance_fraction_retained']:.2%}; target met: {fit['variance_target_met']}.
Held-out reconstruction RMSE: {metrics['heldout_temperature_rmse_C']:.4g} °C.</p>
<p><strong>Separate the two approximations.</strong> Direct uses unseen LGCNN fields. Reconstruction projects
those same unseen fields onto the training basis. Gaussian draws replace the training coefficient law by
a multivariate normal law. A successful KL compression does not validate that Gaussian replacement.</p>
<h2>Affected-area distributions</h2><p>All exceeding cells in the native output crop, without a source-connectivity rule.
Strict native-precision T − background &gt; threshold; no sample clipping or filtering.</p>
<table><tr><th>Threshold [°C]</th><th>Ensemble</th><th>Draws</th><th>Mean [km²]</th><th>SD [km²]</th><th>q05 / q95 [km²]</th></tr>{''.join(rows)}</table>
<img src="area_distributions.png" alt="Held-out, reconstructed and Gaussian area distribution functions">
<h2>Spatial risk</h2><p>Native-cell exceedance probabilities. The analytic Gaussian map avoids candidate Monte Carlo noise;
the held-out map remains a finite-sample estimate. View the native arrays for individual cells.</p>
<img src="spatial_probabilities.png" alt="Spatial exceedance probability and discrepancy maps">
<table><tr><th>Warming [°C]</th><th>Approximation</th><th>Area Wasserstein distance [km²]</th><th>Probability MAE</th><th>Maximum probability error</th></tr>{''.join(errors)}</table>
<p>Wasserstein distance is the average absolute difference between area quantiles.
Spatial errors compare against the held-out empirical map; they include its finite-sample error.</p>
{decision_table}
<h2>Gaussianity screen</h2><p>{html.escape(result['normality']['interpretation'])}.
Tests include held-out coefficient projections, pair sums/differences and random projections,
with Holm correction within this case. Non-rejection is not proof of Gaussianity or decision adequacy.</p>
{'<img src="latent_qq.png" alt="Held-out coefficient normal quantile comparisons">' if fit['rank'] else ''}
<h2>Interpretation and reusable data</h2><p>No automatic adequacy claim is made: errors need application-specific tolerances.
Mean Monte Carlo errors and event intervals condition on the fitted law; they exclude uncertainty from estimating
the law. Neither model fit nor these ensembles validate the frozen LGCNN against physical simulations.</p>
<p><a href="assessment.json">Complete diagnostics and provenance</a> · <a href="area_samples.npz">Area draws</a> ·
<a href="latent_scores.npz">Paired sample-indexed output coefficients</a> · <a href="sample_links.json">Input fingerprints</a> ·
<a href="gaussian_law.npz">Gaussian coefficient law</a> · <a href="model/model.json">KL model definition</a></p>
</html>"""
    (root / "index.html").write_text(text, encoding="utf-8")


def overview(output, report):
    """A compact cross-case comparison, without pooling unrelated laws."""
    import matplotlib.pyplot as plt
    cases = report["cases"]
    labels = [f"{c['scenario']} · n={c['n_measurements']} · truth={c['trial']} · {c['design']}" for c in cases]
    thresholds = [x["threshold_C"] for x in cases[0]["thresholds"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(3, .48 * len(cases) + 1.5)), squeeze=False)
    for j, (key, title) in enumerate((("area_wasserstein_m2", "Gaussian area Wasserstein distance [km²]"),
                                     ("probability_mae", "Gaussian spatial probability MAE"))):
        values = np.array([[t["gaussian"][key] for t in c["thresholds"]] for c in cases])
        if j == 0:
            values /= 1e6
        ax = axes[0, j]
        im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0)
        ax.set_xticks(range(len(thresholds)), [f"> {t:g} °C" for t in thresholds])
        ax.set_yticks(range(len(cases)), labels if j == 0 else [""] * len(cases))
        ax.set_title(title)
        for r in range(len(cases)):
            for t in range(len(thresholds)):
                ax.text(t, r, f"{values[r,t]:.3g}", ha="center", va="center",
                        color="white" if values[r, t] > .65 * max(values.max(), 1e-15) else "black")
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=150); plt.close(fig)
    links = "".join(f'<li><a href="{html.escape(c["path"], quote=True)}/index.html">{html.escape(c["path"])}</a>'
                    f' — KL rank {c["rank"]}, retained variance {c["variance_fraction_retained"]:.2%}, '
                    f'held-out RMSE {c["heldout_temperature_rmse_C"]:.4g} °C</li>' for c in cases)
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Output-field assessments</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px}img{max-width:100%}li{margin:12px 0}</style>'
        '<h1>Output-field assessments</h1><p>Cases are assessed separately; no kernel probability weights are assumed. '
        'These discrepancies combine KL compression, Gaussian-law approximation and finite held-out sampling error. '
        'Use each case report to separate the first two; no universal adequacy cutoff is imposed.</p>'
        '<img src="comparison.png" alt="Case-by-threshold heatmaps of Gaussian approximation errors"><ul>'
        + links + '</ul><p><a href="manifest.json">All numerical comparison summaries</a></p>', encoding="utf-8")


def run(study_dir, output_dir, *, profiles=None, counts=None, train_fraction=.7,
        max_rank=32, variance_fraction=.99, gaussian_draws=1000, seed=20261110,
        chunk_cells=4096, threads=4, area_events=(), progress=print):
    study, output = Path(study_dir).resolve(), Path(output_dir).resolve()
    if output == study or output.is_relative_to(study) or study.is_relative_to(output):
        raise ValueError("output must be outside the source study, with neither directory containing the other")
    if output.exists():
        raise ValueError("choose a fresh output directory; existing results are never overwritten")
    if threads < 1 or gaussian_draws < 8 or chunk_cells < 1 or max_rank < 1 or not 0 < variance_fraction <= 1:
        raise ValueError("positive threads/chunk/rank, at least 8 Gaussian draws, and variance fraction in (0,1] required")
    manifest = read_json(study / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("source study must be complete")
    cases = inventory(study, manifest, profiles, counts)
    background = float(manifest["model"]["release25"]["background_temperature"])
    cell_size = float(manifest["configuration"]["cell_size_m"])
    thresholds = [float(x) for x in manifest["configuration"]["risk_thresholds_C"]]
    if (not np.isfinite([background, cell_size, *thresholds]).all() or cell_size <= 0 or not thresholds
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("source must declare finite background, positive cell size and unique risk thresholds")
    if any(tau not in thresholds or not np.isfinite(budget) or budget < 0 for tau, budget in area_events):
        raise ValueError("area events require an existing threshold and a finite nonnegative area budget in m2")
    # Validate splits before creating any result directory.
    for _, summary in cases:
        split_indices(summary["n_samples"], train_fraction, seed)
    output.mkdir(parents=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    configuration = {"train_fraction": train_fraction, "max_rank": max_rank, "variance_fraction": variance_fraction,
                     "gaussian_draws": gaussian_draws, "seed": seed, "chunk_cells": chunk_cells, "threads": threads,
                     "area_events": list(area_events), "profiles": profiles, "counts": counts}
    report = {"schema_version": 1, "status": "running", "source_study_id": manifest.get("study_id"),
              "reference_id": manifest.get("reference_id"), "source_manifest_sha256": file_hash(study / "manifest.json"),
              "configuration": configuration, "code_commit": commit,
              "analysis_source_sha256": {Path(__file__).name: file_hash(__file__),
                                         "output_field.py": file_hash(Path(__file__).parents[1] / "statistics/output_field.py")},
              "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "purpose": "output-only KL compression and Gaussian-field adequacy assessment; no learned input-output mapping",
              "case_pooling": "none; identical global draw split per equal-sized case preserves paired study indices",
              "cases": []}
    write_json(output / "manifest.json", report)
    try:
        with threadpool_limits(limits=threads):
            for number, (case, summary) in enumerate(cases):
                relative = case.relative_to(study)
                progress(f"[{number + 1}/{len(cases)}] Verifying {relative}")
                fields, records, hashes = verified_source(case, summary)
                train, test = split_indices(len(fields), train_fraction, seed)
                target = output / relative
                target.mkdir(parents=True)
                model = fit_kl(fields, train, target / "model", max_rank=max_rank,
                               variance_fraction=variance_fraction, chunk_cells=chunk_cells, progress=progress)
                scores = project(fields, model, chunk_cells)
                candidate = gaussian_scores(scores[train], gaussian_draws, seed + 1)
                is_train = np.zeros(len(fields), dtype=bool); is_train[train] = True
                np.savez(target / "latent_scores.npz", scores=scores, is_train=is_train,
                         sample_index=np.arange(len(fields)), source_input_sha256=np.array([r["input_sha256"] for r in records]))
                law_mean, law_covariance, _ = score_law(scores[train])
                np.savez(target / "gaussian_law.npz", mean=law_mean, covariance=law_covariance,
                         candidate_scores=candidate, candidate_seed=seed + 1)
                write_json(target / "sample_links.json", {"source_study_id": manifest.get("study_id"),
                           "reference_id": manifest.get("reference_id"), "case_relative_path": relative.as_posix(),
                           "case": {k: summary[k] for k in ("scenario", "trial", "design", "n_measurements")},
                           "records": [{k: r[k] for k in ("sample_index", "seed", "input_sha256", "temperature_sha256") if k in r} for r in records],
                           "mapping_note": "use exact sample-linked input coordinates; input previews/histograms do not recover the native input law"})
                metrics = assess_fields(fields, model, scores, train, test, candidate, target,
                                        background=background, thresholds=thresholds, cell_size_m=cell_size,
                                        chunk_cells=chunk_cells, area_events=area_events, progress=progress)
                normality = normality_screen(scores[train], scores[test], seed=seed + 2)
                result = {"case": {k: summary[k] for k in ("scenario", "trial", "design", "n_measurements", "n_samples")},
                          "source_sha256": hashes, "source_relative_path": relative.as_posix(),
                          "fit": model.info, "n_heldout": len(test), "background_temperature_C": background,
                          "cell_size_m": cell_size, "metrics": metrics, "normality": normality}
                write_json(target / "assessment.json", result)
                figures(target, model, metrics)
                case_report(target, result)
                report["cases"].append({"path": relative.as_posix(), **result["case"],
                                       "rank": model.rank, "variance_fraction_retained": model.info["variance_fraction_retained"],
                                       "variance_target_met": model.info["variance_target_met"],
                                       "heldout_temperature_rmse_C": metrics["heldout_temperature_rmse_C"],
                                       "gaussianity_screen": normality["interpretation"], "thresholds": metrics["thresholds"]})
                write_json(output / "manifest.json", report)
                del fields, model
        overview(output, report)
        report["status"] = "complete"
        write_json(output / "manifest.json", report)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "manifest.json", report)
        raise
    progress(f"Complete: {output / 'index.html'}")
    return report


def area_event(text):
    try:
        tau, budget = map(float, text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use THRESHOLD_C:AREA_M2, e.g. 0.1:9000000") from error
    return tau, budget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, help="Completed RUN directory containing manifest.json")
    parser.add_argument("--output-dir", required=True, help="Fresh directory outside the source study")
    parser.add_argument("--profile", action="append", dest="profiles", help="Exact scenario name; repeat to select multiple; default all")
    parser.add_argument("--n-measurements", type=int, action="append", dest="counts", help="Repeat to select multiple; default all, including prior")
    parser.add_argument("--train-fraction", type=float, default=.7)
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--variance-fraction", type=float, default=.99)
    parser.add_argument("--gaussian-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--chunk-cells", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--area-event", type=area_event, action="append", dest="area_events", default=[],
                        help="Optional declared decision event THRESHOLD_C:AREA_M2; strict area > budget")
    args = parser.parse_args(argv)
    run(args.study_dir, args.output_dir, profiles=args.profiles, counts=args.counts,
        train_fraction=args.train_fraction, max_rank=args.max_rank, variance_fraction=args.variance_fraction,
        gaussian_draws=args.gaussian_draws, seed=args.seed, chunk_cells=args.chunk_cells,
        threads=args.threads, area_events=args.area_events, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
