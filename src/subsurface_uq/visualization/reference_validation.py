"""Present precomputed permeability validation diagnostics without acceptance rules."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_DIRECTIONS = {
    "row": "Array-row direction", "column": "Array-column direction",
    "diagonal_positive": "Positive diagonal", "diagonal_negative": "Negative diagonal",
}


def _save(figure, path, *, subtitle):
    figure.supxlabel(subtitle, fontsize=9)
    try:
        figure.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(figure)


def _envelope(axis, x, curve, *, reference=True):
    """Draw the stored empirical field envelope; do not infer uncertainty bands."""
    values = curve.get("generated_per_field", [])
    for row in values:
        axis.plot(x, row, color="#d97706", alpha=.10, linewidth=.7)
    envelope = curve["generated_envelope"]
    probabilities = np.asarray(envelope["probabilities"])
    quantiles = np.asarray(envelope["values"], dtype=float)
    axis.fill_between(x, quantiles[0], quantiles[-1], color="#e6a34c", alpha=.30,
                      label=f"Generated field P{100 * probabilities[0]:g}–P{100 * probabilities[-1]:g}")
    median_index = int(np.argmin(np.abs(probabilities - .5)))
    axis.plot(x, quantiles[median_index], color="#c56b0c", linewidth=1.5, label="Generated field median")
    if reference and curve.get("reference") is not None:
        axis.plot(x, curve["reference"], color="#222222", linewidth=1.8, label="Fixed reference")
    axis.grid(alpha=.18)


def _distribution_figure(coherence, destination):
    data = coherence["distribution"]
    probabilities = np.asarray(data["probabilities"])
    reference = np.asarray(data["reference_quantiles_log10"])
    generated = np.asarray(data["generated_quantiles_log10"])
    per_field = np.asarray(data["generated_field_quantiles_log10"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    edges = np.asarray(data["bin_edges_log10"])
    centers = (edges[:-1] + edges[1:]) / 2
    density_envelope = np.asarray(data["generated_density_envelope"]["values"], dtype=float)
    axes[0, 0].fill_between(centers, density_envelope[0], density_envelope[-1],
                            color="#e6a34c", alpha=.3, label="Generated field P05–P95")
    axes[0, 0].stairs(data["reference_density"], edges, color="#222222", label="Fixed reference")
    axes[0, 0].stairs(data["generated_density"], edges, color="#c56b0c", label="Generated pixel pool")
    axes[0, 0].set(title="Spatial log-permeability histogram", xlabel="log10 K", ylabel="Density")
    axes[0, 0].legend(fontsize=8)
    for row in per_field:
        axes[0, 1].plot(row, probabilities, color="#d97706", alpha=.13, linewidth=.7)
        axes[1, 0].plot(reference, row, color="#d97706", alpha=.13, linewidth=.7)
    axes[0, 1].plot(data["cdf_log10"], data["reference_cdf"], color="#222222", label="Fixed reference")
    axes[0, 1].plot(data["cdf_log10"], data["generated_cdf"], color="#c56b0c", label="Generated pixel pool")
    axes[0, 1].set(title="Spatial CDF at histogram edges", xlabel="log10 K", ylabel="Spatial probability")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(reference, reference, color="#222222", label="Exact marginal agreement")
    axes[1, 0].plot(reference, generated, color="#c56b0c", label="Generated pixel pool")
    axes[1, 0].set(title="Reference–generated quantile comparison", xlabel="Reference log10 K quantiles",
                   ylabel="Generated log10 K quantiles")
    axes[1, 0].legend(fontsize=8)
    for index, row in enumerate(data["log_ratio_quantiles"]):
        axes[1, 1].plot(probabilities, row, linewidth=1, alpha=.6,
                        label="One generated field" if index == 0 else None)
    axes[1, 1].axhline(0, color="#222222", linewidth=1, label="K = Kref")
    axes[1, 1].set(title="Within-field multiplicative changes", xlabel="Spatial quantile probability",
                   ylabel="log10(K / Kref)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.18)
    figure.suptitle(f"Distributional coherence with the fixed reference · {coherence['generated_field_count']} fields", fontsize=14)
    _save(figure, destination, subtitle=(
        "Histograms mix heterogeneous locations; they are not one pointwise lognormal law. "
        "Thin lines show individual generated fields, not confidence intervals."))


def _correlation_figure(coherence, destination):
    figure, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    groups = (("variograms", "Total-field semivariance", "Semivariance of log10 K"),
              ("autocorrelation", "Total-field spatial correlation", "Spatial correlation"),
              ("residual_variograms", "Log-ratio semivariance", "Semivariance of log10(K / Kref)"))
    for row, (direction, label) in enumerate(_DIRECTIONS.items()):
        for column, (group, title, ylabel) in enumerate(groups):
            axis = axes[row, column]
            curve = coherence[group][direction]
            _envelope(axis, curve["distance_m"], curve)
            axis.set(title=f"{label}: {title}", xlabel="Lag distance (m)", ylabel=ylabel)
            if row == 0:
                axis.legend(fontsize=7, loc="best")
    figure.suptitle(f"Directional spatial structure on matched support · {coherence['generated_field_count']} fields", fontsize=14)
    _save(figure, destination, subtitle=(
        "Shading is an empirical range across generated fields, not a confidence interval. "
        "Total-field similarity can hide meaningful residual changes."))


def _spectra_figure(coherence, destination):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, kind, xlabel in ((axes[0, 0], "radial", "Spatial frequency (cycles/m)"),
                               (axes[0, 1], "angular", "Direction modulo 180° (degrees)")):
        curve = coherence["spectra"][kind]
        edges = np.asarray(curve["coordinate_edges"])
        coordinates = (edges[:-1] + edges[1:]) / 2
        if kind == "angular":
            coordinates = np.rad2deg(coordinates)
        _envelope(axis, coordinates, curve)
        axis.set(title=f"{kind.capitalize()} power", xlabel=xlabel, ylabel="Integrated bin power")
        if kind == "radial":
            peak = max(float(np.max(curve["reference"])),
                       float(np.max(curve["generated_per_field"])), np.finfo(float).tiny)
            axis.set_yscale("symlog", linthresh=max(peak * 1e-8, np.finfo(float).tiny))
            axis.set_ylabel("Integrated bin power (symlog)")
        axis.legend(fontsize=7)
    spectra = coherence["reference_fidelity"]["spectra"]
    reference_power = np.asarray(spectra["reference"]["power_2d"])
    generated_power = np.asarray(spectra["generated"]["power_2d"])
    # A shared display floor keeps zero-power bins finite without changing reported spectra.
    largest = max(float(np.max(reference_power)), float(np.max(generated_power)), np.finfo(float).tiny)
    floor = max(largest * 1e-10, np.finfo(float).tiny)
    limits = (np.log10(floor), np.log10(largest))
    if limits[0] == limits[1]:
        limits = (limits[0] - 1, limits[1] + 1)
    for axis, side, power, label in ((axes[1, 0], "reference", reference_power, "Fixed reference"),
                                      (axes[1, 1], "generated", generated_power, "Generated ensemble mean")):
        data = spectra[side]
        x, y = data["frequency_x_per_m"], data["frequency_y_per_m"]
        handle = axis.imshow(np.log10(np.maximum(power, floor)), origin="lower", aspect="equal",
                             extent=(x[0], x[-1], y[0], y[-1]), cmap="magma",
                             vmin=limits[0], vmax=limits[1], interpolation="nearest")
        axis.set(title=f"2-D spectrum: {label}", xlabel="Column frequency (cycles/m)",
                 ylabel="Row frequency (cycles/m)")
        figure.colorbar(handle, ax=axis, shrink=.85, label="log10 power (display floor)")
    figure.suptitle(f"Spectral coherence of centered log-permeability · {coherence['generated_field_count']} fields", fontsize=14)
    _save(figure, destination, subtitle=(
        "Hann-window periodograms on identical grids; common color scale for both 2-D panels. "
        "Directional power describes texture and does not by itself establish an artifact."))


def _probe_figure(probe, destination):
    figure, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    indices = np.arange(len(probe["expected_log10_mean"]))
    for axis, expected, empirical, title, ylabel in (
        (axes[0, 0], "expected_log10_mean", "empirical_log10_mean", "Pointwise log means", "Mean log10 K"),
        (axes[0, 1], "expected_log10_variance", "empirical_log10_variance", "Pointwise log variances", "Variance of log10 K"),
        (axes[0, 2], "expected_arithmetic_ratio_to_reference", "empirical_arithmetic_ratio_to_reference",
         "Arithmetic means relative to reference", "Mean K / Kref"),
    ):
        axis.plot(indices, probe[expected], "o-", color="#222222", label="Declared retained law")
        axis.plot(indices, probe[empirical], "x--", color="#c56b0c", label="Generated fields")
        axis.set(title=title, xlabel="Stored probe index", ylabel=ylabel)
        axis.grid(alpha=.18)
        axis.legend(fontsize=7)
    qq = probe["standardized_residual_qq"]
    axis = axes[1, 0]
    for index, values in enumerate(qq.get("per_probe_empirical_quantiles", [])):
        if values is not None:
            axis.plot(qq["normal_quantiles"], values, color="#d97706", alpha=.25,
                      label="Individual probes" if index == 0 else None)
    if qq.get("pooled_empirical_quantiles") is not None:
        axis.plot(qq["normal_quantiles"], qq["pooled_empirical_quantiles"], color="#c56b0c", label="Pooled probe residuals")
    axis.plot(qq["normal_quantiles"], qq["normal_quantiles"], color="#222222", label="Standard normal")
    axis.set(title="Standardized residual quantiles", xlabel="Standard normal quantile", ylabel="Empirical residual quantile")
    axis.legend(fontsize=7)
    axis.grid(alpha=.18)
    for axis, key, label in ((axes[1, 1], "expected_correlation", "Declared retained law"),
                              (axes[1, 2], "empirical_correlation", "Generated fields")):
        # None entries denote undefined correlations, e.g. zero-variance probes.
        values = np.asarray(probe[key], dtype=float)
        handle = axis.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1,
                             cmap="RdBu_r", interpolation="nearest")
        axis.set(title=f"Probe correlation: {label}", xlabel="Stored probe index", ylabel="Stored probe index")
        figure.colorbar(handle, ax=axis, shrink=.8)
    figure.suptitle(f"Residual law at {probe['probe_count']} fixed probes · {probe['field_count']} fields", fontsize=14)
    _save(figure, destination, subtitle=(
        "Expected moments use the actual retained KL law and declared centering. "
        "Probe dependence and finite MC/RQMC samples limit apparent agreement; no iid pixel test is implied."))


def _stability_figure(convergence, destination):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    counts = convergence["sample_counts"]
    for axis, key, title, ylabel in (
        (axes[0], "log10_mean_rmse", "Probe mean error", "RMSE in log10 K"),
        (axes[1], "covariance_relative_frobenius_error", "Probe covariance error", "Relative Frobenius error"),
        (axes[2], "correlation_rmse", "Probe correlation error", "Correlation RMSE"),
    ):
        axis.plot(counts, convergence[key], "o-", color="#c56b0c", label="Nested design prefixes")
        axis.set(title=title, xlabel="Generated field count", ylabel=ylabel)
        axis.set_xscale("log", base=2)
        axis.grid(alpha=.2)
    reference = convergence.get("iid_gaussian_expected_covariance_relative_frobenius_rmse")
    if reference is not None and any(value is not None for value in reference):
        axes[1].plot(counts, reference, "--", color="#222222", label="iid MC sampling-error reference")
        axes[1].legend(fontsize=7)
    figure.suptitle("Ensemble-size sensitivity within one validation design", fontsize=13)
    _save(figure, destination, subtitle=(
        "Prefixes share samples and are not independent replications. Apparent stabilization is not a "
        "convergence proof; RQMC error needs independent scrambles. No confidence bounds are shown."))


def _metadata_block(value):
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n```"


def _number(value):
    return "undefined" if value is None else f"{value:.6g}"


def _native_tables(report):
    native = report.get("native_marginals", {})
    lines = ["Reference marginal on native support:", "", _metadata_block(native.get("reference", {})), ""]
    rows = native.get("generated_per_field", [])
    if rows:
        lines += ["Generated marginals on native support:", "",
                  "| Field index | Mean log10 K | SD log10 K | Minimum K (m²) | Median K (m²) | Maximum K (m²) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            median_index = row["quantile_probabilities"].index(.5)
            values = [row["log10_mean"], row["log10_std"], row["range_m2"][0],
                      row["quantiles_m2"][median_index], row["range_m2"][1]]
            lines.append(f"| {row['sample_index']} | " + " | ".join(_number(value) for value in values) + " |")
        lines.append("")
    rows = report.get("native_agreement", [])
    if rows:
        lines += ["Agreement at corresponding native cells (no coarsening):", "",
                  "| Field index | Log10 bias | Log10 RMSE | Spatial correlation | Ratio P05 | Ratio P50 | Ratio P95 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            values = [row["log10_bias"], row["log10_rmse"], row["log10_spatial_correlation"],
                      *row["spatial_ratio_quantiles_05_50_95"]]
            lines.append(f"| {row['sample_index']} | " + " | ".join(_number(value) for value in values) + " |")
        lines.append("")
    return lines


def _markdown_report(report, outputs):
    coherence = report["coherence"]
    fidelity = coherence["reference_fidelity"]
    marginal = fidelity["marginal"]
    source = {key: value for key, value in report.get("source", {}).items() if key != "fields"}
    status = report.get("assessment", {}).get("status", "reference_review_required")
    lines = ["# Generated permeability validation", "", f"Assessment status: **{status}**.", "",
             "This report compares generated fields with one fixed reference scenario. "
             "RUN_1, RUN_2 and RUN_3 remain separate reference scenarios; no geological posterior "
             "or scenario probabilities are inferred.", "",
             "## Evidence and support", "",
             f"Generated field count: **{fidelity['generated_field_count']}**. "
             "Spatial comparisons use the same grid and support for the reference and generated fields. "
             "Native-grid summaries and coarsened spatial summaries describe different scales.", "",
             _metadata_block(report.get("support", {})), "",
             "Source and generation choices, including artifact/reference/design checksums when available:", "",
             _metadata_block(source), "",
             "Individual native field checksums are retained in `validation.json`.", "",
             "## Distributional agreement", "",
             f"Log10-permeability Wasserstein distance: **{marginal['wasserstein_distance_log10']:.6g}**. "
             "This is a descriptive distance between spatial distributions, not a significance test. "
             "The heterogeneous reference produces a mixture over locations; its pooled histogram "
             "need not be lognormal even when each pointwise scenario law is lognormal.", "",
             f"![Distributional diagnostics]({outputs['distribution_png']})", "",
             *_native_tables(report),
             "## Spatial agreement", "",
             "Read the directional variograms, spatial correlations and residual variograms together. "
             "Strong total-field correlation alone can be dominated by the fixed heterogeneous reference. "
             "The residual curves describe the actual perturbations around that reference. "
             "Row and column labels follow array axes; they require orientation metadata before "
             "being interpreted as geographic directions.", "",
             f"![Directional diagnostics]({outputs['correlation_png']})", "",
             "Radial, angular and 2-D spectra reveal scale changes and directional texture on the stated support. "
             "Coarsening can hide fine-scale differences; these figures do not certify native-grid behavior.", "",
             f"![Spectral diagnostics]({outputs['spectra_png']})", ""]
    if "probe_law_png" in outputs:
        lines += ["## Declared residual law", "",
                  "Probe diagnostics compare empirical ensemble moments and dependence with the actual "
                  "retained Gaussian/KL law, including the stated median or arithmetic-mean centering. "
                  "They check implementation consistency with an assumed law; they do not identify "
                  "the correct residual amplitude or covariance from the reference field.", "",
                  f"![Probe law diagnostics]({outputs['probe_law_png']})", ""]
        if "stability_png" in outputs:
            lines += ["Nested prefixes reveal how the same design's probe errors change with sample count. "
                      "These prefixes reuse samples and cannot establish integration uncertainty or convergence "
                      "on their own. Use independent scrambles for RQMC error assessment.", "",
                      f"![Sample-budget sensitivity]({outputs['stability_png']})", ""]
    else:
        lines += ["## Declared residual law", "",
                  "No retained-law probe comparison was supplied. Spatial agreement alone does not "
                  "validate the declared latent-coordinate law.", ""]
    lines += ["## Review before LGCNN uncertainty propagation", "",
              "1. Confirm reference identity, permeability convention, orientation, support and generator settings.",
              "2. Review native marginal ranges and quantiles, local ratios and log-residual errors for plausible perturbations.",
              "3. Inspect directional correlation, residual structure and spectra for unwanted scale or directional changes.",
              "4. Compare probe means, variances and correlations with the retained law; increase the ensemble size "
              "or independent randomized replicates when discrepancies remain unstable.",
              "5. Assess LGCNN training support separately using proven training membership, patch geometry and preprocessing. "
              "Agreement with one nominal reference does not certify LGCNN training support.",
              "6. Check temperature-QoI convergence across sample budgets and KL truncations before a larger scenario study.", "",
              "Empirical P05–P95 envelopes describe variation across the generated fields. They are not confidence intervals. "
              "Pixels within a field are spatially correlated; probes are dependent; scrambled Sobol points are not iid draws. "
              "Small ensembles may poorly resolve tails or covariance. There is no automatic universal approval threshold. "
              "The diagnostics do not reject or filter generated fields and do not change the MC/RQMC/Hermite input law.", "",
              "Assessment supplied by the validation workflow:", "", _metadata_block(report.get("assessment", {})), "",
              "Machine-readable metrics and individual field comparisons are recorded in `validation.json`.", ""]
    return "\n".join(lines)


def write_reference_validation_report(report, output_dir):
    """Write PNGs and a Markdown review from precomputed schema-1 diagnostics.

    Scientific metrics are read from ``report``. This presentation layer never
    samples fields, recomputes fidelity metrics or decides whether UQ is valid.
    Returned paths are relative to ``output_dir`` for portable manifests.
    """
    if report.get("schema_version") != 1:
        raise ValueError("reference validation report requires schema_version=1")
    coherence = report["coherence"]
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {"distribution_png": "validation_distribution.png",
             "correlation_png": "validation_correlation.png",
             "spectra_png": "validation_spectra.png",
             "markdown": "validation_report.md"}
    _distribution_figure(coherence, output / paths["distribution_png"])
    _correlation_figure(coherence, output / paths["correlation_png"])
    _spectra_figure(coherence, output / paths["spectra_png"])
    probe = report.get("probe_law")
    if probe and "expected_log10_mean" in probe:
        paths["probe_law_png"] = "validation_probe_law.png"
        _probe_figure(probe, output / paths["probe_law_png"])
        convergence = probe.get("prefix_convergence")
        if convergence and convergence.get("sample_counts") and max(convergence["sample_counts"]) > 1:
            paths["stability_png"] = "validation_stability.png"
            _stability_figure(convergence, output / paths["stability_png"])
    (output / paths["markdown"]).write_text(_markdown_report(report, paths), encoding="utf-8")
    return paths
