"""Describe and plot a realization relative to its fixed nominal reference."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _log_fields(reference, generated):
    reference, generated = (np.asarray(value, dtype=np.float64) for value in (reference, generated))
    if reference.ndim != 2 or reference.shape != generated.shape or not reference.size:
        raise ValueError("reference and generated permeability must have the same nonempty 2-D shape")
    if any(not np.all(np.isfinite(value)) or np.any(value <= 0) for value in (reference, generated)):
        raise ValueError("permeability must be finite and strictly positive")
    return np.log10(reference), np.log10(generated)


def reference_field_agreement(reference, generated):
    """Full-grid spatial descriptors of one realization, without acceptance rules.

    The ratio quantiles describe pixels within this field, not probabilities
    across geological RUNs. A high total-field correlation can coexist with
    important local changes because the nominal heterogeneous field dominates.
    """
    log_reference, log_generated = _log_fields(reference, generated)
    residual = log_generated - log_reference
    a = log_reference - log_reference.mean()
    b = log_generated - log_generated.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    correlation = (float(np.clip(np.sum(a * b) / denominator, -1., 1.))
                   if denominator > 0 and np.ptp(log_reference) > 0 and np.ptp(log_generated) > 0 else None)
    ratio_quantiles = 10. ** np.quantile(residual, [.05, .5, .95])
    return {
        "pixel_count": int(residual.size),
        "shape": list(residual.shape),
        "log10_bias": float(residual.mean()),
        "log10_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "log10_mae": float(np.mean(np.abs(residual))),
        "log10_max_abs_difference": float(np.max(np.abs(residual))),
        "log10_spatial_correlation": correlation,
        "spatial_ratio_quantiles_05_50_95": ratio_quantiles.tolist(),
        "fraction_pixels_within_10_percent": float(np.mean(
            (residual >= np.log10(.90)) & (residual <= np.log10(1.10)))),
        "reference_log10_range": [float(log_reference.min()), float(log_reference.max())],
        "generated_log10_range": [float(log_generated.min()), float(log_generated.max())],
        "interpretation": "full-grid agreement with the fixed nominal reference; descriptive spatial statistics, not calibration, training-support certification or sample acceptance",
    }


def plot_reference_field_comparison(reference, generated, destination, *,
                                    cell_size_m, title, log10_limits=None,
                                    residual_limit=None, agreement=None):
    """Save reference, realization, log-ratio and pixel correspondence as PNG.

    Both permeability panels share their color scale. The residual uses a
    symmetric scale about zero. Only the scatter display is subsampled; all
    agreement statistics use the complete native grid.
    """
    log_reference, log_generated = _log_fields(reference, generated)
    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive and finite")
    if log10_limits is None:
        log10_limits = (min(log_reference.min(), log_generated.min()),
                        max(log_reference.max(), log_generated.max()))
        if log10_limits[0] == log10_limits[1]:
            log10_limits = (log10_limits[0] - .05, log10_limits[1] + .05)
    if len(log10_limits) != 2 or not np.all(np.isfinite(log10_limits)) or log10_limits[0] >= log10_limits[1]:
        raise ValueError("log10_limits must be an increasing finite pair")
    residual = log_generated - log_reference
    if residual_limit is None:
        residual_limit = max(float(np.max(np.abs(residual))), 1e-6)
    if not np.isfinite(residual_limit) or residual_limit <= 0:
        raise ValueError("residual_limit must be positive and finite")
    stats = reference_field_agreement(reference, generated) if agreement is None else agreement
    rows, columns = log_reference.shape
    extent = (0., columns * cell_size_m / 1000., 0., rows * cell_size_m / 1000.)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9.5), constrained_layout=True)
    try:
        for axis, values, label in zip(axes.flat[:3],
                                      (log_reference, log_generated, residual),
                                      ("Fixed reference permeability", "Generated permeability", "Multiplicative change from reference")):
            difference = axis is axes[1, 0]
            handle = axis.imshow(values, origin="lower", extent=extent, aspect="equal",
                                 cmap="RdBu_r" if difference else "viridis",
                                 vmin=-residual_limit if difference else log10_limits[0],
                                 vmax=residual_limit if difference else log10_limits[1],
                                 interpolation="nearest")
            axis.set(title=label, xlabel="Array-column distance (km)", ylabel="Array-row distance (km)")
            figure.colorbar(handle, ax=axis, shrink=.85,
                            label="log10(K / Kref)" if difference else "log10 K (stored convention)")
        axis = axes[1, 1]
        stride = max(1, int(np.ceil(np.sqrt(log_reference.size / 4096))))
        axis.scatter(log_reference[::stride, ::stride].ravel(),
                     log_generated[::stride, ::stride].ravel(), s=3, alpha=.18, rasterized=True)
        axis.plot(log10_limits, log10_limits, color="black", linewidth=1, label="Exact agreement")
        axis.set(xlim=log10_limits, ylim=log10_limits, aspect="equal",
                 title="Pixel correspondence (display subset)",
                 xlabel="Reference log10 K", ylabel="Generated log10 K")
        correlation = stats["log10_spatial_correlation"]
        correlation_label = "undefined (constant field)" if correlation is None else f"{correlation:.5f}"
        q05, q50, q95 = stats["spatial_ratio_quantiles_05_50_95"]
        text = (f"Full-grid log10 RMSE: {stats['log10_rmse']:.4f}\n"
                f"Full-grid log10 bias: {stats['log10_bias']:+.4f}\n"
                f"Spatial log10 correlation: {correlation_label}\n"
                f"K / Kref spatial P05 / P50 / P95:\n{q05:.3f} / {q50:.3f} / {q95:.3f}\n"
                f"Pixels within ±10%: {100 * stats['fraction_pixels_within_10_percent']:.1f}%")
        axis.text(.03, .97, text, transform=axis.transAxes, va="top", fontsize=9,
                  bbox={"facecolor": "white", "alpha": .9, "edgecolor": ".8"})
        figure.suptitle(title, fontsize=13)
        figure.supxlabel(f"Permeability inputs • native {cell_size_m:g} m grid • statistics use all {rows * columns:,} cells", fontsize=10)
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=170)
    finally:
        plt.close(figure)
    return path
