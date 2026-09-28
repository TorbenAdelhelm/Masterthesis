from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from ..sampling.calibration import (
    CovarianceCalibrationResult,
    correlation_for_offsets,
)
from ..sampling.geospatial import (
    LGCNNDomainGeoreference,
    ReferencePermeabilitySurface,
    orient_raw_field,
)

Array = np.ndarray


def _model_semivariogram(
    result: CovarianceCalibrationResult,
    direction: str,
    distances_m: Array,
) -> Array:
    distances = np.asarray(distances_m, dtype=np.float64)
    if direction == "y":
        dy, dx = distances, np.zeros_like(distances)
    elif direction == "x":
        dy, dx = np.zeros_like(distances), distances
    elif direction == "diag":
        component = distances / np.sqrt(2.0)
        dy, dx = component, component
    else:
        raise ValueError("direction must be y, x or diag")
    rho = correlation_for_offsets(
        result.covariance_model,
        delta_y_m=dy,
        delta_x_m=dx,
        length_scale_y_m=result.length_scale_y_m,
        length_scale_x_m=result.length_scale_x_m,
    )
    structured_std = (
        result.std_log10_k
        if result.structured_std_log10_k is None
        else result.structured_std_log10_k
    )
    semivariance = (
        result.nugget_std_log10_k**2
        + structured_std**2 * (1.0 - rho)
    )
    # A nugget is the discontinuity for h>0; by definition gamma(0)=0.
    semivariance = np.where(distances == 0.0, 0.0, semivariance)
    return semivariance


def plot_variogram_fits(
    variograms: Mapping[str, tuple[Array, Array]],
    results: Sequence[CovarianceCalibrationResult],
    destination: str | Path,
) -> Path:
    """Plot empirical and fitted y/x/diagonal semivariograms."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, direction, title in zip(
        axes,
        ("y", "x", "diag"),
        ("y direction", "x direction", "main diagonal"),
    ):
        distances, empirical = variograms[direction]
        axis.plot(distances, empirical, marker="o", markersize=3, label="empirical")
        dense = np.linspace(0.0, float(np.max(distances)), 300)
        for result in results:
            axis.plot(
                dense,
                _model_semivariogram(result, direction, dense),
                label=result.covariance_model,
            )
        axis.set_title(title)
        axis.set_xlabel("lag distance [m]")
        axis.set_ylabel("semivariance of log10(K)")
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_length_scale_comparison(
    results: Sequence[CovarianceCalibrationResult],
    destination: str | Path,
) -> Path:
    """Compare fitted directional length scales and anisotropy ratios."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [result.covariance_model for result in results]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(
        x - width / 2.0,
        [result.length_scale_x_m for result in results],
        width,
        label="ell_x",
    )
    axis.bar(
        x + width / 2.0,
        [result.length_scale_y_m for result in results],
        width,
        label="ell_y",
    )
    axis.set_xticks(x, labels, rotation=15)
    axis.set_ylabel("fitted length scale [m]")
    axis.set_title("Covariance length-scale comparison")
    axis.legend(loc="best")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_heldout_reconstruction(
    *,
    truth_log10_k: Array,
    posterior_mean_log10_k: Array,
    posterior_std_log10_k: Array,
    observation_indices: Array,
    cell_size_m: float,
    model_name: str,
    destination: str | Path,
) -> Path:
    """Plot hidden truth, conditional mean/std and reconstruction error."""

    truth = np.asarray(truth_log10_k, dtype=np.float64)
    mean = np.asarray(posterior_mean_log10_k, dtype=np.float64)
    std = np.asarray(posterior_std_log10_k, dtype=np.float64)
    if truth.shape != mean.shape or truth.shape != std.shape or truth.ndim != 2:
        raise ValueError("truth, posterior mean and posterior std must share one 2-D shape")
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = truth.shape
    extent = (0.0, w * cell_size_m, 0.0, h * cell_size_m)
    observations = np.asarray(observation_indices, dtype=np.int64)

    vmin = min(float(np.min(truth)), float(np.min(mean)))
    vmax = max(float(np.max(truth)), float(np.max(mean)))
    max_error = float(np.max(np.abs(mean - truth)))
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    panels = (
        (axes[0, 0], truth, "held-out truth", vmin, vmax, "log10(K / m²)"),
        (axes[0, 1], mean, "conditional mean", vmin, vmax, "log10(K / m²)"),
        (axes[1, 0], std, "conditional std", None, None, "std. dev. log10(K / m²)"),
        (
            axes[1, 1],
            mean - truth,
            "conditional mean - truth",
            -max_error if max_error > 0.0 else None,
            max_error if max_error > 0.0 else None,
            "log10(K / m²) error",
        ),
    )
    for axis, field, title, lower, upper, label in panels:
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            vmin=lower,
            vmax=upper,
        )
        if observations.size:
            axis.scatter(
                (observations[:, 1] + 0.5) * cell_size_m,
                (observations[:, 0] + 0.5) * cell_size_m,
                marker="x",
                s=20,
                label="synthetic borehole",
            )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        figure.colorbar(image, ax=axis, label=label)
    axes[0, 0].legend(loc="best")
    figure.suptitle(f"Held-out reconstruction: {model_name}")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_heldout_metric_comparison(
    rows: Sequence[Mapping[str, object]],
    destination: str | Path,
) -> Path:
    """Plot the principal held-out reconstruction metrics across models."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["covariance_model"]) for row in rows]
    rmse = [float(row["log10_mean_rmse"]) for row in rows]
    coverage = [float(row["gaussian_90pct_coverage"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, rmse)
    axes[0].set_ylabel("RMSE in log10(K)")
    axes[0].set_title("Held-out conditional-mean error")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, coverage)
    axes[1].axhline(0.90, linestyle="--", label="nominal 90%")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("coverage fraction")
    axes[1].set_title("Gaussian posterior 90% coverage")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(loc="best")
    axes[1].grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_measurement_cv_comparison(
    rows: Sequence[Mapping[str, object]],
    destination: str | Path,
) -> Path:
    """Plot spatial-CV RMSE, coverage and Gaussian NLPD for real measurements."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["covariance_model"]) for row in rows]
    rmse = [float(row["rmse_log10_k"]) for row in rows]
    coverage = [float(row["coverage_90"]) for row in rows]
    nlpd = [float(row["gaussian_nlpd"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar(labels, rmse)
    axes[0].set_ylabel("RMSE in log10(K_h)")
    axes[0].set_title("Spatial-CV prediction error")
    axes[1].bar(labels, coverage)
    axes[1].axhline(0.90, linestyle="--", label="nominal 90%")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("coverage fraction")
    axes[1].set_title("Spatial-CV posterior coverage")
    axes[1].legend(loc="best")
    axes[2].bar(labels, nlpd)
    axes[2].set_ylabel("Gaussian NLPD")
    axes[2].set_title("Spatial-CV predictive density")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_nugget_fraction_comparison(
    results: Sequence[CovarianceCalibrationResult],
    destination: str | Path,
) -> Path:
    """Plot the fitted nugget fraction of total log-space variance."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [result.covariance_model for result in results]
    fractions = [float(result.nugget_fraction) for result in results]
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.bar(labels, fractions)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("nugget fraction")
    axis.set_title("Fitted nugget fraction of total log10(K_h) variance")
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_georeference_alignment(
    *,
    reference: ReferencePermeabilitySurface,
    raw_field: Array,
    mapping: LGCNNDomainGeoreference,
    destination: str | Path,
) -> Path:
    """Visual check of the inferred Munich crop and oriented raw permeability field."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_log = np.where(
        reference.active_mask, np.log10(reference.values), np.nan
    )
    raw_geo = orient_raw_field(raw_field, mapping.transform)
    raw_corrected = (
        np.log10(np.asarray(raw_geo, dtype=np.float64))
        - mapping.log10_unit_shift_raw_minus_reference
    )
    finite_reference = reference_log[np.isfinite(reference_log)]
    finite_raw = raw_corrected[np.isfinite(raw_corrected)]
    combined = np.concatenate((finite_reference, finite_raw))
    vmin, vmax = np.quantile(combined, [0.01, 0.99])

    dx = reference.spacing_x_m
    dy = reference.spacing_y_m
    reference_extent = (
        float(reference.x_m[0] - 0.5 * dx),
        float(reference.x_m[-1] + 0.5 * dx),
        float(reference.y_m[0] - 0.5 * dy),
        float(reference.y_m[-1] + 0.5 * dy),
    )
    raw_extent = (
        mapping.west_edge_m,
        mapping.east_edge_m,
        mapping.south_edge_m,
        mapping.north_edge_m,
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    image0 = axes[0].imshow(
        reference_log,
        origin="lower",
        extent=reference_extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    rectangle_x = [
        mapping.west_edge_m,
        mapping.east_edge_m,
        mapping.east_edge_m,
        mapping.west_edge_m,
        mapping.west_edge_m,
    ]
    rectangle_y = [
        mapping.south_edge_m,
        mapping.south_edge_m,
        mapping.north_edge_m,
        mapping.north_edge_m,
        mapping.south_edge_m,
    ]
    axes[0].plot(rectangle_x, rectangle_y, linewidth=2, label=mapping.run_name)
    axes[0].set_title("Munich reference + inferred LGCNN crop")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="best")
    figure.colorbar(image0, ax=axes[0], label=f"log10({reference.value_name})")

    image1 = axes[1].imshow(
        raw_corrected,
        origin="lower",
        extent=raw_extent,
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(
        f"Oriented raw field (unit-shift corrected)\n{mapping.transform}"
    )
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    figure.colorbar(image1, ax=axes[1], label=f"log10({reference.value_name})")

    figure.suptitle(
        f"Georeference validation: corr={mapping.correlation:.4f}, "
        f"RMSE={mapping.centered_rmse_log10:.4f}"
    )
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
