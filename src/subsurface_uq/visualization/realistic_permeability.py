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
from ..sampling.new_domain import NewLGCNNDomain

Array = np.ndarray



def save_permeability_field_png(
    permeability_m2: Array,
    destination: str | Path,
    *,
    vmin_log10_k: float | None = None,
    vmax_log10_k: float | None = None,
) -> Path:
    """Save one permeability realization as a PNG in log10(K [m^2]).

    The PNG is a visualization artifact, not a lossless numerical
    representation. The production generator therefore keeps the float32 NPY
    sample alongside this image. Supplying a common vmin/vmax gives all samples
    the same color scale, which makes visual comparisons meaningful.
    """

    values = np.asarray(permeability_m2, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("permeability field must be two-dimensional")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("permeability field must be finite and strictly positive")

    lower = None if vmin_log10_k is None else float(vmin_log10_k)
    upper = None if vmax_log10_k is None else float(vmax_log10_k)
    if lower is not None and not np.isfinite(lower):
        raise ValueError("vmin_log10_k must be finite")
    if upper is not None and not np.isfinite(upper):
        raise ValueError("vmax_log10_k must be finite")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("vmin_log10_k must be smaller than vmax_log10_k")

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(
        path,
        np.log10(values),
        origin="lower",
        cmap="viridis",
        vmin=lower,
        vmax=upper,
    )
    return path


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



def plot_generated_permeability_comparison(
    *,
    generated_permeability_m2: Array,
    conditioned_reference_log10_k: Array,
    training_reference_permeability_m2: Array,
    training_reference_label: str,
    domain: NewLGCNNDomain,
    measurement_x_m: Array,
    measurement_y_m: Array,
    shared_log10_limits: tuple[float, float],
    destination: str | Path,
) -> Path:
    """Save a four-panel visual comparison for one generated realization.

    Panels use one robust physical log10(K) color scale for the generated,
    conditioned-reference and selected training field. The final panel shows the
    generated minus conditioned-reference residual in log10 units. Training
    fields are compared in their local array coordinates because they are not
    assumed to be pixelwise registered to the newly selected Munich domain.
    """

    generated = np.asarray(generated_permeability_m2, dtype=np.float64)
    reference = np.asarray(conditioned_reference_log10_k, dtype=np.float64)
    training = np.asarray(training_reference_permeability_m2, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2 or training.ndim != 2:
        raise ValueError("generated/reference/training comparison fields must be 2-D")
    if generated.shape != reference.shape or generated.shape != training.shape:
        raise ValueError("comparison fields must share one spatial shape")
    if not np.all(np.isfinite(generated)) or np.any(generated <= 0.0):
        raise ValueError("generated permeability must be finite and positive")
    if not np.all(np.isfinite(training)) or np.any(training <= 0.0):
        raise ValueError("training permeability must be finite and positive")
    if not np.all(np.isfinite(reference)):
        raise ValueError("conditioned reference must be finite")

    lower, upper = (float(shared_log10_limits[0]), float(shared_log10_limits[1]))
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("shared_log10_limits must contain finite lower < upper")

    generated_log = np.log10(generated)
    training_log = np.log10(training)
    residual = generated_log - reference
    residual_limit = float(np.quantile(np.abs(residual), 0.99))
    residual_limit = max(residual_limit, np.finfo(float).eps)

    projected_extent = (
        domain.west_edge_m,
        domain.east_edge_m,
        domain.south_edge_m,
        domain.north_edge_m,
    )
    local_extent = (
        0.0,
        domain.size_m[1],
        0.0,
        domain.size_m[0],
    )

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    image0 = axes[0, 0].imshow(
        generated_log,
        origin="lower",
        extent=projected_extent,
        aspect="equal",
        vmin=lower,
        vmax=upper,
        cmap="viridis",
    )
    axes[0, 0].scatter(
        measurement_x_m,
        measurement_y_m,
        marker="x",
        s=16,
        linewidths=0.8,
        label="conditioning measurement",
    )
    axes[0, 0].set_title("Generated realization")
    axes[0, 0].set_xlabel("projected x [m]")
    axes[0, 0].set_ylabel("projected y [m]")
    axes[0, 0].legend(loc="best")

    image1 = axes[0, 1].imshow(
        reference,
        origin="lower",
        extent=projected_extent,
        aspect="equal",
        vmin=lower,
        vmax=upper,
        cmap="viridis",
    )
    axes[0, 1].scatter(
        measurement_x_m,
        measurement_y_m,
        marker="x",
        s=16,
        linewidths=0.8,
    )
    axes[0, 1].set_title("Measurement-conditioned reference")
    axes[0, 1].set_xlabel("projected x [m]")
    axes[0, 1].set_ylabel("projected y [m]")

    image2 = axes[1, 0].imshow(
        training_log,
        origin="lower",
        extent=local_extent,
        aspect="equal",
        vmin=lower,
        vmax=upper,
        cmap="viridis",
    )
    axes[1, 0].set_title(f"Closest training reference: {training_reference_label}")
    axes[1, 0].set_xlabel("local x [m]")
    axes[1, 0].set_ylabel("local y [m]")

    image3 = axes[1, 1].imshow(
        residual,
        origin="lower",
        extent=projected_extent,
        aspect="equal",
        vmin=-residual_limit,
        vmax=residual_limit,
        cmap="coolwarm",
    )
    axes[1, 1].set_title("Generated - conditioned reference")
    axes[1, 1].set_xlabel("projected x [m]")
    axes[1, 1].set_ylabel("projected y [m]")

    figure.colorbar(
        image0,
        ax=[axes[0, 0], axes[0, 1], axes[1, 0]],
        label="log10(k / m²)",
        shrink=0.88,
    )
    figure.colorbar(
        image3,
        ax=axes[1, 1],
        label="difference in log10(k / m²)",
    )
    figure.suptitle(
        "Permeability realization fidelity comparison\n"
        "common physical scale from training q01-q99"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_new_domain_summary(
    *,
    mean_log10_k: Array,
    std_log10_k: Array,
    domain: NewLGCNNDomain,
    measurement_x_m: Array,
    measurement_y_m: Array,
    destination: str | Path,
) -> Path:
    """Plot empirical ensemble mean/std on the new projected LGCNN domain."""

    mean = np.asarray(mean_log10_k, dtype=np.float64)
    std = np.asarray(std_log10_k, dtype=np.float64)
    if mean.shape != domain.shape or std.shape != domain.shape:
        raise ValueError("mean/std fields must match the new-domain shape")
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    extent = (
        domain.west_edge_m,
        domain.east_edge_m,
        domain.south_edge_m,
        domain.north_edge_m,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    image0 = axes[0].imshow(mean, origin="lower", extent=extent, aspect="equal")
    axes[0].scatter(measurement_x_m, measurement_y_m, marker="x", s=18, label="conditioning measurement")
    axes[0].set_title("Empirical mean log10(k)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].legend(loc="best")
    figure.colorbar(image0, ax=axes[0], label="log10(k / m^2)")

    image1 = axes[1].imshow(std, origin="lower", extent=extent, aspect="equal")
    axes[1].scatter(measurement_x_m, measurement_y_m, marker="x", s=18)
    axes[1].set_title("Empirical std. dev. log10(k)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    figure.colorbar(image1, ax=axes[1], label="std. dev. log10(k)")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path
