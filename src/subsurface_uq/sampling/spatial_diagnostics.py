"""Matched-support spectral diagnostics, with no sample rejection or PCA."""
from __future__ import annotations

import numpy as np

from .training_compatibility import permeability_ensemble_fidelity


def log_field_spectrum(fields, *, cell_size_m, radial_bins=24, angular_bins=36):
    """Mean 2-D periodogram of centered log10 fields with a Hann taper.

    Power sums equal tapered log variance divided by mean taper squared.
    Angular power is folded modulo pi. It measures directional texture; an
    axis preference is not itself proof of an artifact in anisotropic data.
    """
    values = np.asarray(fields, dtype=float)
    if values.ndim == 2:
        values = values[None]
    if (values.ndim != 3 or min(values.shape[1:]) < 3
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or not np.isfinite(cell_size_m) or cell_size_m <= 0
            or radial_bins <= 0 or angular_bins <= 0):
        raise ValueError("positive finite [N,H,W] fields (H,W>=3), cell size and bins required")
    height, width = values.shape[1:]
    taper = np.outer(np.hanning(height), np.hanning(width))
    taper_energy = float(np.mean(taper ** 2))
    power = np.zeros((height, width), dtype=float)
    for field in values:
        log_field = np.log10(field)
        transformed = np.fft.fftshift(np.fft.fft2((log_field - log_field.mean()) * taper))
        power += np.abs(transformed) ** 2 / ((height * width) ** 2 * taper_energy)
    power /= len(values)
    fy = np.fft.fftshift(np.fft.fftfreq(height, d=cell_size_m))
    fx = np.fft.fftshift(np.fft.fftfreq(width, d=cell_size_m))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.hypot(xx, yy)
    angle = np.mod(np.arctan2(yy, xx), np.pi)
    active = radius > 0
    radial_edges = np.linspace(0, float(radius.max()), int(radial_bins) + 1)
    angular_edges = np.linspace(0, np.pi, int(angular_bins) + 1)
    radial_power, _ = np.histogram(radius[active], bins=radial_edges, weights=power[active])
    angular_power, _ = np.histogram(angle[active], bins=angular_edges, weights=power[active])
    nonzero_power = float(power[active].sum())
    axis_distance = np.minimum.reduce((angle, np.abs(angle - np.pi / 2), np.pi - angle))
    axis_fraction = (float(power[active & (axis_distance <= np.deg2rad(10))].sum()) / nonzero_power
                     if nonzero_power > 0 else None)
    return {
        "space": "centered_log10_permeability", "cell_size_m": float(cell_size_m),
        "shape": [height, width], "field_count": len(values), "window": "hann",
        "frequency_x_per_m": fx.tolist(), "frequency_y_per_m": fy.tolist(),
        "power_2d": power.tolist(), "total_taper_corrected_power": float(power.sum()),
        "radial_edges_per_m": radial_edges.tolist(), "radial_power": radial_power.tolist(),
        "angular_edges_rad": angular_edges.tolist(), "angular_power": angular_power.tolist(),
        "axis_band_half_width_deg": 10, "axis_band_power_fraction": axis_fraction,
    }


def compare_reference_ensemble(reference, generated, *, cell_size_m, max_lag_cells=24):
    """Total-field fidelity plus 2-D/radial/angular spectral comparison."""
    reference = np.asarray(reference)
    if reference.ndim == 2:
        reference = reference[None]
    generated = np.asarray(generated)
    fidelity = permeability_ensemble_fidelity(
        reference, generated, cell_size_m=cell_size_m, spatial_stride=1,
        max_lag_cells=min(max_lag_cells, min(reference.shape[1:]) - 1),
    )
    spectra = {
        "reference": log_field_spectrum(reference, cell_size_m=cell_size_m),
        "generated": log_field_spectrum(generated, cell_size_m=cell_size_m),
    }
    for key in ("radial_power", "angular_power"):
        left, right = (np.asarray(spectra[side][key]) for side in ("reference", "generated"))
        scale = float(np.linalg.norm(left))
        spectra[f"{key}_relative_l2"] = float(np.linalg.norm(right - left) / scale) if scale > 0 else None
    # Both diagonals are retained separately; rotations need not preserve them.
    diagonal = {}
    for name, flip in (("positive", False), ("negative", True)):
        curves = []
        for ensemble in (reference, generated):
            logs = np.log10(ensemble[:, :, ::-1] if flip else ensemble)
            curves.append([float(0.5 * np.mean((logs[:, lag:, lag:] - logs[:, :-lag, :-lag]) ** 2))
                           for lag in range(1, min(max_lag_cells, min(logs.shape[1:]) - 1) + 1)])
        diagonal[name] = {"reference": curves[0], "generated": curves[1],
                          "distance_m": (np.arange(1, len(curves[0]) + 1) * np.sqrt(2) * cell_size_m).tolist()}
    fidelity["both_diagonal_variograms"] = diagonal
    fidelity["spectra"] = spectra
    fidelity["marginal"]["reference_range_m2"] = [float(reference.min()), float(reference.max())]
    fidelity["marginal"]["generated_range_m2"] = [float(generated.min()), float(generated.max())]
    fidelity["marginal"]["spatial_histogram_semantics"] = "mixture over heterogeneous locations; not one pointwise lognormal marginal"
    fidelity["spectral_interpretation"] = "matched-grid descriptive comparison; no universal acceptance threshold"
    return fidelity
