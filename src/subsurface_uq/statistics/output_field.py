"""Train-only snapshot KL and held-out Gaussian-field diagnostics.

The spatial basis is empirical and nonstationary. Gaussianity is an additional
assumption on its coefficients, never a consequence of PCA decorrelation.
All native-grid operations stream spatial blocks, without a cell x cell matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import stats


def split_indices(n, train_fraction=0.7, seed=20261110):
    if not 0 < train_fraction < 1 or n < 12:
        raise ValueError("need at least 12 draws and a train fraction strictly between 0 and 1")
    n_train = int(n * train_fraction)
    if n_train < 3 or n - n_train < 8:
        raise ValueError("need at least 3 training and 8 held-out draws")
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[:n_train]), np.sort(order[n_train:])


@dataclass
class FieldKL:
    mean: np.ndarray
    modes: np.ndarray
    train_variance: np.ndarray
    info: dict

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        return cls(np.load(root / "mean.npy", mmap_mode="r"),
                   np.load(root / "modes.npy", mmap_mode="r"),
                   np.load(root / "training_variance.npy", mmap_mode="r"),
                   json.loads((root / "model.json").read_text()))

    @property
    def rank(self):
        return self.modes.shape[0]

    def block(self, scores, start, stop):
        """Continuous temperature reconstruction; no clipping or truncation."""
        return np.asarray(scores, dtype=np.float64) @ self.modes[:, start:stop] + self.mean[start:stop]


def fit_kl(fields, train_indices, directory, *, max_rank=32, variance_fraction=0.99,
           chunk_cells=4096, progress=lambda message: None):
    """Fit Euclidean-grid KL via the training-draw Gram matrix (method of snapshots).

    Uniform physical cell weights change eigenvalue units, not the reconstructed
    fields. Modes are stored as float32; scores and validations use these stored
    modes so serialization error is included. Native temperature means are float64.
    """
    if fields.ndim != 3 or fields.dtype.kind != "f" or fields.dtype.itemsize not in (4, 8):
        raise ValueError("fields must be floating float32/float64 [draw, row, column]")
    if max_rank < 1 or chunk_cells < 1 or not 0 < variance_fraction <= 1:
        raise ValueError("positive rank/chunk size and variance fraction in (0,1] required")
    indices = np.asarray(train_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 3
            or len(np.unique(indices)) != len(indices) or np.any(indices < 0)
            or np.any(indices >= len(fields))):
        raise ValueError("unique valid integer training indices required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    n = len(indices)
    flat = fields.reshape(len(fields), -1)
    cells = flat.shape[1]
    mean = np.lib.format.open_memmap(root / "mean.npy", mode="w+", dtype="float64", shape=(cells,))
    variance = np.lib.format.open_memmap(root / "training_variance.npy", mode="w+", dtype="float64", shape=(cells,))
    gram = np.zeros((n, n), dtype=np.float64)
    progress("Accumulating training covariance in spatial blocks")
    for start in range(0, cells, chunk_cells):
        stop = min(start + chunk_cells, cells)
        block = np.asarray(flat[indices, start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite training temperatures")
        # Center relative to one draw: exactly constant float64 fields must not
        # acquire a spurious covariance from repeated-sum roundoff.
        mean[start:stop] = block[0] + (block - block[0]).mean(axis=0)
        block -= mean[start:stop]
        variance[start:stop] = np.sum(block * block, axis=0) / (n - 1)
        gram += (block @ block.T) / (n - 1)
    values, vectors = np.linalg.eigh(gram)
    values, vectors = values[::-1], vectors[:, ::-1]
    tolerance = max(float(values[0]), 0.) * n * np.finfo(float).eps
    if values[-1] < -max(tolerance, np.finfo(float).tiny):
        raise ValueError("training covariance is numerically indefinite")
    positive = values > tolerance
    spectrum = np.maximum(values, 0.)
    total = float(spectrum.sum())
    available = min(int(positive.sum()), n - 1)
    required = int(np.searchsorted(np.cumsum(spectrum), variance_fraction * total) + 1) if total else 0
    rank = min(required, max_rank, available)
    modes = np.lib.format.open_memmap(root / "modes.npy", mode="w+", dtype="float32", shape=(rank, cells))
    progress(f"Writing {rank} spatial KL modes")
    for start in range(0, cells, chunk_cells):
        stop = min(start + chunk_cells, cells)
        centered = np.asarray(flat[indices, start:stop], dtype=np.float64) - mean[start:stop]
        if rank:
            modes[:, start:stop] = (vectors[:, :rank].T @ centered) / np.sqrt((n - 1) * spectrum[:rank, None])
    retained = float(spectrum[:rank].sum() / total) if total else 1.
    info = {"schema_version": 1, "shape": list(fields.shape[1:]), "source_dtype": str(fields.dtype),
            "n_train": n, "train_indices": indices.tolist(), "rank": rank,
            "numerical_rank": available, "max_rank": max_rank, "variance_fraction_target": variance_fraction,
            "variance_fraction_retained": retained, "variance_target_met": bool(retained >= variance_fraction - 1e-12),
            "training_variance_trace_C2": total, "eigenvalues_C2": spectrum.tolist(),
            "inner_product": "unweighted sum over uniform native cells",
            "model": "T(x)=training_mean(x)+sum_j score_j*mode_j(x); Gaussian scores are a separate candidate law",
            "mode_dtype": "float32", "clipping": False}
    (root / "model.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    mean.flush(); variance.flush(); modes.flush()
    return FieldKL.load(root)


def project(fields, model, chunk_cells=4096):
    if list(fields.shape[1:]) != model.info["shape"] or chunk_cells < 1:
        raise ValueError("incompatible field shape or chunk size")
    flat = fields.reshape(len(fields), -1)
    scores = np.zeros((len(fields), model.rank))
    for start in range(0, flat.shape[1], chunk_cells):
        stop = min(start + chunk_cells, flat.shape[1])
        block = np.asarray(flat[:, start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("nonfinite temperatures in projected fields")
        scores += (block - model.mean[start:stop]) @ model.modes[:, start:stop].T
    return scores


def score_law(training_scores):
    scores = np.asarray(training_scores, dtype=np.float64)
    mean = scores.mean(axis=0)
    centered = scores - mean
    covariance = centered.T @ centered / (len(scores) - 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.maximum(eigenvalues, 0.))
    return mean, covariance, root


def gaussian_scores(training_scores, n_draws, seed):
    if n_draws < 8:
        raise ValueError("need at least 8 Gaussian candidate draws")
    mean, _, root = score_law(training_scores)
    return mean + np.random.default_rng(seed).standard_normal((n_draws, len(mean))) @ root.T


def normality_screen(training_scores, heldout_scores, *, seed=20261111, n_directions=32):
    """Necessary Gaussianity screens, including joint projections, Holm corrected.

    Whitening uses training moments only. Non-rejection is never acceptance of a
    Gaussian process; low power, estimated moments and basis selection remain.
    """
    mean, covariance, _ = score_law(training_scores)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    active = eigenvalues > (max(float(eigenvalues.max()), 0.) * 1e-10) if len(mean) else np.zeros(0, bool)
    white = ((heldout_scores - mean) @ vectors[:, active]) / np.sqrt(eigenvalues[active])
    d = white.shape[1]
    directions = []
    labels = []
    for i in range(min(d, 8)):
        direction = np.zeros(d); direction[i] = 1
        directions.append(direction); labels.append(f"whitened_coordinate_{i}")
    for i in range(min(d - 1, 4)):
        for sign in (-1, 1):
            direction = np.zeros(d); direction[i] = 1 / np.sqrt(2); direction[i + 1] = sign / np.sqrt(2)
            directions.append(direction); labels.append(f"pair_{i}_{i+1}_{sign:+d}")
    if d:
        random = np.random.default_rng(seed).standard_normal((n_directions, d))
        random /= np.linalg.norm(random, axis=1)[:, None]
        directions.extend(random); labels.extend(f"random_projection_{i}" for i in range(n_directions))
    checks = []
    for label, direction in zip(labels, directions):
        values = white @ direction
        # A constant projection can have an atom under a non-Gaussian joint law.
        if np.ptp(values) == 0:
            p, skew, kurt = 0., None, None
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                p = float(stats.normaltest(values).pvalue) if len(values) >= 8 else None
            skew = float(stats.skew(values, bias=False))
            kurt = float(stats.kurtosis(values, bias=False))
        checks.append({"projection": label, "p_value": p, "skewness": skew, "excess_kurtosis": kurt})
    tested = sorted((i for i, item in enumerate(checks) if item["p_value"] is not None), key=lambda i: checks[i]["p_value"])
    previous = 0.
    for order, i in enumerate(tested):
        adjusted = min(1., max(previous, (len(tested) - order) * checks[i]["p_value"]))
        checks[i]["holm_p_value"] = previous = adjusted
    rejected = sum(c.get("holm_p_value", 1.) < .05 for c in checks)
    return {"method": "D'Agostino-Pearson held-out projection tests; Holm within this case only",
            "n_heldout": len(heldout_scores), "active_dimensions": d, "checks": checks,
            "rejected_at_0.05": rejected,
            "interpretation": ("evidence against joint Gaussianity" if rejected else
                               "no rejection at this sample size; not evidence of adequacy") if d else "degenerate constant field",
            "limitations": "necessary finite-projection screen only; no omnibus proof, no across-case correction"}


def native_exceedance(temperatures, background, threshold, dtype):
    values = np.asarray(temperatures, dtype=dtype)
    return values - np.asarray(background, dtype=dtype) > np.asarray(threshold, dtype=dtype)


def native_boundary(background, threshold, dtype):
    """Continuous boundary giving the strict native-dtype event after rounding.

    Gaussian temperatures are continuous before native rounding. At the midpoint
    between adjacent floats the rounding tie has probability zero for nonzero SD.
    """
    dtype = np.dtype(dtype)
    bg, tau = dtype.type(background), dtype.type(threshold)
    value = dtype.type(bg + tau)
    for _ in range(32):
        if native_exceedance(value, bg, tau, dtype):
            previous = np.nextafter(value, dtype.type(-np.inf))
            if not native_exceedance(previous, bg, tau, dtype):
                return (float(previous) + float(value)) / 2
            value = previous
        else:
            value = np.nextafter(value, dtype.type(np.inf))
        if not np.isfinite(value):
            break
    raise ValueError("threshold cannot be represented safely in native temperature dtype")


def summarize(values):
    values = np.asarray(values, dtype=float)
    sd = float(values.std(ddof=1))
    return {"n": len(values), "mean": float(values.mean()), "sd": sd,
            "mean_mcse": sd / np.sqrt(len(values)),
            "q05_q50_q95": np.quantile(values, [.05, .5, .95]).tolist()}


def assess_fields(fields, model, scores, train_indices, test_indices, candidate_scores, directory,
                  *, background, thresholds, cell_size_m, chunk_cells=4096, area_events=(),
                  progress=lambda message: None):
    """Native-grid comparisons: direct holdout, its KL reconstruction, Gaussian law.

    Area has units m2. Probability differences are probability points, not percent.
    Expected Gaussian area is integrated analytically; finite Gaussian draws give
    the complete area-distribution estimate and an MC precision diagnostic.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if (not len(thresholds) or not np.isfinite(thresholds).all() or not np.isfinite(background)
            or not np.isfinite(cell_size_m) or cell_size_m <= 0 or chunk_cells < 1):
        raise ValueError("finite thresholds/background, positive cell size/chunk required")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    flat = fields.reshape(len(fields), -1)
    heldout = scores[test_indices]
    law_mean, _, law_root = score_law(scores[train_indices])
    n_test, n_gaussian = len(test_indices), len(candidate_scores)
    area = {"direct": np.zeros((n_test, len(thresholds))),
            "reconstruction": np.zeros((n_test, len(thresholds))),
            "gaussian": np.zeros((n_gaussian, len(thresholds)))}
    probability = {name: np.zeros((len(thresholds), flat.shape[1]), dtype=np.float32)
                   for name in (*area, "gaussian_analytic")}
    mean_map = np.empty(flat.shape[1]); sd_map = np.empty(flat.shape[1])
    squared_error = centered_energy = below_direct = below_gaussian = 0.
    progress("Comparing held-out reconstructions and Gaussian risks on the native grid")
    for start in range(0, flat.shape[1], chunk_cells):
        stop = min(start + chunk_cells, flat.shape[1])
        direct = np.asarray(flat[test_indices, start:stop])
        reconstruction = model.block(heldout, start, stop)
        gaussian = model.block(candidate_scores, start, stop)
        squared_error += float(np.sum((direct - reconstruction) ** 2))
        centered_energy += float(np.sum((direct - model.mean[start:stop]) ** 2))
        mu = model.block(law_mean[None, :], start, stop)[0]
        sd = np.linalg.norm(law_root.T @ model.modes[:, start:stop], axis=0)
        mean_map[start:stop], sd_map[start:stop] = mu, sd
        below_direct += int((direct < fields.dtype.type(background)).sum())
        below_gaussian += int((gaussian.astype(fields.dtype) < fields.dtype.type(background)).sum())
        for t, threshold in enumerate(thresholds):
            for name, block in (("direct", direct), ("reconstruction", reconstruction), ("gaussian", gaussian)):
                event = native_exceedance(block, background, threshold, fields.dtype)
                area[name][:, t] += event.sum(axis=1) * cell_size_m ** 2
                probability[name][t, start:stop] = event.mean(axis=0)
            p = np.asarray(native_exceedance(mu, background, threshold, fields.dtype), dtype=float)
            active = sd > 0
            p[active] = stats.norm.sf((native_boundary(background, threshold, fields.dtype) - mu[active]) / sd[active])
            probability["gaussian_analytic"][t, start:stop] = p
    np.save(root / "gaussian_mean.npy", mean_map.reshape(fields.shape[1:]))
    np.save(root / "gaussian_sd.npy", sd_map.reshape(fields.shape[1:]))
    for name, values in probability.items():
        np.save(root / f"probability_{name}.npy", values.reshape((len(thresholds), *fields.shape[1:])))
    np.savez(root / "area_samples.npz", thresholds_C=thresholds, **{f"{name}_m2": x for name, x in area.items()})
    metrics = []
    for t, threshold in enumerate(thresholds):
        entry = {"threshold_C": float(threshold), "area_m2": {name: summarize(values[:, t]) for name, values in area.items()},
                 "gaussian_expected_area_analytic_m2": float(probability["gaussian_analytic"][t].sum(dtype=float) * cell_size_m ** 2)}
        for name in ("reconstruction", "gaussian"):
            p = probability["gaussian_analytic" if name == "gaussian" else name][t]
            difference = p - probability["direct"][t]
            entry[name] = {"area_wasserstein_m2": float(stats.wasserstein_distance(area["direct"][:, t], area[name][:, t])),
                           "probability_mae": float(np.mean(np.abs(difference))),
                           "probability_rmse": float(np.sqrt(np.mean(difference ** 2))),
                           "probability_max_abs_error": float(np.max(np.abs(difference)))}
        entry["area_events"] = []
        for tau, budget in area_events:
            if float(tau) != float(threshold):
                continue
            risk = {}
            for name, values in area.items():
                count = int((values[:, t] > budget).sum())
                ci = stats.binomtest(count, len(values)).proportion_ci(method="exact")
                risk[name] = {"probability": count / len(values), "count": count,
                              "ci95": [float(ci.low), float(ci.high)]}
            entry["area_events"].append({"criterion": "area_m2 > budget_m2", "budget_m2": budget, "risk": risk})
        metrics.append(entry)
    return {"thresholds": metrics, "heldout_temperature_rmse_C": float(np.sqrt(squared_error / (n_test * flat.shape[1]))),
            "heldout_relative_squared_reconstruction_error": float(squared_error / centered_energy) if centered_energy else None,
            "below_background_cell_draw_fraction": {"direct": below_direct / (n_test * flat.shape[1]),
                                                       "gaussian": below_gaussian / (n_gaussian * flat.shape[1])},
            "below_background_interpretation": "diagnostic only; no physical lower bound is asserted",
            "area_definition": "all native output cells with native-dtype T-background > threshold; no source connectivity",
            "scope": "fixed output crop, pressure/material/pumps and LGCNN weights; permeability-induced surrogate uncertainty only",
            "precision": "mean MCSE and exact binomial intervals condition on the fitted law; no fit uncertainty included",
            "adequacy": "not automatically accepted; compare errors with application-declared decision tolerances"}
