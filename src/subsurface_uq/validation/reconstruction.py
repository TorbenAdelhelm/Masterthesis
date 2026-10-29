"""Spatially withheld diagnostics for an exact-reading reconstruction law.

Selection accepts only revealed observations. Test values enter scoring after
selection. Correlated test pixels are never counted as independent replicates.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.special import ndtr, ndtri

from ..sampling.covariance_reconstruction import OrdinaryKrigingSystem
from ..sampling.measurement_density import gaussian_crps, observation_condition_number


def spatial_test_partition(shape, *, block_cells, fraction, buffer_cells, seed):
    """Choose complete test blocks from geometry before any readings are seen."""
    if (len(shape) != 2 or min(shape) < 3 or isinstance(block_cells, bool)
            or not isinstance(block_cells, int) or block_cells < 1
            or isinstance(buffer_cells, bool) or not isinstance(buffer_cells, int) or buffer_cells < 0
            or not np.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("valid grid, positive block size, nonnegative buffer and fraction in (0,1) required")
    rows, cols = np.indices(shape)
    ncols = (shape[1]+block_cells-1)//block_cells
    blocks = (rows//block_cells)*ncols+cols//block_cells
    available = np.unique(blocks)
    count = max(1, int(np.ceil(fraction*len(available))))
    if count >= len(available):
        raise ValueError("test partition must leave training blocks")
    chosen = np.sort(np.random.default_rng(seed).choice(available, count, replace=False))
    test = np.isin(blocks, chosen)
    excluded = (maximum_filter(test, size=2*buffer_cells+1, mode="constant", cval=0)
                if buffer_cells else test.copy())
    if np.all(excluded):
        raise ValueError("test buffer leaves no observation region")
    return test, excluded, blocks, {"test_block_ids": chosen.tolist(),
        "block_cells": block_cells, "buffer_cells": buffer_cells, "seed": seed,
        "test_fraction_requested": fraction, "test_fraction_actual": float(test.mean()),
        "reference_values_used": False}


def block_folds(indices, block_cells, n_folds, seed):
    """Keep observation blocks together, balancing their observation counts."""
    keys, inverse, counts = np.unique(np.asarray(indices)//block_cells, axis=0,
                                     return_inverse=True, return_counts=True)
    folds = min(n_folds, len(keys))
    if folds < 2:
        raise ValueError("selection requires observations in at least two spatial blocks")
    order = np.random.default_rng(seed).permutation(len(keys))
    order = sorted(order, key=lambda k: -counts[k])
    sizes = np.zeros(folds, dtype=int)
    assignment = np.empty(len(keys), dtype=int)
    for k in order:
        f = int(np.argmin(sizes))
        assignment[k] = f
        sizes[f] += counts[k]
    return assignment[inverse], inverse


def predict_points(shape, spec, indices, readings, queries, *, cell_size_m, chunk_size=2048):
    condition = observation_condition_number(spec, indices, cell_size_m)
    system = OrdinaryKrigingSystem(shape, spec, indices, readings, cell_size_m=cell_size_m)
    means, variances = [], []
    for start in range(0, len(queries), chunk_size):
        m, v = system.moments(system.features(queries[start:start+chunk_size]))
        means.append(m)
        variances.append(v)
    return np.concatenate(means), np.sqrt(np.concatenate(variances))*spec.sigma_R, condition


def select_covariance(shape, indices, readings, candidates, *, cell_size_m,
                      block_cells, n_folds, seed):
    """Select from declared candidates by equal-block CV CRPS of revealed data.

    No reference array or future-budget reading is accepted by this function.
    The selected covariance is frozen for the design's nested larger budgets.
    """
    indices, readings = np.asarray(indices), np.asarray(readings, dtype=float)
    if readings.shape != (len(indices),) or not np.all(np.isfinite(readings)):
        raise ValueError("one finite reading per observation required")
    folds, blocks = block_folds(indices, block_cells, n_folds, seed)
    # The mean and unit variance do not depend on amplitude. Cache only within
    # this observation set, so no future readings can influence selection.
    predictions = {}
    ranked, rejected = [], []
    for spec in candidates:
        key = (spec.family, spec.ell_row_m, spec.ell_col_m, spec.angle_deg)
        try:
            if key not in predictions:
                unit = replace(spec, sigma_R=1.)
                observation_condition_number(unit, indices, cell_size_m)
                mean, std = np.empty(len(indices)), np.empty(len(indices))
                for fold in np.unique(folds):
                    train, valid = folds != fold, folds == fold
                    mean[valid], std[valid], _ = predict_points(shape, unit, indices[train],
                        readings[train], indices[valid], cell_size_m=cell_size_m)
                predictions[key] = mean, std
            mean, unit_std = predictions[key]
            scores = gaussian_crps(mean, unit_std*spec.sigma_R, readings)
            score = float(np.mean([scores[blocks == b].mean() for b in np.unique(blocks)]))
            ranked.append({"covariance": asdict(spec), "block_mean_crps_log10": score})
        except (ValueError, np.linalg.LinAlgError, ArithmeticError) as exc:
            rejected.append({"covariance": asdict(spec), "reason": str(exc)})
    ranked.sort(key=lambda r: (r["block_mean_crps_log10"], tuple(r["covariance"].values())))
    if not ranked:
        raise ValueError("no candidate supports the revealed-observation spatial folds")
    return {"selected": ranked[0], "ranking": ranked, "unavailable": rejected,
        "selection_measurements": len(indices), "fold_assignment": folds.tolist(),
        "selection_blocks": len(np.unique(blocks)), "effective_folds": len(np.unique(folds)),
        "criterion": "mean across spatial blocks of observation CV CRPS",
        "test_values_used": False, "future_readings_used": False,
        "parameter_uncertainty": "candidate selection uncertainty is not included in conditional variance"}


def predictive_scores(truth, mean, std):
    truth, mean, std = np.broadcast_arrays(np.asarray(truth, dtype=float),
        np.asarray(mean, dtype=float), np.asarray(std, dtype=float))
    if truth.ndim != 1 or not len(truth) or not np.all(np.isfinite([truth, mean, std])) or np.any(std < 0):
        raise ValueError("finite nonempty predictive vectors and nonnegative SD required")
    error = truth-mean
    positive = std > 0
    z = np.divide(error, std, out=np.zeros_like(error), where=positive)
    levels = [.5, .8, .9, .95, .99]
    result = {"n_points": len(truth), "rmse_log10": float(np.sqrt(np.mean(error**2))),
        "bias_log10": float(error.mean()), "rms_predictive_sd_log10": float(np.sqrt(np.mean(std**2))),
        "mean_crps_log10": float(gaussian_crps(mean, std, truth).mean()),
        "zero_predictive_sd_count": int((~positive).sum()),
        "reliability": {str(p): float(np.mean(np.abs(error) <= ndtri((1+p)/2)*std)) for p in levels}}
    result.update(standardized_rmse=float(np.sqrt(np.mean(z**2))) if np.all(positive) else None,
        standardized_bias=float(z.mean()) if np.all(positive) else None,
        mean_gaussian_nlpd=float(np.mean(.5*z**2+np.log(std)+.5*np.log(2*np.pi))) if np.all(positive) else None,
        squared_error_to_variance_ratio=float(np.mean(error**2)/np.mean(std**2)) if np.any(positive) else None,
        pit_mean=float(ndtr(z).mean()) if np.all(positive) else None)
    return result


def withheld_diagnostics(truth, mean, std, block_ids, *, rules):
    """Descriptive spatial scores and predeclared exploratory adequacy checks."""
    truth, mean, std, block_ids = map(np.asarray, (truth, mean, std, block_ids))
    if block_ids.shape != truth.shape:
        raise ValueError("one spatial block identifier per withheld point required")
    score = predictive_scores(truth, mean, std)
    blocks = [{"block_id": int(b), **predictive_scores(truth[block_ids == b],
               mean[block_ids == b], std[block_ids == b])} for b in np.unique(block_ids)]
    # Equal block weights keep cropped edge blocks from disappearing in the
    # overall assessment. They are descriptive units, not independent trials.
    coverage = float(np.mean([b["reliability"]["0.9"] for b in blocks]))
    zrmse = score["standardized_rmse"]
    bias = score["standardized_bias"]
    checks = {"enough_test_blocks": len(blocks) >= rules["minimum_test_blocks"],
        "coverage90": rules["coverage90_min"] <= coverage <= rules["coverage90_max"],
        "standardized_rmse": zrmse is not None and rules["standardized_rmse_min"] <= zrmse <= rules["standardized_rmse_max"],
        "standardized_bias": bias is not None and abs(bias) <= rules["absolute_standardized_bias_max"]}
    boundaries = np.unique(np.quantile(std, np.linspace(0, 1, 6)))
    labels = np.digitize(std, boundaries[1:-1], right=True)
    bins = [{"bin": int(b), **predictive_scores(truth[labels == b], mean[labels == b], std[labels == b])}
            for b in np.unique(labels)]
    return {"aggregate": score, "equal_block_coverage90": coverage,
        "blocks": blocks, "uncertainty_bins": bins, "checks": checks,
        "adequate_for_exploratory_propagation": all(checks.values()), "rules": rules,
        "interpretation": "predeclared descriptive checks on withheld spatial blocks; no iid-pixel confidence intervals or geological calibration certificate"}
