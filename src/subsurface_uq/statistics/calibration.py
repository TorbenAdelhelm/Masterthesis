"""Finite-ensemble calibration for independent hidden-truth trials.

Ranks include randomized ties: discretized temperatures, threshold areas and
constant outputs can have atoms. Spatial pixels are never treated as iid trials.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def ensemble_truth_scores(truth, samples, *, rng, level=.9):
    sample = np.asarray(samples, dtype=float)
    if (sample.ndim != 1 or len(sample) < 2 or not np.isfinite(truth)
            or not np.all(np.isfinite(sample)) or not 0 < level < 1):
        raise ValueError("finite truth, >=2 finite draws and level in (0,1) required")
    ordered = np.sort(sample)
    n = len(sample)
    rank = int(np.sum(sample < truth) + rng.integers(0, int(np.sum(sample == truth))+1))
    lower = int(np.ceil((1-level)/2*(n+1)))
    upper = int(np.floor((1+level)/2*(n+1))) - 1
    pair_term = np.dot(2*np.arange(1,n+1)-n-1, ordered)/n**2
    return {"truth": float(truth), "n_samples": n, "mean": float(sample.mean()),
            "sd": float(sample.std(ddof=1)), "mean_error": float(sample.mean()-truth),
            "crps": float(np.mean(np.abs(sample-truth))-pair_term),
            "rank": rank, "randomized_pit": float((rank+rng.random())/(n+1)),
            "rank_covered": lower <= rank <= upper,
            "rank_interval": [lower, upper], "nominal_level": level,
            "attainable_rank_coverage": (upper-lower+1)/(n+1),
            "q05": float(np.quantile(sample,.05)), "q95": float(np.quantile(sample,.95)),
            "rank_interpretation": "truth rank among n iid conditional draws, with uniform tie breaking"}


def calibration_summary(scores):
    """Summarize one scalar, scenario, layout and count across iid truths only."""
    if not scores or len({s["n_samples"] for s in scores}) != 1:
        raise ValueError("one fixed ensemble budget and nonempty independent-trial records required")
    targets = {s["attainable_rank_coverage"] for s in scores}
    if len(targets) != 1:
        raise ValueError("rank coverage targets must agree")
    covered = sum(s["rank_covered"] for s in scores)
    interval = binomtest(covered,len(scores)).proportion_ci(method="exact")
    return {"n_independent_truths": len(scores), "n_samples": scores[0]["n_samples"],
            "rank_histogram": np.bincount([s["rank"] for s in scores],minlength=scores[0]["n_samples"]+1).tolist(),
            "attainable_rank_coverage": targets.pop(), "rank_coverage": covered/len(scores),
            "rank_coverage_ci95": [float(interval.low),float(interval.high)],
            "mean_crps": float(np.mean([s["crps"] for s in scores])),
            "mean_error_rmse": float(np.sqrt(np.mean([s["mean_error"]**2 for s in scores]))),
            "rms_predictive_sd": float(np.sqrt(np.mean([s["sd"]**2 for s in scores]))),
            "mean_predictive_variance": float(np.mean([s["sd"]**2 for s in scores])),
            "ci_unit": "independent hidden-truth trial at this fixed scalar/layout/count; no pixel or layout pooling",
            "interpretation": "model-conditional consistency diagnostic; not empirical geological or simulator validation"}
