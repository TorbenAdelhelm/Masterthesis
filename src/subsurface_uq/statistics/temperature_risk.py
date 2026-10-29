"""Initial RQ2 metrics for independent temperature draws in one fixed design.

These use warming anomalies, physical cell areas and declared boundary rules.
Source-connected plume geometry is a separate analysis, not inferred here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from ..rq1.analysis import scalar_summary


def validate_output_grid(shape, receptors, roi):
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("temperature output must be a 2-D grid with at least two cells per axis")
    for r, c in receptors:
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            raise ValueError("receptor lies outside the temperature output grid")
    if roi is not None:
        r0, r1, c0, c1 = roi
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            raise ValueError("mean-anomaly ROI lies outside the temperature output grid")


def scalar_temperature_qois(temperature, *, receptors, roi, background):
    validate_output_grid(temperature.shape, receptors, roi)
    values = {f"receptor_{i+1:03d}": float(temperature[r,c])
              for i,(r,c) in enumerate(receptors)}
    if roi is not None:
        r0,r1,c0,c1 = roi
        values["mean_temperature_anomaly"] = float(np.mean(temperature[r0:r1,c0:c1], dtype=np.float64)-background)
    return values


def _exceedance(delta, threshold, comparison):
    # Match the output dtype at a nominal threshold boundary, as in the legacy
    # accumulator, while retaining an explicitly selectable > or >= convention.
    value = np.asarray(threshold, dtype=delta.dtype)
    return delta > value if comparison == ">" else delta >= value


def receptor_risk(values, *, background, threshold, comparison=">"):
    if comparison not in {">", ">="}:
        raise ValueError("threshold comparison must be > or >=")
    sample = np.asarray(values)
    if sample.ndim != 1 or not len(sample) or not np.all(np.isfinite(sample)):
        raise ValueError("receptor temperatures must be a nonempty finite vector")
    delta = sample - np.asarray(background, dtype=sample.dtype)
    exceeded = _exceedance(delta, threshold, comparison)
    excess = np.maximum(delta-np.asarray(threshold, dtype=sample.dtype), 0.)
    count = int(exceeded.sum())
    interval = binomtest(count, len(sample)).proportion_ci(confidence_level=.95, method="exact")
    return {"count": len(sample), "exceedance_count": count, "probability": count/len(sample),
            "probability_ci95": [float(interval.low), float(interval.high)],
            "ci_method": "Clopper-Pearson; iid conditional draws for this fixed design",
            "expected_excess_C": float(np.mean(excess, dtype=np.float64)),
            "conditional_excess_C": float(np.mean(excess[exceeded], dtype=np.float64)) if count else None}


def summarize_risk_store(store, output, *, thresholds, background, receptors,
                         cell_size_m, budgets, comparison=">", chunk_rows=16):
    """Write final probability/severity maps and nested scalar risk summaries.

    Temperature arrays are disk backed. At most one spatial block and one
    threshold's final maps are resident during the streaming reduction.
    """
    if comparison not in {">", ">="}:
        raise ValueError("threshold comparison must be > or >=")
    source = np.load(store, mmap_mode="r", allow_pickle=False)
    if source.ndim != 3 or chunk_rows < 1:
        raise ValueError("temperature store must be [samples,rows,columns] with positive chunk_rows")
    validate_output_grid(source.shape[1:], receptors, None)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    areas = np.zeros((len(source), len(thresholds)))
    receptor_values = np.array([source[:,r,c] for r,c in receptors], dtype=source.dtype).T
    if not receptors:
        receptor_values = np.empty((len(source), 0), dtype=source.dtype)
    entries = []
    for t, threshold in enumerate(thresholds):
        probability = np.empty(source.shape[1:], dtype=np.float32)
        severity = np.empty_like(probability)
        for start in range(0, source.shape[1], chunk_rows):
            stop = min(start+chunk_rows, source.shape[1])
            block = np.asarray(source[:,start:stop])
            if not np.all(np.isfinite(block)):
                raise ValueError("temperature store contains nonfinite values")
            delta = block-np.asarray(background, dtype=block.dtype)
            exceeded = _exceedance(delta, threshold, comparison)
            probability[start:stop] = exceeded.mean(axis=0)
            severity[start:stop] = np.maximum(delta-np.asarray(threshold, dtype=delta.dtype),0).mean(axis=0,dtype=np.float64)
            areas[:,t] += exceeded.sum(axis=(1,2))*cell_size_m**2
        probability_path, severity_path = f"exceedance_probability_{t:02d}.npy", f"expected_excess_{t:02d}.npy"
        np.save(root/probability_path, probability)
        np.save(root/severity_path, severity)
        checkpoints = []
        for n in budgets:
            checkpoints.append({"n_samples": n, "area_m2": scalar_summary(areas[:n,t]),
                "receptors": {f"receptor_{i+1:03d}": receptor_risk(receptor_values[:n,i],
                    background=background, threshold=threshold, comparison=comparison)
                    for i in range(len(receptors))}})
        entries.append({"threshold_C": threshold, "probability_map": probability_path,
                        "expected_excess_map": severity_path, "checkpoints": checkpoints})
    np.savez(root/"risk_samples.npz", area_m2=areas, receptor_temperature_C=receptor_values,
             thresholds_C=np.asarray(thresholds))
    return {"anomaly": "T-background; warming only", "comparison": comparison,
            "cell_area_m2": cell_size_m**2, "thresholds": entries,
            "geometry": "all exceeding output cells; no source-connectivity or plume-length claim",
            "domain": "fixed valid temperature output crop; area outside crop unobserved"}
