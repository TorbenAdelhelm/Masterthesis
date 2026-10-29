"""Scalar regional temperature losses for reference-relative RQ1/RQ2 risk."""
from __future__ import annotations

import numpy as np


def empirical_tail_risk(values, probability=.95):
    """Empirical VaR and expected shortfall, including atoms at the quantile."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or not 0<probability<1:
        raise ValueError("finite scalar sample vector and probability in (0,1) required")
    q=float(np.quantile(values,probability,method="inverted_cdf"))
    return {"probability":probability,"VaR":q,
        "expected_shortfall":float(q+np.maximum(values-q,0.).mean()/(1-probability)),
        "method":"empirical inverse CDF; atom-aware integrated upper-tail mean"}


def regional_deviation_samples(temperatures, reference, roi):
    """Per-draw maximum, RMS and signed mean differences on a fixed output ROI.

    A quantile of the per-draw maximum is simultaneous over evaluated ROI cells.
    It is not a maximum of pointwise quantiles or a continuum-space guarantee.
    """
    if temperatures.ndim!=3 or tuple(temperatures.shape[1:])!=tuple(reference.shape):
        raise ValueError("temperature ensemble and reference output grids must agree")
    if roi is None:
        raise ValueError("explicit regional output-grid ROI required")
    r0,r1,c0,c1=roi
    if not (0<=r0<r1<=reference.shape[0] and 0<=c0<c1<=reference.shape[1]):
        raise ValueError("ROI outside valid output crop")
    rows=[]
    for field in temperatures:
        delta=np.asarray(field[r0:r1,c0:c1],dtype=float)-reference[r0:r1,c0:c1]
        if not np.all(np.isfinite(delta)):
            raise ValueError("nonfinite regional temperature difference")
        rows.append([np.max(np.abs(delta)),np.sqrt(np.mean(delta**2)),delta.mean()])
    return {name:np.asarray(rows)[:,i] for i,name in enumerate(("maximum_absolute_deviation_C","rms_deviation_C","mean_deviation_C"))}
