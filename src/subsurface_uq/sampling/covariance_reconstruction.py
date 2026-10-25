"""Covariance-only observation designs and unknown-constant-mean conditioning.

The design API accepts geometry and covariance, never a reference array. The
ordinary-kriging API receives only selected readings. A flat prior for the
constant mean includes its posterior uncertainty; zero readings are improper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from .gstools_reference import GSToolsReferenceSpec
from .reference_conditioning import _indices, _positive_integer


@dataclass
class CovarianceDesign:
    indices: np.ndarray
    cholesky: np.ndarray
    probabilities: np.ndarray
    preselection_variance: np.ndarray
    metadata: dict


def covariance_locations(shape, covariance, *, cell_size_m, n_measurements,
                         candidate_stride_cells=1, seed=0, pivot_floor=1.e-7):
    """Draw nested locations proportional to ordinary-kriging variance.

    The first location is uniform. Thereafter the unknown-mean contribution is
    included. Rank-one updates use unit correlation, so amplitude and reference
    values cannot affect placement. Almost redundant candidates are ineligible
    under an explicit numerical innovation-variance floor, without a nugget.
    """
    if len(shape) != 2 or any(_positive_integer(n, "axis") < 3 for n in shape):
        raise ValueError("shape must have two axes of at least three cells")
    if not isinstance(covariance, GSToolsReferenceSpec):
        raise ValueError("GSToolsReferenceSpec required")
    if not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if not np.isfinite(pivot_floor) or not 0 < pivot_floor < 1:
        raise ValueError("pivot_floor must lie in (0,1)")
    count = _positive_integer(n_measurements, "n_measurements")
    stride = _positive_integer(candidate_stride_cells, "candidate_stride_cells")
    seed = _positive_integer(seed, "seed", allow_zero=True)
    rr, cc = np.meshgrid(np.arange(0, shape[0], stride), np.arange(0, shape[1], stride), indexing="ij")
    candidates = np.column_stack((rr.ravel(), cc.ravel()))
    count = min(count, len(candidates))
    features = np.zeros((len(candidates), count))
    variance = np.ones(len(candidates))
    trend_residual = np.ones(len(candidates))
    trend_precision = 0.
    h = np.zeros(count)
    selected, probabilities, variances = [], [], []
    rng = np.random.default_rng(seed)
    stop = "maximum_measurements"
    for k in range(count):
        available = variance > pivot_floor
        if selected:
            available[selected] = False
        if not available.any():
            stop = "numerical_innovation_variance_floor"
            break
        score = variance.copy() if k == 0 else variance + trend_residual**2 / trend_precision
        score = np.where(available, np.maximum(score, 0.), 0.)
        probability = score / score.sum()
        chosen = int(rng.choice(len(candidates), p=probability))
        selected.append(chosen)
        probabilities.append(probability[chosen])
        variances.append(score[chosen])
        offsets = (candidates-candidates[chosen])*cell_size_m
        cross = covariance.correlation(offsets[:, 0], offsets[:, 1])
        if k:
            cross = cross-features[:, :k]@features[chosen, :k]
        pivot = np.sqrt(variance[chosen])
        column = cross/pivot
        h[k] = trend_residual[chosen]/pivot
        # Exact zeros/diagonal remove accumulation at already selected cells.
        column[np.asarray(selected[:-1], dtype=int)] = 0.
        column[chosen] = pivot
        features[:, k] = column
        trend_residual -= column*h[k]
        trend_precision += h[k]**2
        variance -= column**2
        if variance.min(initial=0) < -1.e-8:
            raise ArithmeticError("design variance update lost positive semidefiniteness")
        variance = np.maximum(variance, 0.)
    n = len(selected)
    chol = np.tril(features[np.asarray(selected), :n])
    return CovarianceDesign(candidates[selected], chol, np.asarray(probabilities), np.asarray(variances),
        {"policy": "random draw proportional to ordinary-kriging variance; first uniform",
         "reference_values_used": False, "reference_mean_used": False,
         "amplitude_affects_locations": False, "candidate_count": len(candidates),
         "candidate_stride_cells": stride, "seed": seed, "requested_measurements": n_measurements,
         "actual_measurements": n, "stop_reason": stop, "normalized_pivot_floor": pivot_floor,
         "measurement_error_variance": 0., "added_nugget": None})


class OrdinaryKrigingSystem:
    """Condition on exact readings; estimate the mean from these readings only.

    Correlation is factored independently of amplitude. Unit posterior covariance
    is C-C_xB C_BB^-1 C_Bx + r(x)r(x')/(1' C_BB^-1 1).
    A realization is u + W(d-u_B), with ordinary-kriging weights W summing to one.
    This retains the uncertainty of the unknown constant mean.
    """

    def __init__(self, shape, covariance, indices, readings, *, cell_size_m, cholesky=None):
        self.shape, self.covariance, self.cell_size_m = tuple(shape), covariance, float(cell_size_m)
        self.indices = _indices(indices, self.shape)
        self.readings = np.asarray(readings, dtype=np.float64)
        if not len(self.indices):
            raise ValueError("unknown-mean ordinary kriging needs at least one reading")
        if len(np.unique(self.indices, axis=0)) != len(self.indices):
            raise ValueError("duplicate observation cells")
        if self.readings.shape != (len(self.indices),) or not np.all(np.isfinite(self.readings)):
            raise ValueError("one finite log-reading per observation required")
        offsets = (self.indices[:, None]-self.indices[None, :])*cell_size_m
        correlation = covariance.correlation(offsets[..., 0], offsets[..., 1])
        self.cholesky = np.linalg.cholesky(correlation) if cholesky is None else np.asarray(cholesky)
        np.testing.assert_allclose(self.cholesky@self.cholesky.T, correlation, atol=2.e-8, rtol=1.e-10)
        self.h = solve_triangular(self.cholesky, np.ones(len(self.indices)), lower=True)
        self.white_readings = solve_triangular(self.cholesky, self.readings, lower=True)
        self.precision = float(self.h@self.h)
        self.estimated_mean = float(self.h@self.white_readings/self.precision)

    def features(self, queries):
        queries = _indices(queries, self.shape)
        offsets = (queries[:, None]-self.indices[None, :])*self.cell_size_m
        cross = self.covariance.correlation(offsets[..., 0], offsets[..., 1])
        return solve_triangular(self.cholesky, cross.T, lower=True).T

    def moments(self, features):
        r = 1.-features@self.h
        mean = features@self.white_readings+r*self.estimated_mean
        variance = 1.-np.sum(features**2, axis=1)+r**2/self.precision
        if variance.min(initial=0) < -1.e-7:
            raise ArithmeticError("negative ordinary-kriging variance")
        return mean, np.maximum(variance, 0.)

    def unit_draws(self, features, prior_values, prior_at_data):
        """Return mean and zero-mean unit-amplitude conditional realizations."""
        white_prior = solve_triangular(self.cholesky, prior_at_data, lower=True)
        r = 1.-features@self.h
        mean, variance = self.moments(features)
        residual = prior_values-features@white_prior-r[:, None]*(self.h@white_prior/self.precision)
        return mean, variance, residual
