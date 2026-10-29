"""Known-law Gaussian conditioning without access to a hidden truth field.

Y = mean_log10 + Z, Z ~ N(0, sigma_R**2 * correlation). The mean and
covariance are declared experimental assumptions, not fitted site properties.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

from .gstools_reference import GSToolsReferenceSpec
from .reference_conditioning import _indices


class KnownMeanConditioner:
    """Simple Gaussian kriging with exact observations and a proper prior.

    Only geometry and revealed log10 readings enter this API. Zero readings
    recover the prior. Unit residuals must come from the same correlation law.
    Unlike ordinary kriging, this model assumes a KNOWN mean, also used when
    generating the independent hidden truths.
    """

    def __init__(self, shape, covariance, indices, *, mean_log10, cell_size_m):
        if not isinstance(covariance, GSToolsReferenceSpec) or covariance.center != "median":
            raise ValueError("median-centered GSToolsReferenceSpec required")
        if (len(shape) != 2 or any(isinstance(v, bool) or int(v) != v or v < 3 for v in shape)
                or not np.isfinite(mean_log10) or not np.isfinite(cell_size_m) or cell_size_m <= 0):
            raise ValueError("finite mean, positive spacing and 2-D grid required")
        self.shape = tuple(map(int, shape))
        self.spec, self.mean_log10, self.cell_size_m = covariance, float(mean_log10), float(cell_size_m)
        self.indices = _indices(indices, self.shape).copy()
        if len(np.unique(self.indices, axis=0)) != len(self.indices):
            raise ValueError("duplicate observation cells")
        offsets = (self.indices[:, None] - self.indices[None, :]) * cell_size_m
        correlation = covariance.correlation(offsets[..., 0], offsets[..., 1])
        if len(self.indices):
            eigenvalues = np.linalg.eigvalsh(correlation)
            if eigenvalues[0] <= eigenvalues[-1] * 1.e-12:
                raise ValueError("exact observation covariance loses numerical rank")
            self.condition_number = float(eigenvalues[-1] / eigenvalues[0])
            self.cholesky = np.linalg.cholesky(correlation)
        else:
            self.condition_number = None
            self.cholesky = np.empty((0, 0))

    def features(self, queries):
        queries = _indices(queries, self.shape)
        if not len(self.indices):
            return np.empty((len(queries), 0))
        offsets = (queries[:, None] - self.indices[None, :]) * self.cell_size_m
        cross = self.spec.correlation(offsets[..., 0], offsets[..., 1])
        return solve_triangular(self.cholesky, cross.T, lower=True).T

    def _readings(self, readings, count):
        values = np.asarray(readings, dtype=float)
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or not 0 <= count <= len(self.indices) or values.shape != (count,)
                or not np.all(np.isfinite(values))):
            raise ValueError("one finite log10 reading per selected prefix cell required")
        return values

    def moments(self, features, readings, *, count=None):
        count = len(self.indices) if count is None else count
        readings = self._readings(readings, count)
        f = np.asarray(features, dtype=float)[:, :count]
        white = solve_triangular(self.cholesky[:count, :count], readings-self.mean_log10, lower=True)
        variance = 1. - np.sum(f*f, axis=1)
        if variance.min(initial=0.) < -1.e-8:
            raise ArithmeticError("negative conditional variance")
        return self.mean_log10 + f@white, self.spec.sigma_R**2*np.maximum(variance, 0.)

    def update_coefficients(self, readings, prior_at_data, *, count):
        readings = self._readings(readings, count)
        prior_at_data = np.asarray(prior_at_data, dtype=float)
        if prior_at_data.shape != (count,) or not np.all(np.isfinite(prior_at_data)):
            raise ValueError("matching finite unit-Gaussian prior readings required")
        innovation = readings - self.mean_log10 - self.spec.sigma_R*prior_at_data
        return solve_triangular(self.cholesky[:count, :count], innovation, lower=True)

    def conditional_log_field(self, prior_unit, readings, *, count=None, features=None, chunk_size=8192):
        """Matheron update, chunked on the native grid; no nugget or filtering."""
        count = len(self.indices) if count is None else count
        prior = np.asarray(prior_unit, dtype=float)
        if prior.shape != self.shape or not np.all(np.isfinite(prior)) or chunk_size < 1:
            raise ValueError("finite matching prior grid and positive chunk_size required")
        coefficient = self.update_coefficients(readings, prior[tuple(self.indices[:count].T)], count=count)
        result = self.mean_log10 + self.spec.sigma_R*prior
        if count:
            for start in range(0, prior.size, chunk_size):
                stop = min(start+chunk_size, prior.size)
                f = (features[start:stop, :count] if features is not None else
                     self.features(np.column_stack(np.unravel_index(np.arange(start, stop), self.shape)))[:, :count])
                result.ravel()[start:stop] += f@coefficient
        return result


def physical_permeability(logs):
    """Convert to native float32 m²; fail rather than clip or replace a draw."""
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        values = np.power(10., np.asarray(logs, dtype=float)).astype(np.float32)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ArithmeticError("generated permeability outside positive finite float32 support")
    return values
