"""Reference-centered radial laws fitted with GSTools, sampled on a native grid.

Circulant embedding avoids random-frequency and KL truncation. A smooth extension
is applied ONLY outside the rectangle of offsets occurring in the requested
field. The embedding must be positive semidefinite up to a recorded numerical
covariance error, otherwise it is enlarged. Generated draws are never filtered.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math

import numpy as np
from scipy import fft
from scipy.special import betainc


@dataclass(frozen=True)
class GSToolsReferenceSpec:
    family: str
    ell_row_m: float
    ell_col_m: float
    sigma_R: float = .05
    center: str = "median"

    def __post_init__(self):
        if self.family not in {"gaussian", "matern32_radial", "exponential_radial"}:
            raise ValueError("unsupported radial covariance family")
        if self.center not in {"median", "arithmetic-mean"}:
            raise ValueError("unknown centering convention")
        if any(not math.isfinite(x) or x <= 0 for x in
               (self.ell_row_m, self.ell_col_m, self.sigma_R)):
            raise ValueError("lengths and sigma_R must be positive finite numbers")

    def gstools_model(self):
        try:
            import gstools as gs
        except ImportError as exc:
            raise ImportError("GSTools covariance verification requires subsurface-uq[geostat]") from exc
        args = dict(dim=2, var=1., nugget=0., angles=0.,
                    len_scale=[self.ell_col_m, self.ell_row_m])
        if self.family == "gaussian":
            return gs.Gaussian(**args, rescale=1 / np.sqrt(2.))
        if self.family == "matern32_radial":
            return gs.Matern(**args, nu=1.5, rescale=np.sqrt(2.))
        return gs.Exponential(**args, rescale=1.)

    def correlation(self, row_offset_m, col_offset_m):
        radius = np.hypot(np.asarray(row_offset_m) / self.ell_row_m,
                          np.asarray(col_offset_m) / self.ell_col_m)
        if self.family == "gaussian":
            return np.exp(-.5 * radius**2)
        if self.family == "matern32_radial":
            radius *= np.sqrt(3.)
            return (1 + radius) * np.exp(-radius)
        return np.exp(-radius)

    def metadata(self):
        return {**asdict(self), "sigma_R_units": "log10_permeability",
                "covariance_geometry": "radial anisotropic; zero rotation",
                "gstools_model": str(self.gstools_model()),
                "coordinate_order": "array rows/columns; GSTools x=columns,y=rows",
                "marginal": "pointwise untruncated lognormal around the heterogeneous reference",
                "assumption_status": "reference-texture-guided residual correlation, not calibrated uncertainty"}


class GSToolsReferenceFFTSampler:
    """A fixed linear Gaussian map of a validated periodic embedding, then crop.

    The requested grid is not made periodic: every covariance lag that occurs
    within the crop equals the GSTools radial law, up to the measured roundoff
    correction. Numerical failure enlarges the embedding, never rejects draws.
    """

    def __init__(self, spec, shape, *, cell_size_m=5., workers=8,
                 covariance_tolerance=1.e-11, max_embedding_points=268435456):
        if not isinstance(spec, GSToolsReferenceSpec):
            raise ValueError("GSToolsReferenceSpec required")
        if (len(shape) != 2 or any(isinstance(n, bool) or int(n) != n or n < 3 for n in shape)
                or not math.isfinite(cell_size_m) or cell_size_m <= 0
                or not isinstance(workers, int) or workers < 1
                or not math.isfinite(covariance_tolerance) or not 0 < covariance_tolerance <= 1.e-8):
            raise ValueError("positive native grid, worker count and small numerical tolerance required")
        self.spec, self.shape = spec, tuple(map(int, shape))
        self.cell_size_m, self.workers = float(cell_size_m), workers
        self.covariance_tolerance = covariance_tolerance
        # Check analytic closed forms against GSTools, including off-axis lags.
        offsets = np.array([[0., 0.], [5., 0.], [0., 5.], [5., 5.],
                            [spec.ell_row_m, 0.], [0., spec.ell_col_m],
                            [spec.ell_row_m, spec.ell_col_m]])
        expected = spec.gstools_model().cov_spatial(offsets[:, ::-1].T)
        np.testing.assert_allclose(spec.correlation(*offsets.T), expected, rtol=1.e-12, atol=2.e-15)
        embedding = tuple(2 ** math.ceil(math.log2(3 * n)) for n in self.shape)
        attempts = []
        while math.prod(embedding) <= max_embedding_points:
            spectrum, record = self._embedding(embedding)
            attempts.append(record)
            if record["negative_covariance_mass"] <= covariance_tolerance:
                break
            del spectrum
            gc.collect()
            # Grow the relatively least-extended correlation direction first.
            axis = int(np.argmax(np.array([spec.ell_row_m, spec.ell_col_m]) / np.array(embedding)))
            embedding = tuple(n * (2 if i == axis else 1) for i, n in enumerate(embedding))
        else:
            raise ValueError(f"no PSD embedding within memory bound; attempts={attempts}")
        np.maximum(spectrum, 0., out=spectrum)
        represented = fft.irfft2(spectrum, s=embedding, workers=workers)
        checks = [(0, 0), (1, 0), (0, 1), (1, 1),
                  (self.shape[0] - 1, 0), (0, self.shape[1] - 1),
                  (self.shape[0] - 1, self.shape[1] - 1)]
        actual = np.array([represented[r, c] for r, c in checks])
        exact = spec.correlation(np.array(checks)[:, 0] * self.cell_size_m,
                                 np.array(checks)[:, 1] * self.cell_size_m)
        error = float(np.max(np.abs(actual - exact)))
        if error > covariance_tolerance * 1.1 + 1.e-14:
            raise ValueError("embedding failed its requested-lag covariance check")
        self.represented_unit_variance = float(represented[0, 0])
        increment_errors = {}
        for name, index in (("row", 1), ("column", 2)):
            target = 2 * (1 - exact[index])
            increment_errors[name] = float(abs(2 * (actual[0] - actual[index]) - target) / target)
        del represented
        self.embedding_shape = embedding
        np.sqrt(spectrum, out=spectrum)
        self.sqrt_spectrum = spectrum
        self.metadata = {"backend": "GSTools-matched covariance / FFT circulant embedding",
            "spec": spec.metadata(), "native_shape": list(self.shape), "cell_size_m": self.cell_size_m,
            "embedding_shape": list(embedding), "embedding_attempts": attempts,
            "extension": "C4 flat-crop window outside all requested row/column covariance lags only",
            "covariance_tolerance": covariance_tolerance, "requested_lag_max_absolute_error": error,
            "native_increment_relative_representation_errors": increment_errors,
            "represented_unit_variance": self.represented_unit_variance,
            "numerical_negative_eigenvalue_correction_only": True,
            "KL_truncation": None, "random_frequency_truncation": None,
            "sampling": "MC; iid real Gaussian spectral degrees of freedom; deterministic seed map",
            "sample_filtering": None, "workers": workers}

    def _embedding(self, embedding):
        m, n = embedding
        offsets = [np.minimum(np.arange(k), k - np.arange(k)) * self.cell_size_m for k in embedding]
        limits = (np.array(self.shape) - 1) * self.cell_size_m
        tapers = [1 - betainc(5, 5, np.clip((offset - limit) / (k * self.cell_size_m / 2 - limit), 0, 1))
                  for offset, limit, k in zip(offsets, limits, embedding)]
        cov = np.empty(embedding, dtype=np.float64)
        for start in range(0, m, 64):
            stop = min(m, start + 64)
            values = self.spec.correlation(offsets[0][start:stop, None], offsets[1][None, :])
            values *= tapers[0][start:stop, None]
            values *= tapers[1][None, :]
            cov[start:stop] = values
        transformed = fft.rfft2(cov, workers=self.workers, overwrite_x=True)
        del cov, values
        spectrum = transformed.real.copy()
        del transformed
        weights = np.ones(n // 2 + 1)
        weights[1:-1] = 2
        negative_mass = 0.
        for start in range(0, m, 64):
            negative_mass -= float(np.sum(np.minimum(spectrum[start:start + 64], 0) * weights)) / (m * n)
        return spectrum, {"shape": list(embedding), "minimum_eigenvalue": float(spectrum.min()),
                          "maximum_eigenvalue": float(spectrum.max()),
                          "negative_covariance_mass": negative_mass}

    def sample_residual(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        m, n = self.embedding_shape
        z = np.empty((m, n // 2 + 1), dtype=np.complex128)
        scale = np.sqrt(m * n / 2.)
        for start in range(0, m, 64):
            stop = min(m, start + 64)
            z[start:stop].real = rng.standard_normal((stop - start, z.shape[1])) * scale
            z[start:stop].imag = rng.standard_normal((stop - start, z.shape[1])) * scale
            z[start:stop] *= self.sqrt_spectrum[start:stop]
        # The real-FFT boundary columns require Hermitian row symmetry.
        for column in (0, n // 2):
            boundary = fft.fft(rng.standard_normal(m), workers=self.workers) * np.sqrt(n)
            z[:, column] = boundary * self.sqrt_spectrum[:, column]
        field = fft.irfft2(z, s=self.embedding_shape, workers=self.workers, overwrite_x=True)
        del z
        residual = np.ascontiguousarray(field[:self.shape[0], :self.shape[1]]) * self.spec.sigma_R
        return residual

    def sample_permeability(self, reference, seed):
        reference = np.asarray(reference)
        if (reference.shape != self.shape or not np.all(np.isfinite(reference)) or np.any(reference <= 0)):
            raise ValueError("reference must be positive finite and match the native grid")
        residual = self.sample_residual(seed)
        if self.spec.center == "arithmetic-mean":
            residual -= .5 * np.log(10.) * self.spec.sigma_R**2 * self.represented_unit_variance
        field = (reference * np.power(10., residual)).astype(np.float32)
        if not np.all(np.isfinite(field)) or np.any(field <= 0):
            raise ValueError("generated field is outside positive finite float32 range")
        return field
