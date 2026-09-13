from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


def matern32_correlation_matrix(
    size: int,
    domain_length_m: float,
    length_scale_m: float,
) -> Array:
    """Return the 1-D Matérn-3/2 correlation matrix on cell centers.

    ``size`` cells cover ``domain_length_m`` with spacing
    ``domain_length_m / size``. The returned matrix has unit diagonal and uses

    ``rho(r) = (1 + sqrt(3) r) exp(-sqrt(3) r)``

    with ``r = |x-x'| / length_scale_m``.
    """

    size = int(size)
    domain_length_m = float(domain_length_m)
    length_scale_m = float(length_scale_m)
    if size <= 0:
        raise ValueError("size must be positive")
    if not np.isfinite(domain_length_m) or domain_length_m <= 0.0:
        raise ValueError("domain_length_m must be finite and positive")
    if not np.isfinite(length_scale_m) or length_scale_m <= 0.0:
        raise ValueError("length_scale_m must be finite and positive")

    spacing = domain_length_m / size
    locations = (np.arange(size, dtype=np.float64) + 0.5) * spacing
    scaled_distance = np.abs(locations[:, None] - locations[None, :]) / length_scale_m
    root3_r = np.sqrt(3.0) * scaled_distance
    return (1.0 + root3_r) * np.exp(-root3_r)


def _sorted_psd_eigendecomposition(matrix: Array) -> tuple[Array, Array]:
    """Symmetric eigendecomposition sorted from largest to smallest value."""

    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=np.float64))
    scale = max(float(np.max(np.abs(values))), 1.0)
    if float(np.min(values)) < -1e-10 * scale:
        raise RuntimeError("covariance matrix has a materially negative eigenvalue")
    values = np.clip(values, 0.0, None)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


@dataclass
class KLLogGaussianPermeabilityMap:
    """Discrete KL map for a separable log10-Gaussian permeability field.

    The log10-permeability covariance is the separable product

    ``sigma^2 * C_y ⊗ C_x``

    where each one-dimensional factor is a Matérn-3/2 correlation matrix. This
    avoids constructing the full ``(H*W) x (H*W)`` covariance matrix. If
    ``C_y u_i = lambda_i^y u_i`` and ``C_x v_j = lambda_j^x v_j``, the two-
    dimensional eigenpairs are outer products with eigenvalues

    ``sigma^2 * lambda_i^y * lambda_j^x``.

    Array/domain convention: ``shape=(H,W)``, ``domain_size_m=(L_y,L_x)`` and
    ``length_scale_m=(ell_y,ell_x)``. ``n_modes`` takes precedence over
    ``energy_threshold`` when supplied.
    """

    shape: tuple[int, int]
    domain_size_m: tuple[float, float]
    mean_log10_k: float
    std_log10_k: float
    length_scale_m: tuple[float, float]
    n_modes: int | None = None
    energy_threshold: float = 0.95

    _eigvals_y: Array = field(init=False, repr=False)
    _eigvecs_y: Array = field(init=False, repr=False)
    _eigvals_x: Array = field(init=False, repr=False)
    _eigvecs_x: Array = field(init=False, repr=False)
    _mode_y: Array = field(init=False, repr=False)
    _mode_x: Array = field(init=False, repr=False)
    _eigenvalues: Array = field(init=False, repr=False)
    _total_variance: float = field(init=False, repr=False)
    _retained_energy_fraction: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("shape must contain two positive entries")
        if len(self.domain_size_m) != 2 or any(
            not np.isfinite(value) or float(value) <= 0.0 for value in self.domain_size_m
        ):
            raise ValueError("domain_size_m must contain two finite positive entries")
        if len(self.length_scale_m) != 2 or any(
            not np.isfinite(value) or float(value) <= 0.0 for value in self.length_scale_m
        ):
            raise ValueError("length_scale_m must contain two finite positive entries")
        if not np.isfinite(self.mean_log10_k):
            raise ValueError("mean_log10_k must be finite")
        if not np.isfinite(self.std_log10_k) or self.std_log10_k <= 0.0:
            raise ValueError("std_log10_k must be finite and positive")
        if not np.isfinite(self.energy_threshold) or not (0.0 < self.energy_threshold <= 1.0):
            raise ValueError("energy_threshold must lie in (0, 1]")

        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.domain_size_m = (
            float(self.domain_size_m[0]),
            float(self.domain_size_m[1]),
        )
        self.length_scale_m = (
            float(self.length_scale_m[0]),
            float(self.length_scale_m[1]),
        )
        self.mean_log10_k = float(self.mean_log10_k)
        self.std_log10_k = float(self.std_log10_k)
        self.energy_threshold = float(self.energy_threshold)

        total_modes = self.shape[0] * self.shape[1]
        if self.n_modes is not None:
            self.n_modes = int(self.n_modes)
            if not (1 <= self.n_modes <= total_modes):
                raise ValueError(
                    f"n_modes must lie in [1, {total_modes}], got {self.n_modes}"
                )

        corr_y = matern32_correlation_matrix(
            self.shape[0], self.domain_size_m[0], self.length_scale_m[0]
        )
        corr_x = matern32_correlation_matrix(
            self.shape[1], self.domain_size_m[1], self.length_scale_m[1]
        )
        self._eigvals_y, self._eigvecs_y = _sorted_psd_eigendecomposition(corr_y)
        self._eigvals_x, self._eigvecs_x = _sorted_psd_eigendecomposition(corr_x)

        product_values = (
            self.std_log10_k**2
            * np.multiply.outer(self._eigvals_y, self._eigvals_x)
        )
        flat_values = product_values.ravel()
        order = np.argsort(flat_values)[::-1]
        sorted_values = flat_values[order]
        self._total_variance = float(np.sum(sorted_values, dtype=np.float64))
        if not self._total_variance > 0.0:
            raise RuntimeError("KL covariance has zero total variance")

        if self.n_modes is None:
            cumulative = np.cumsum(sorted_values, dtype=np.float64)
            target = min(
                self.energy_threshold * self._total_variance,
                float(cumulative[-1]),
            )
            selected_count = int(np.searchsorted(cumulative, target, side="left") + 1)
        else:
            selected_count = self.n_modes

        selected = order[:selected_count]
        mode_y, mode_x = np.unravel_index(selected, product_values.shape)
        self._mode_y = np.asarray(mode_y, dtype=np.int64)
        self._mode_x = np.asarray(mode_x, dtype=np.int64)
        self._eigenvalues = np.asarray(sorted_values[:selected_count], dtype=np.float64)
        if np.any(self._eigenvalues <= 0.0):
            raise RuntimeError("selected KL modes must have strictly positive eigenvalues")

        self.n_modes = int(selected_count)
        self._retained_energy_fraction = float(
            np.sum(self._eigenvalues, dtype=np.float64) / self._total_variance
        )

    @property
    def dimension(self) -> int:
        return int(self.n_modes)

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def eigenvalues(self) -> Array:
        return self._eigenvalues.copy()

    @property
    def mode_pairs(self) -> Array:
        return np.column_stack((self._mode_y, self._mode_x))

    @property
    def retained_energy_fraction(self) -> float:
        return self._retained_energy_fraction

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "map": "KLLogGaussianPermeabilityMap",
            "log_space": "log10",
            "covariance": "separable_matern32",
            "shape": [self.shape[0], self.shape[1]],
            "domain_size_m": [self.domain_size_m[0], self.domain_size_m[1]],
            "axis_convention": "shape=(H,W), domain=(L_y,L_x), length_scale=(ell_y,ell_x)",
            "mean_log10_k": self.mean_log10_k,
            "std_log10_k": self.std_log10_k,
            "length_scale_m": [self.length_scale_m[0], self.length_scale_m[1]],
            "dimension": self.dimension,
            "selection": "fixed_n_modes" if self.n_modes is not None else "energy_threshold",
            "energy_threshold_requested": self.energy_threshold,
            "retained_energy_fraction": self.retained_energy_fraction,
        }

    def _prepare_coordinates(self, coordinates: Array) -> tuple[Array, bool]:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        single = coordinates.ndim == 1
        if single:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2:
            raise ValueError(
                "coordinates must have shape [m] or [B,m], "
                f"got {coordinates.shape}"
            )
        if coordinates.shape[1] != self.dimension:
            raise ValueError(
                f"expected stochastic dimension {self.dimension}, "
                f"got {coordinates.shape[1]}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("stochastic coordinates must be finite")
        return coordinates, single

    def map_log10_coordinates(self, coordinates: Array) -> Array:
        """Map coordinates to truncated-KL log10-permeability fields."""

        coordinates, single = self._prepare_coordinates(coordinates)
        weighted = coordinates * np.sqrt(self._eigenvalues)[None, :]
        vectors_y = self._eigvecs_y[:, self._mode_y]
        vectors_x = self._eigvecs_x[:, self._mode_x]
        fields = self.mean_log10_k + np.einsum(
            "bm,ym,xm->byx",
            weighted,
            vectors_y,
            vectors_x,
            optimize=True,
        )
        return fields[0] if single else fields

    def map_coordinates(self, coordinates: Array) -> Array:
        """Map standard-normal KL coordinates to physical permeability fields."""

        log10_fields = self.map_log10_coordinates(coordinates)
        permeability = np.power(10.0, log10_fields)
        if not np.all(np.isfinite(permeability)):
            raise ValueError("permeability transformation produced non-finite values")
        if np.any(permeability <= 0.0):
            raise ValueError("permeability transformation produced non-positive values")
        return permeability.astype(np.float32, copy=False)
