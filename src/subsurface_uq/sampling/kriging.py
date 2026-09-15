from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kl import KLLogGaussianPermeabilityMap

Array = np.ndarray


@dataclass
class ConditionalKLLogGaussianPermeabilityMap:
    """Condition a truncated log10-Gaussian KL field on borehole cells.

    The prior is

    ``Y = mean_log10_k + B xi,  xi ~ N(0, I)``

    with ``Y = log10(K)`` and ``B`` supplied implicitly by
    :class:`KLLogGaussianPermeabilityMap`. Observations at integer grid cells are

    ``d = H Y + eps,  eps ~ N(0, R)``.

    Gaussian conditioning is carried out in the retained KL-coordinate space.
    The posterior coordinate law is factorized as

    ``xi = posterior_mean + L eta,  eta ~ N(0, I_r)``.

    Consequently ``eta`` remains an explicit vector of independent standard
    normal coordinates, which is suitable for Monte Carlo now and Hermite PCE
    later. For zero observation noise this is exact conditioning of the
    *truncated KL prior*; it is not a claim about modes that were discarded by
    KL truncation.

    This is the known-mean Gaussian/simple-kriging analogue. Ordinary kriging
    with an estimated unknown mean is intentionally outside the current scope.
    """

    prior: KLLogGaussianPermeabilityMap
    observation_indices: Array
    observation_log10_k: Array
    observation_std_log10_k: float | Array = 0.0
    rank_tolerance: float = 1e-10

    _indices: Array = field(init=False, repr=False)
    _values: Array = field(init=False, repr=False)
    _noise_std: Array = field(init=False, repr=False)
    _posterior_mean: Array = field(init=False, repr=False)
    _posterior_covariance: Array = field(init=False, repr=False)
    _posterior_factor: Array = field(init=False, repr=False)
    _posterior_eigenvalues: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prior, KLLogGaussianPermeabilityMap):
            raise TypeError("prior must be a KLLogGaussianPermeabilityMap")
        if not np.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive")
        self.rank_tolerance = float(self.rank_tolerance)

        raw_indices = np.asarray(self.observation_indices)
        if raw_indices.ndim != 2 or raw_indices.shape[1] != 2:
            raise ValueError("observation_indices must have shape [n,2]")
        if raw_indices.shape[0] == 0:
            raise ValueError("at least one conditioning observation is required")
        if not np.all(np.isfinite(raw_indices)):
            raise ValueError("observation_indices must be finite")
        rounded = np.rint(raw_indices)
        if not np.array_equal(raw_indices, rounded):
            raise ValueError("observation_indices must be integer-valued grid cells")
        indices = rounded.astype(np.int64)
        if len({tuple(row) for row in indices.tolist()}) != indices.shape[0]:
            raise ValueError("duplicate conditioning cells are not supported")

        values = np.asarray(self.observation_log10_k, dtype=np.float64).reshape(-1)
        if values.shape[0] != indices.shape[0]:
            raise ValueError("one observation_log10_k value is required per conditioning cell")
        if not np.all(np.isfinite(values)):
            raise ValueError("observation_log10_k values must be finite")

        noise = np.asarray(self.observation_std_log10_k, dtype=np.float64)
        if noise.ndim == 0:
            noise = np.full(values.shape, float(noise), dtype=np.float64)
        else:
            noise = noise.reshape(-1)
        if noise.shape != values.shape:
            raise ValueError("observation_std_log10_k must be scalar or length n_observations")
        if not np.all(np.isfinite(noise)) or np.any(noise < 0.0):
            raise ValueError("observation_std_log10_k must be finite and non-negative")

        # Validates grid bounds and returns the retained KL loadings A = H B.
        A = self.prior.mode_matrix_at_indices(indices)
        residual = values - self.prior.mean_log10_k
        innovation_covariance = A @ A.T + np.diag(noise**2)
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.T
        )

        s_values, s_vectors = np.linalg.eigh(innovation_covariance)
        s_scale = max(float(np.max(np.abs(s_values))), 1.0)
        s_tol = self.rank_tolerance * s_scale
        if float(np.min(s_values)) < -10.0 * s_tol:
            raise RuntimeError("conditioning covariance is not positive semidefinite")
        positive = s_values > s_tol
        if not np.any(positive):
            raise ValueError("conditioning observations contain no variance under the KL prior")
        inverse = (
            s_vectors[:, positive]
            * (1.0 / s_values[positive])[None, :]
        ) @ s_vectors[:, positive].T

        # With exact observations, detect values that the truncated prior cannot
        # satisfy rather than silently projecting them to a least-squares target.
        if np.all(noise == 0.0):
            projected = innovation_covariance @ (inverse @ residual)
            mismatch = float(np.linalg.norm(residual - projected))
            scale = max(float(np.linalg.norm(residual)), 1.0)
            if mismatch > 100.0 * self.rank_tolerance * scale:
                raise ValueError(
                    "exact observations are incompatible with the retained KL subspace; "
                    "increase n_modes or use a non-zero observation uncertainty"
                )

        gain = A.T @ inverse
        posterior_mean = gain @ residual
        posterior_covariance = np.eye(self.prior.dimension) - gain @ A
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.T
        )

        c_values, c_vectors = np.linalg.eigh(posterior_covariance)
        c_scale = max(float(np.max(np.abs(c_values))), 1.0)
        c_tol = self.rank_tolerance * c_scale
        if float(np.min(c_values)) < -10.0 * c_tol:
            raise RuntimeError("posterior KL-coordinate covariance is not positive semidefinite")
        c_values = np.clip(c_values, 0.0, None)
        active = c_values > c_tol
        if not np.any(active):
            raise ValueError(
                "conditioning removed all retained stochastic KL directions; "
                "the posterior is deterministic at the chosen truncation"
            )

        posterior_factor = c_vectors[:, active] * np.sqrt(c_values[active])[None, :]

        self._indices = indices
        self._values = values
        self._noise_std = noise
        self._posterior_mean = posterior_mean
        self._posterior_covariance = posterior_covariance
        self._posterior_factor = posterior_factor
        self._posterior_eigenvalues = c_values[active]

    @classmethod
    def from_permeability_observations(
        cls,
        *,
        prior: KLLogGaussianPermeabilityMap,
        observation_indices: Array,
        observation_k: Array,
        observation_std_log10_k: float | Array = 0.0,
        rank_tolerance: float = 1e-10,
    ) -> "ConditionalKLLogGaussianPermeabilityMap":
        """Construct the conditional map from positive physical permeability data."""

        observation_k = np.asarray(observation_k, dtype=np.float64)
        if not np.all(np.isfinite(observation_k)) or np.any(observation_k <= 0.0):
            raise ValueError("observation_k must contain finite positive permeability values")
        return cls(
            prior=prior,
            observation_indices=observation_indices,
            observation_log10_k=np.log10(observation_k),
            observation_std_log10_k=observation_std_log10_k,
            rank_tolerance=rank_tolerance,
        )

    @property
    def dimension(self) -> int:
        """Number of independent standard-normal posterior coordinates ``eta``."""

        return int(self._posterior_factor.shape[1])

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.prior.field_shape

    @property
    def posterior_coordinate_mean(self) -> Array:
        return self._posterior_mean.copy()

    @property
    def posterior_coordinate_covariance(self) -> Array:
        return self._posterior_covariance.copy()

    @property
    def posterior_eigenvalues(self) -> Array:
        return self._posterior_eigenvalues.copy()

    @property
    def observation_indices_array(self) -> Array:
        return self._indices.copy()

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "map": "ConditionalKLLogGaussianPermeabilityMap",
            "conditioning": "known_mean_gaussian_simple_kriging_in_truncated_kl_space",
            "log_space": "log10",
            "coordinate_distribution": "iid_standard_normal",
            "prior_dimension": int(self.prior.dimension),
            "posterior_dimension": self.dimension,
            "n_observations": int(self._indices.shape[0]),
            "observation_indices": self._indices.tolist(),
            "observation_log10_k": self._values.tolist(),
            "observation_std_log10_k": self._noise_std.tolist(),
            "exact_conditioning": bool(np.all(self._noise_std == 0.0)),
            "posterior_coordinate_mean_norm": float(np.linalg.norm(self._posterior_mean)),
            "rank_tolerance": self.rank_tolerance,
            "prior": self.prior.metadata,
            "interpretation": (
                "Conditioning is exact only relative to the retained KL prior. "
                "Independent posterior coordinates are obtained by factorizing the "
                "conditioned KL-coordinate covariance."
            ),
        }

    def _prepare_coordinates(self, coordinates: Array) -> tuple[Array, bool]:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        single = coordinates.ndim == 1
        if single:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2:
            raise ValueError(
                "coordinates must have shape [r] or [B,r], "
                f"got {coordinates.shape}"
            )
        if coordinates.shape[1] != self.dimension:
            raise ValueError(
                f"expected posterior stochastic dimension {self.dimension}, "
                f"got {coordinates.shape[1]}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("posterior stochastic coordinates must be finite")
        return coordinates, single

    def prior_coordinates(self, coordinates: Array) -> Array:
        """Map independent posterior coordinates ``eta`` to prior KL coordinates ``xi``."""

        coordinates, single = self._prepare_coordinates(coordinates)
        xi = self._posterior_mean[None, :] + coordinates @ self._posterior_factor.T
        return xi[0] if single else xi

    def map_log10_coordinates(self, coordinates: Array) -> Array:
        """Map iid posterior coordinates to conditioned ``log10(K)`` fields."""

        xi = self.prior_coordinates(coordinates)
        return self.prior.map_log10_coordinates(xi)

    def map_coordinates(self, coordinates: Array) -> Array:
        """Map iid posterior coordinates to conditioned physical permeability fields."""

        xi = self.prior_coordinates(coordinates)
        return self.prior.map_coordinates(xi)
