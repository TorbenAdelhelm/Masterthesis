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


@dataclass
class ContinuousPointConditionalKLLogGaussianPermeabilityMap:
    """Condition a truncated KL field on continuous local (y,x) observations.

    Observation noise must be strictly positive. For the Munich measurement
    workflow this noise contains the fitted nugget (and optionally additional
    known measurement error). The posterior covariance in retained KL
    coordinates is represented through an SVD of the whitened observation
    operator, avoiding an m x m dense covariance/eigendecomposition.

    The map retains exactly m independent posterior standard-normal coordinates,
    where m is the retained prior KL dimension. This keeps the finite stochastic
    representation compatible with MC, RQMC and later Hermite PCE.
    """

    prior: KLLogGaussianPermeabilityMap
    observation_coordinates_yx_m: Array
    observation_log10_k: Array
    observation_std_log10_k: float | Array
    rank_tolerance: float = 1e-10

    _coordinates: Array = field(init=False, repr=False)
    _values: Array = field(init=False, repr=False)
    _noise_std: Array = field(init=False, repr=False)
    _A: Array = field(init=False, repr=False)
    _posterior_mean: Array = field(init=False, repr=False)
    _right_vectors: Array = field(init=False, repr=False)
    _sqrt_delta: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.prior, KLLogGaussianPermeabilityMap):
            raise TypeError("prior must be a KLLogGaussianPermeabilityMap")
        if not np.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive")
        self.rank_tolerance = float(self.rank_tolerance)

        coordinates = np.asarray(self.observation_coordinates_yx_m, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("observation_coordinates_yx_m must have shape [n,2]")
        if coordinates.shape[0] == 0 or not np.all(np.isfinite(coordinates)):
            raise ValueError("continuous observation coordinates must be finite and non-empty")

        values = np.asarray(self.observation_log10_k, dtype=np.float64).reshape(-1)
        if values.shape[0] != coordinates.shape[0] or not np.all(np.isfinite(values)):
            raise ValueError("one finite observation_log10_k value is required per point")

        noise = np.asarray(self.observation_std_log10_k, dtype=np.float64)
        if noise.ndim == 0:
            noise = np.full(values.shape, float(noise), dtype=np.float64)
        else:
            noise = noise.reshape(-1)
        if noise.shape != values.shape:
            raise ValueError("observation_std_log10_k must be scalar or length n")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError(
                "continuous KL conditioning requires strictly positive observation noise; "
                "use the fitted nugget and/or a justified measurement-error standard deviation"
            )

        A = self.prior.mode_matrix_at_coordinates(coordinates)
        residual = values - self.prior.mean_log10_k

        # Posterior mean: A^T (A A^T + R)^-1 residual.
        innovation = A @ A.T + np.diag(noise**2)
        innovation = 0.5 * (innovation + innovation.T)
        try:
            chol = np.linalg.cholesky(innovation)
            solved = np.linalg.solve(chol.T, np.linalg.solve(chol, residual))
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(innovation)
            scale = max(float(np.max(np.abs(eigvals))), 1.0)
            tol = self.rank_tolerance * scale
            positive = eigvals > tol
            if not np.any(positive):
                raise ValueError("conditioning covariance contains no positive variance")
            inverse = (
                eigvecs[:, positive]
                * (1.0 / eigvals[positive])[None, :]
            ) @ eigvecs[:, positive].T
            solved = inverse @ residual
        posterior_mean = A.T @ solved

        # C_post = (I + W^T W)^-1 with W = R^-1/2 A.
        # Its symmetric square root can be applied without materializing an m x m matrix:
        # L = I + V diag(1/sqrt(1+s^2)-1) V^T.
        W = A / noise[:, None]
        _, singular_values, vt = np.linalg.svd(W, full_matrices=False)
        if singular_values.size:
            cutoff = self.rank_tolerance * max(float(singular_values[0]), 1.0)
            active = singular_values > cutoff
            singular_values = singular_values[active]
            right_vectors = vt[active, :].T
        else:
            singular_values = np.empty(0, dtype=np.float64)
            right_vectors = np.empty((self.prior.dimension, 0), dtype=np.float64)
        sqrt_delta = 1.0 / np.sqrt(1.0 + singular_values**2) - 1.0

        self._coordinates = coordinates
        self._values = values
        self._noise_std = noise
        self._A = A
        self._posterior_mean = posterior_mean
        self._right_vectors = right_vectors
        self._sqrt_delta = sqrt_delta

    @property
    def dimension(self) -> int:
        return self.prior.dimension

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.prior.field_shape

    @property
    def posterior_coordinate_mean(self) -> Array:
        return self._posterior_mean.copy()

    @property
    def observation_coordinates_array(self) -> Array:
        return self._coordinates.copy()

    @property
    def observation_std_array(self) -> Array:
        return self._noise_std.copy()

    @property
    def prior_predictive_diagnostics(self) -> dict[str, object]:
        """Describe how compatible the observations are with the unconditioned prior."""

        innovation = self._A @ self._A.T + np.diag(self._noise_std**2)
        innovation = 0.5 * (innovation + innovation.T)
        residual = self._values - self.prior.mean_log10_k
        try:
            solved = np.linalg.solve(innovation, residual)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(innovation, rcond=self.rank_tolerance) @ residual
        marginal_std = np.sqrt(
            np.maximum(np.diag(innovation), np.finfo(float).eps)
        )
        standardized = residual / marginal_std
        mahalanobis_sq = float(residual @ solved)
        z90 = 1.6448536269514722
        return {
            "measurement_count": int(residual.size),
            "rmse_log10": float(np.sqrt(np.mean(residual * residual))),
            "mean_standardized_residual": float(np.mean(standardized)),
            "std_standardized_residual": (
                float(np.std(standardized, ddof=1))
                if standardized.size > 1
                else 0.0
            ),
            "max_abs_standardized_residual": float(np.max(np.abs(standardized))),
            "marginal_90pct_coverage": float(
                np.mean(np.abs(standardized) <= z90)
            ),
            "mahalanobis_squared": mahalanobis_sq,
            "mahalanobis_squared_per_observation": float(
                mahalanobis_sq / max(residual.size, 1)
            ),
            "posterior_coordinate_mean_norm": float(
                np.linalg.norm(self._posterior_mean)
            ),
            "interpretation": (
                "These are diagnostics of the real measurements under the "
                "training-informed unconditioned prior plus observation noise. "
                "They do not filter posterior samples."
            ),
        }

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "map": "ContinuousPointConditionalKLLogGaussianPermeabilityMap",
            "conditioning": "continuous_point_known_mean_gaussian_in_truncated_kl_space",
            "posterior_factorization": "svd_low_rank_square_root",
            "coordinate_distribution": "iid_standard_normal",
            "prior_dimension": int(self.prior.dimension),
            "posterior_dimension": int(self.dimension),
            "n_observations": int(self._coordinates.shape[0]),
            "observation_coordinates_yx_m": self._coordinates.tolist(),
            "observation_log10_k": self._values.tolist(),
            "observation_std_log10_k": self._noise_std.tolist(),
            "observation_std_log10_k_min": float(np.min(self._noise_std)),
            "observation_std_log10_k_max": float(np.max(self._noise_std)),
            "prior_predictive_diagnostics": self.prior_predictive_diagnostics,
            "rank_tolerance": self.rank_tolerance,
            "prior": self.prior.metadata,
        }

    def _prepare_coordinates(self, coordinates: Array) -> tuple[Array, bool]:
        eta = np.asarray(coordinates, dtype=np.float64)
        single = eta.ndim == 1
        if single:
            eta = eta[None, :]
        if eta.ndim != 2 or eta.shape[1] != self.dimension:
            raise ValueError(
                f"posterior coordinates must have shape [B,{self.dimension}] or [{self.dimension}]"
            )
        if not np.all(np.isfinite(eta)):
            raise ValueError("posterior coordinates must be finite")
        return eta, single

    def prior_coordinates(self, coordinates: Array) -> Array:
        eta, single = self._prepare_coordinates(coordinates)
        xi = eta.copy()
        if self._right_vectors.shape[1]:
            projected = eta @ self._right_vectors
            xi += (projected * self._sqrt_delta[None, :]) @ self._right_vectors.T
        xi += self._posterior_mean[None, :]
        return xi[0] if single else xi

    def map_log10_coordinates(self, coordinates: Array) -> Array:
        return self.prior.map_log10_coordinates(self.prior_coordinates(coordinates))

    def map_coordinates(self, coordinates: Array) -> Array:
        return self.prior.map_coordinates(self.prior_coordinates(coordinates))

    def map_log10_coordinates_at_points(
        self,
        coordinates: Array,
        points_yx_m: Array,
    ) -> Array:
        xi = self.prior_coordinates(coordinates)
        return self.prior.map_log10_coordinates_at_points(xi, points_yx_m)

    def posterior_moments_at_points(self, points_yx_m: Array) -> tuple[Array, Array]:
        basis = self.prior.mode_matrix_at_coordinates(points_yx_m)
        mean = self.prior.mean_log10_k + basis @ self._posterior_mean
        transformed = basis.copy()
        if self._right_vectors.shape[1]:
            projected = transformed @ self._right_vectors
            transformed += (projected * self._sqrt_delta[None, :]) @ self._right_vectors.T
        variance = np.sum(transformed**2, axis=1)
        return mean, np.sqrt(np.maximum(variance, 0.0))
