from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.stats import norm

Array = np.ndarray


@dataclass(frozen=True)
class EmpiricalNormalScoreTransform:
    """Monotone empirical transform between physical log10(K) and Gaussian scores.

    The transform is fitted to a deterministic spatial subsample of the exact
    permeability fields used to train the real-K LGCNN. It preserves an iid
    Gaussian stochastic coordinate representation while replacing the restrictive
    lognormal marginal assumption by an empirical training-data marginal.
    """

    probabilities: Array
    log10_quantiles: Array
    fit_spatial_stride: int
    source_value_count: int
    requested_quantile_count: int
    tail_probability: float

    def __post_init__(self) -> None:
        probabilities = np.asarray(self.probabilities, dtype=np.float64).reshape(-1)
        quantiles = np.asarray(self.log10_quantiles, dtype=np.float64).reshape(-1)
        if probabilities.size < 3 or probabilities.shape != quantiles.shape:
            raise ValueError("normal-score transform needs at least three paired knots")
        if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(quantiles)):
            raise ValueError("normal-score transform knots must be finite")
        if np.any(np.diff(probabilities) <= 0.0):
            raise ValueError("normal-score probabilities must be strictly increasing")
        if np.any(np.diff(quantiles) < 0.0):
            raise ValueError("normal-score quantiles must be non-decreasing")
        if not (0.0 < float(probabilities[0]) < float(probabilities[-1]) < 1.0):
            raise ValueError("normal-score probabilities must lie strictly inside (0,1)")
        if int(self.fit_spatial_stride) <= 0:
            raise ValueError("fit_spatial_stride must be positive")
        if int(self.source_value_count) <= 0:
            raise ValueError("source_value_count must be positive")
        if int(self.requested_quantile_count) < 3:
            raise ValueError("requested_quantile_count must be at least three")
        if not (0.0 < float(self.tail_probability) < 0.5):
            raise ValueError("tail_probability must lie in (0,0.5)")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "log10_quantiles", quantiles)
        object.__setattr__(self, "fit_spatial_stride", int(self.fit_spatial_stride))
        object.__setattr__(self, "source_value_count", int(self.source_value_count))
        object.__setattr__(self, "requested_quantile_count", int(self.requested_quantile_count))
        object.__setattr__(self, "tail_probability", float(self.tail_probability))

    @classmethod
    def fit(
        cls,
        permeability_fields: Array,
        *,
        spatial_stride: int = 4,
        n_quantiles: int = 1025,
        tail_probability: float = 1.0e-4,
    ) -> "EmpiricalNormalScoreTransform":
        values = np.asarray(permeability_fields)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError("permeability_fields must have shape [N,H,W]")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("permeability fields must be finite and strictly positive")
        spatial_stride = int(spatial_stride)
        n_quantiles = int(n_quantiles)
        tail_probability = float(tail_probability)
        if spatial_stride <= 0:
            raise ValueError("spatial_stride must be positive")
        if n_quantiles < 3:
            raise ValueError("n_quantiles must be at least three")
        if not (0.0 < tail_probability < 0.5):
            raise ValueError("tail_probability must lie in (0,0.5)")

        sampled = np.log10(
            np.asarray(values[:, ::spatial_stride, ::spatial_stride], dtype=np.float64)
        ).reshape(-1)
        probabilities = np.linspace(
            tail_probability,
            1.0 - tail_probability,
            n_quantiles,
            dtype=np.float64,
        )
        quantiles = np.quantile(sampled, probabilities)
        return cls(
            probabilities=probabilities,
            log10_quantiles=quantiles,
            fit_spatial_stride=spatial_stride,
            source_value_count=int(sampled.size),
            requested_quantile_count=n_quantiles,
            tail_probability=tail_probability,
        )

    @property
    def lower_log10(self) -> float:
        return float(self.log10_quantiles[0])

    @property
    def upper_log10(self) -> float:
        return float(self.log10_quantiles[-1])

    def _forward_knots(self) -> tuple[Array, Array]:
        """Return unique physical knots and averaged probabilities for interpolation."""

        unique_values, inverse = np.unique(self.log10_quantiles, return_inverse=True)
        if unique_values.size == self.log10_quantiles.size:
            return unique_values, self.probabilities
        sums = np.bincount(inverse, weights=self.probabilities)
        counts = np.bincount(inverse)
        return unique_values, sums / counts

    def to_score(self, log10_k: Array) -> Array:
        values = np.asarray(log10_k, dtype=np.float64)
        knots, probabilities = self._forward_knots()
        clipped = np.clip(values, knots[0], knots[-1])
        probability = np.interp(clipped, knots, probabilities)
        probability = np.clip(
            probability,
            self.probabilities[0],
            self.probabilities[-1],
        )
        return norm.ppf(probability)

    def from_score(self, score: Array) -> Array:
        score_values = np.asarray(score, dtype=np.float64)
        probability = norm.cdf(score_values)
        probability = np.clip(
            probability,
            self.probabilities[0],
            self.probabilities[-1],
        )
        return np.interp(
            probability,
            self.probabilities,
            self.log10_quantiles,
        )

    def permeability_to_score(self, permeability_m2: Array) -> Array:
        values = np.asarray(permeability_m2, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("permeability values must be finite and strictly positive")
        return self.to_score(np.log10(values))

    def score_to_permeability(self, score: Array) -> Array:
        log10_k = self.from_score(score)
        return np.power(10.0, log10_k)

    def local_score_std(
        self,
        log10_k: Array,
        std_log10_k: float | Array,
        *,
        minimum_std: float = 1.0e-3,
    ) -> Array:
        """Approximate log10 observation uncertainty in Gaussian-score units.

        The central finite-difference transform respects the local slope of the
        empirical CDF. This is preferable to reusing one log10-space nugget after
        a nonlinear normal-score transformation.
        """

        values = np.asarray(log10_k, dtype=np.float64)
        sigma = np.asarray(std_log10_k, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
            raise ValueError("values/std_log10_k must be finite and std non-negative")
        lower = self.to_score(values - sigma)
        upper = self.to_score(values + sigma)
        transformed = 0.5 * np.abs(upper - lower)
        return np.maximum(transformed, float(minimum_std))

    def diagnostics(self, permeability_fields: Array) -> dict[str, float]:
        values = np.asarray(permeability_fields)
        if values.ndim == 2:
            values = values[None, ...]
        sampled = np.log10(
            np.asarray(
                values[:, :: self.fit_spatial_stride, :: self.fit_spatial_stride],
                dtype=np.float64,
            )
        ).reshape(-1)
        scores = self.to_score(sampled)
        return {
            "sample_count": int(scores.size),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)),
            "score_q05": float(np.quantile(scores, 0.05)),
            "score_q50": float(np.quantile(scores, 0.50)),
            "score_q95": float(np.quantile(scores, 0.95)),
            "fraction_below_transform_support": float(np.mean(sampled < self.lower_log10)),
            "fraction_above_transform_support": float(np.mean(sampled > self.upper_log10)),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "transform": "empirical_normal_score",
            "physical_variable": "log10_intrinsic_permeability_m2",
            "latent_variable": "standard_normal_score",
            "probabilities": self.probabilities.tolist(),
            "log10_quantiles": self.log10_quantiles.tolist(),
            "fit_spatial_stride": int(self.fit_spatial_stride),
            "source_value_count": int(self.source_value_count),
            "requested_quantile_count": int(self.requested_quantile_count),
            "tail_probability": float(self.tail_probability),
            "tail_policy": "clip_to_empirical_quantile_support",
            "interpretation": (
                "The empirical marginal is learned from the exact full fields used to "
                "train the real-K LGCNN. Gaussian spatial dependence is modeled in "
                "normal-score space; inverse transformation restores the empirical "
                "training marginal without changing the iid Gaussian latent-coordinate law."
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmpiricalNormalScoreTransform":
        if str(payload.get("transform", "")) != "empirical_normal_score":
            raise ValueError("normal-score payload has an unsupported transform")
        return cls(
            probabilities=np.asarray(payload["probabilities"], dtype=np.float64),
            log10_quantiles=np.asarray(payload["log10_quantiles"], dtype=np.float64),
            fit_spatial_stride=int(payload["fit_spatial_stride"]),
            source_value_count=int(payload["source_value_count"]),
            requested_quantile_count=int(payload["requested_quantile_count"]),
            tail_probability=float(payload["tail_probability"]),
        )


@dataclass(frozen=True)
class NormalScoreConditionalPermeabilityMap:
    """Physical permeability map obtained from a conditional Gaussian-score KL map."""

    transform: EmpiricalNormalScoreTransform
    gaussian_map: object

    @property
    def dimension(self) -> int:
        return int(self.gaussian_map.dimension)

    @property
    def field_shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.gaussian_map.field_shape)

    @property
    def metadata(self) -> dict[str, object]:
        gaussian_metadata = getattr(self.gaussian_map, "metadata", None)
        return {
            "map": "NormalScoreConditionalPermeabilityMap",
            "coordinate_distribution": "iid_standard_normal",
            "physical_variable": "intrinsic_permeability_m2",
            "marginal_model": "empirical_training_normal_score_inverse",
            "normal_score_transform": self.transform.to_dict(),
            "gaussian_score_map": gaussian_metadata,
        }

    def map_score_coordinates(self, coordinates: Array) -> Array:
        return np.asarray(
            self.gaussian_map.map_log10_coordinates(coordinates),
            dtype=np.float64,
        )

    def map_log10_coordinates(self, coordinates: Array) -> Array:
        return self.transform.from_score(self.map_score_coordinates(coordinates))

    def map_coordinates(self, coordinates: Array) -> Array:
        permeability = np.power(10.0, self.map_log10_coordinates(coordinates))
        return permeability.astype(np.float32, copy=False)

    def map_log10_coordinates_at_points(
        self,
        coordinates: Array,
        points_yx_m: Array,
    ) -> Array:
        scores = self.gaussian_map.map_log10_coordinates_at_points(
            coordinates,
            points_yx_m,
        )
        return self.transform.from_score(scores)

    def posterior_reference_log10(self) -> Array:
        """Return inverse-normal-score transform of the latent posterior mean field.

        This is a deterministic conditional reference/median-style field for
        visualization. It is not claimed to be the exact physical-space posterior mean
        because the inverse normal-score transform is nonlinear.
        """

        zero = np.zeros(self.dimension, dtype=np.float64)
        return np.asarray(self.map_log10_coordinates(zero), dtype=np.float64)

    def posterior_reference_permeability(self) -> Array:
        return np.power(10.0, self.posterior_reference_log10())
