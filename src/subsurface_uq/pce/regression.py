from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from .basis import evaluate_orthonormal_legendre, total_degree_indices

Array = np.ndarray


@dataclass
class PolynomialChaosRegressor:
    """Scalar non-intrusive PCE fitted by least-squares regression.

    The current implementation targets independent ``U(-1,1)`` coordinates and
    therefore uses an orthonormal Legendre basis. It is intentionally scalar:
    the first end-to-end experiment fits continuous temperature QoIs rather than
    a full temperature field.
    """

    dimension: int
    degree: int
    rcond: float | None = None

    multi_indices_: Array = field(init=False, repr=False)
    coefficients_: Array | None = field(default=None, init=False, repr=False)
    rank_: int | None = field(default=None, init=False)
    singular_values_: Array | None = field(default=None, init=False, repr=False)
    training_rmse_: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        self.degree = int(self.degree)
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.rcond is not None:
            self.rcond = float(self.rcond)
        self.multi_indices_ = total_degree_indices(self.dimension, self.degree)

    @property
    def basis_size(self) -> int:
        return int(self.multi_indices_.shape[0])

    @property
    def expected_basis_size(self) -> int:
        return comb(self.dimension + self.degree, self.degree)

    @property
    def is_fitted(self) -> bool:
        return self.coefficients_ is not None

    @property
    def condition_number(self) -> float | None:
        if self.singular_values_ is None or self.singular_values_.size == 0:
            return None
        smallest = float(self.singular_values_[-1])
        if smallest == 0.0:
            return float("inf")
        return float(self.singular_values_[0] / smallest)

    def design_matrix(self, coordinates: Array) -> Array:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or coordinates.shape[1] != self.dimension:
            raise ValueError(
                f"coordinates must have shape [N,{self.dimension}], "
                f"got {coordinates.shape}"
            )
        return evaluate_orthonormal_legendre(coordinates, self.multi_indices_)

    def fit(self, coordinates: Array, targets: Array) -> "PolynomialChaosRegressor":
        design = self.design_matrix(coordinates)
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError(f"targets must have shape [N], got {targets.shape}")
        if targets.shape[0] != design.shape[0]:
            raise ValueError("targets and coordinates must contain the same number of samples")
        if not np.all(np.isfinite(targets)):
            raise ValueError("targets must be finite")
        if design.shape[0] < design.shape[1]:
            raise ValueError(
                "least-squares PCE requires at least as many samples as basis terms; "
                f"got N={design.shape[0]}, P={design.shape[1]}"
            )

        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design,
            targets,
            rcond=self.rcond,
        )
        rank = int(rank)
        if rank < design.shape[1]:
            raise ValueError(
                "PCE design matrix is rank deficient; "
                f"rank={rank}, basis_size={design.shape[1]}"
            )

        self.coefficients_ = np.asarray(coefficients, dtype=np.float64)
        self.rank_ = rank
        self.singular_values_ = np.asarray(singular_values, dtype=np.float64)
        residual = design @ self.coefficients_ - targets
        self.training_rmse_ = float(np.sqrt(np.mean(residual**2)))
        return self

    def predict(self, coordinates: Array) -> Array:
        if self.coefficients_ is None:
            raise RuntimeError("PolynomialChaosRegressor must be fitted before prediction")
        return self.design_matrix(coordinates) @ self.coefficients_

    @property
    def mean(self) -> float:
        if self.coefficients_ is None:
            raise RuntimeError("PolynomialChaosRegressor must be fitted before moments")
        zero_rows = np.all(self.multi_indices_ == 0, axis=1)
        if int(np.count_nonzero(zero_rows)) != 1:
            raise RuntimeError("PCE basis must contain exactly one constant term")
        return float(self.coefficients_[zero_rows][0])

    @property
    def variance(self) -> float:
        if self.coefficients_ is None:
            raise RuntimeError("PolynomialChaosRegressor must be fitted before moments")
        nonconstant = np.any(self.multi_indices_ != 0, axis=1)
        return float(np.sum(self.coefficients_[nonconstant] ** 2, dtype=np.float64))

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "model": "PolynomialChaosRegressor",
            "basis": "orthonormal_legendre_total_degree",
            "coordinate_distribution": "iid_uniform_minus1_1",
            "dimension": self.dimension,
            "degree": self.degree,
            "basis_size": self.basis_size,
            "rank": self.rank_,
            "condition_number": self.condition_number,
            "training_rmse": self.training_rmse_,
            "mean": None if not self.is_fitted else self.mean,
            "variance": None if not self.is_fitted else self.variance,
        }
