from __future__ import annotations

from itertools import combinations_with_replacement
from math import factorial

import numpy as np
from numpy.polynomial.legendre import legvander
from numpy.polynomial.hermite_e import hermevander

Array = np.ndarray


def total_degree_indices(dimension: int, degree: int) -> Array:
    """Return total-degree multi-indices sorted by degree then lexicographically."""

    dimension = int(dimension)
    degree = int(degree)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if degree < 0:
        raise ValueError("degree must be non-negative")

    # Generate only admissible terms, rather than filtering (degree+1)^dimension.
    indices = []
    for total in range(degree + 1):
        for axes in combinations_with_replacement(range(dimension), total):
            alpha = [0] * dimension
            for axis in axes:
                alpha[axis] += 1
            indices.append(tuple(alpha))
    indices.sort(key=lambda alpha: (sum(alpha), alpha))
    return np.asarray(indices, dtype=np.int64)


def evaluate_orthonormal_legendre(
    coordinates: Array,
    multi_indices: Array,
) -> Array:
    """Evaluate an orthonormal tensor-product Legendre basis on ``[-1,1]^m``.

    Orthonormality is with respect to the probability measure of independent
    ``U(-1,1)`` variables. In one dimension the normalized polynomial is

    ``psi_n(x) = sqrt(2*n + 1) * P_n(x)``,

    because ``E[P_n(X)^2] = 1 / (2*n + 1)`` for ``X ~ U(-1,1)``.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2:
        raise ValueError(
            "coordinates must have shape [m] or [N,m], "
            f"got {coordinates.shape}"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite")
    tolerance = 1e-12
    if np.any(coordinates < -1.0 - tolerance) or np.any(
        coordinates > 1.0 + tolerance
    ):
        raise ValueError("Legendre coordinates must lie in [-1, 1]")
    coordinates = np.clip(coordinates, -1.0, 1.0)

    multi_indices = np.asarray(multi_indices, dtype=np.int64)
    if multi_indices.ndim != 2:
        raise ValueError("multi_indices must have shape [P,m]")
    if multi_indices.shape[1] != coordinates.shape[1]:
        raise ValueError(
            "multi-index dimension does not match coordinate dimension: "
            f"{multi_indices.shape[1]} != {coordinates.shape[1]}"
        )
    if np.any(multi_indices < 0):
        raise ValueError("multi-indices must be non-negative")

    n_samples = coordinates.shape[0]
    n_terms = multi_indices.shape[0]
    design = np.ones((n_samples, n_terms), dtype=np.float64)

    for axis in range(coordinates.shape[1]):
        max_degree = int(np.max(multi_indices[:, axis], initial=0))
        one_dimensional = legvander(coordinates[:, axis], max_degree)
        normalization = np.sqrt(2.0 * np.arange(max_degree + 1) + 1.0)
        one_dimensional *= normalization[None, :]
        design *= one_dimensional[:, multi_indices[:, axis]]

    return design


def evaluate_orthonormal_hermite(coordinates: Array, multi_indices: Array) -> Array:
    """Probabilists' Hermite products, orthonormal under iid N(0,1).

    psi_alpha(eta) = product_j He_alpha_j(eta_j) / sqrt(alpha_j!).
    This approximates a response in Gaussian coordinates, not an inverse CDF.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim == 1:
        coordinates = coordinates[None, :]
    indices = np.asarray(multi_indices, dtype=np.int64)
    if (coordinates.ndim != 2 or not np.all(np.isfinite(coordinates))
            or indices.ndim != 2 or indices.shape[1] != coordinates.shape[1]
            or np.any(indices < 0)):
        raise ValueError("finite [N,m] coordinates and non-negative [P,m] indices required")
    design = np.ones((coordinates.shape[0], indices.shape[0]), dtype=np.float64)
    for axis in range(coordinates.shape[1]):
        degree = int(np.max(indices[:, axis], initial=0))
        values = hermevander(coordinates[:, axis], degree)
        values /= np.sqrt(np.asarray([factorial(n) for n in range(degree + 1)], dtype=float))
        design *= values[:, indices[:, axis]]
    return design
