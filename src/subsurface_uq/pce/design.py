from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc

Array = np.ndarray


def iid_uniform_design(
    n_samples: int,
    dimension: int,
    *,
    seed: int,
) -> Array:
    """Return iid ``U(-1,1)^dimension`` coordinates."""

    n_samples = int(n_samples)
    dimension = int(dimension)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(int(seed))
    return rng.uniform(-1.0, 1.0, size=(n_samples, dimension))


def latin_hypercube_uniform_design(
    n_samples: int,
    dimension: int,
    *,
    seed: int,
) -> Array:
    """Return a randomized Latin-hypercube design mapped to ``[-1,1]^m``."""

    n_samples = int(n_samples)
    dimension = int(dimension)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    sampler = qmc.LatinHypercube(d=dimension, seed=int(seed))
    unit = sampler.random(n=n_samples)
    return 2.0 * unit - 1.0


def iid_gaussian_design(n_samples: int, dimension: int, *, seed: int) -> Array:
    """Independent standard-normal coordinates for Gaussian-law validation."""
    if n_samples <= 0 or dimension <= 0:
        raise ValueError("n_samples and dimension must be positive")
    return np.random.default_rng(seed).standard_normal((int(n_samples), int(dimension)))


def latin_hypercube_gaussian_design(n_samples: int, dimension: int, *, seed: int) -> Array:
    """Stratified design for a product Gaussian measure using its direct inverse CDF."""
    if n_samples <= 0 or dimension <= 0:
        raise ValueError("n_samples and dimension must be positive")
    unit = qmc.LatinHypercube(d=int(dimension), seed=int(seed)).random(int(n_samples))
    return norm.ppf(np.clip(unit, np.finfo(float).eps, 1.0 - np.finfo(float).eps))
