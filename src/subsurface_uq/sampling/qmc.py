from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from scipy.stats import norm, qmc

from .coordinates import StochasticPermeabilityMap

Array = np.ndarray


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass
class ScrambledSobolGaussianPermeabilitySampler:
    """Map a scrambled Sobol design to iid standard-normal coordinates.

    RQ1 compares randomized QMC with iid Monte Carlo at powers-of-two budgets.
    The sampler therefore requires n_samples to be a power of two and uses
    Sobol.random_base2. Uniform Sobol points are transformed component-wise
    through the standard-normal inverse CDF before evaluating the stochastic
    permeability map.
    """

    field_map: StochasticPermeabilityMap
    n_samples: int
    batch_size: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        self.n_samples = int(self.n_samples)
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)
        if not _is_power_of_two(self.n_samples):
            raise ValueError("n_samples must be a positive power of two for scrambled Sobol RQMC")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.field_map.dimension <= 0:
            raise ValueError("field_map.dimension must be positive")

    def __len__(self) -> int:
        return self.n_samples

    @property
    def metadata(self) -> dict[str, object]:
        map_metadata = getattr(self.field_map, "metadata", None)
        return {
            "sampler": "ScrambledSobolGaussianPermeabilitySampler",
            "seed": self.seed,
            "scramble": True,
            "sample_count": self.n_samples,
            "batch_size": self.batch_size,
            "coordinate_distribution": "iid_standard_normal_via_inverse_cdf",
            "uniform_design": "scipy.stats.qmc.Sobol(scramble=True)",
            "coordinate_dimension": int(self.field_map.dimension),
            "field_map": None if map_metadata is None else map_metadata,
        }

    def _coordinates(self) -> Array:
        exponent = int(np.log2(self.n_samples))
        engine = qmc.Sobol(
            d=int(self.field_map.dimension),
            scramble=True,
            seed=self.seed,
        )
        uniform = engine.random_base2(exponent)
        lower = np.nextafter(0.0, 1.0)
        upper = np.nextafter(1.0, 0.0)
        uniform = np.clip(uniform, lower, upper)
        coordinates = norm.ppf(uniform)
        if not np.all(np.isfinite(coordinates)):
            raise RuntimeError("Sobol inverse-CDF transform produced non-finite coordinates")
        return np.asarray(coordinates, dtype=np.float64)

    def __iter__(self) -> Iterator[Array]:
        coordinates = self._coordinates()
        for start in range(0, self.n_samples, self.batch_size):
            stop = min(start + self.batch_size, self.n_samples)
            current = coordinates[start:stop]
            fields = np.asarray(self.field_map.map_coordinates(current))
            expected = (stop - start, *self.field_map.field_shape)
            if fields.shape != expected:
                raise ValueError(
                    "stochastic permeability map returned an unexpected shape; "
                    f"expected {expected}, got {fields.shape}"
                )
            if not np.all(np.isfinite(fields)):
                raise ValueError("stochastic permeability map returned non-finite values")
            if np.any(fields <= 0.0):
                raise ValueError("permeability realizations must be strictly positive")
            yield fields
