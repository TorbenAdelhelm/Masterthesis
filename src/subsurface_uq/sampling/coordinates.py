from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


@runtime_checkable
class StochasticPermeabilityMap(Protocol):
    """Map finite-dimensional stochastic coordinates to permeability fields.

    The map deliberately does not choose an experimental design or probability
    sampling strategy. It only implements ``xi -> K`` so the same object can be
    reused by Monte Carlo and, later, by PCE training designs.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def field_shape(self) -> tuple[int, int]: ...

    def map_coordinates(self, coordinates: Array) -> Array: ...


@dataclass
class GaussianCoordinatePermeabilitySampler:
    """Adapt a stochastic permeability map to the existing sampler interface.

    Coordinates are independent standard normal draws. The random generator is
    re-created for every iteration so repeated iteration with the same seed is
    reproducible. The wrapped map remains independent of this sampling choice.
    """

    field_map: StochasticPermeabilityMap
    n_samples: int
    batch_size: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.field_map.dimension <= 0:
            raise ValueError("field_map.dimension must be positive")
        self.n_samples = int(self.n_samples)
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)

    def __len__(self) -> int:
        return self.n_samples

    @property
    def metadata(self) -> dict[str, object]:
        map_metadata = getattr(self.field_map, "metadata", None)
        return {
            "sampler": "GaussianCoordinatePermeabilitySampler",
            "seed": self.seed,
            "sample_count": self.n_samples,
            "batch_size": self.batch_size,
            "coordinate_distribution": "iid_standard_normal",
            "coordinate_dimension": int(self.field_map.dimension),
            "field_map": None if map_metadata is None else map_metadata,
        }

    def __iter__(self) -> Iterator[Array]:
        rng = np.random.default_rng(self.seed)
        for start in range(0, self.n_samples, self.batch_size):
            size = min(self.batch_size, self.n_samples - start)
            coordinates = rng.standard_normal((size, self.field_map.dimension))
            fields = np.asarray(self.field_map.map_coordinates(coordinates))
            expected = (size, *self.field_map.field_shape)
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
