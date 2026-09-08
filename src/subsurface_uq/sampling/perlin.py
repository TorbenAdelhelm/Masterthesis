from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import noise
import numpy as np

Array = np.ndarray

RELEASE25_PERLIN_SHAPE = (2560, 2560)
RELEASE25_PERLIN_DOMAIN_SIZE_M = (12800.0, 12800.0)
RELEASE25_PERLIN_FREQUENCY = (18.0, 18.0)
RELEASE25_PERLIN_K_MIN = 1.0193679918450561e-11
RELEASE25_PERLIN_K_MAX = 5.09683995922528e-09
RELEASE25_PERLIN_DEFAULT_SEED = 2907
RELEASE25_SYNTHETIC_BACKGROUND_TEMPERATURE_C = 10.6


def historical_perlin_v2_field(
    *,
    shape: tuple[int, int],
    domain_size_m: tuple[float, float],
    k_min: float,
    k_max: float,
    frequency: tuple[float, float],
    offset: tuple[float, float] | list[float] | Array,
) -> Array:
    """Generate one permeability field with the historical 2-D ``perlin_v2`` formula.

    This reproduces the relevant permeability branch of
    ``Dataset-generation-with-Pflotran/scripts/create_varying_field.py`` used for
    the released synthetic LGCNN data:

    1. evaluate ``noise.pnoise2`` on the normalized/scaled domain,
    2. min-max normalize the realized Perlin field to [0, 1],
    3. map that field to ``[log10(k_min), log10(k_max)]``, and
    4. exponentiate with base 10.

    Mathematically, if ``P(x)`` is the raw Perlin realization and
    ``U(x)=(P(x)-P_min)/(P_max-P_min)``, the permeability is

    ``K(x)=10**(log10(k_min) + U(x)*(log10(k_max)-log10(k_min)))``.

    The historical generator used one random base offset and shifted successive
    realizations in x by integer ``base`` values. The caller supplies the final
    offset for one realization so the formula is independently testable.
    """

    h, w = (int(shape[0]), int(shape[1]))
    lx, ly = (float(domain_size_m[0]), float(domain_size_m[1]))
    fx, fy = (float(frequency[0]), float(frequency[1]))
    ox, oy = (float(offset[0]), float(offset[1]))

    if h <= 0 or w <= 0:
        raise ValueError("shape entries must be positive")
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("domain_size_m entries must be positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("frequency entries must be positive")
    if not (np.isfinite(k_min) and np.isfinite(k_max) and 0.0 < k_min < k_max):
        raise ValueError("require finite permeability bounds with 0 < k_min < k_max")

    simulation_area_max = max(lx, ly)
    scale_x = lx / simulation_area_max
    scale_y = ly / simulation_area_max

    values = np.empty((h, w), dtype=np.float64)
    for i in range(h):
        x = (i / h * scale_x + ox) * fx
        for j in range(w):
            y = (j / w * scale_y + oy) * fy
            values[i, j] = noise.pnoise2(x, y)

    current_min = float(np.min(values))
    current_max = float(np.max(values))
    if not current_max > current_min:
        raise RuntimeError("Perlin realization is constant and cannot be min-max normalized")

    unit = (values - current_min) / (current_max - current_min)
    log_min = np.log10(k_min)
    log_max = np.log10(k_max)
    log_k = unit * (log_max - log_min) + log_min
    return np.power(10.0, log_k)


@dataclass
class Release25PerlinPermeabilitySampler:
    """Lazy, reproducible sampler for the released synthetic LGCNN input law.

    The spatial formula follows the historical ``perlin_v2`` implementation.
    Unlike the historical dataset-generation script, which had its seed call
    commented out, this sampler intentionally uses an explicit seed so thesis
    UQ runs are reproducible. ``numpy.random.RandomState`` is used because its
    ``rand`` stream matches the legacy NumPy API used by the historical code.
    """

    n_samples: int
    batch_size: int = 1
    seed: int = RELEASE25_PERLIN_DEFAULT_SEED
    shape: tuple[int, int] = RELEASE25_PERLIN_SHAPE
    domain_size_m: tuple[float, float] = RELEASE25_PERLIN_DOMAIN_SIZE_M
    frequency: tuple[float, float] = RELEASE25_PERLIN_FREQUENCY
    k_min: float = RELEASE25_PERLIN_K_MIN
    k_max: float = RELEASE25_PERLIN_K_MAX
    base_start: int = 0
    base_offset: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("shape must contain two positive entries")
        if len(self.domain_size_m) != 2 or any(float(value) <= 0 for value in self.domain_size_m):
            raise ValueError("domain_size_m must contain two positive entries")
        if len(self.frequency) != 2 or any(float(value) <= 0 for value in self.frequency):
            raise ValueError("frequency must contain two positive entries")
        if not (np.isfinite(self.k_min) and np.isfinite(self.k_max) and 0.0 < self.k_min < self.k_max):
            raise ValueError("require finite permeability bounds with 0 < k_min < k_max")

        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.domain_size_m = (float(self.domain_size_m[0]), float(self.domain_size_m[1]))
        self.frequency = (float(self.frequency[0]), float(self.frequency[1]))
        self.k_min = float(self.k_min)
        self.k_max = float(self.k_max)
        self.seed = int(self.seed)
        self.base_start = int(self.base_start)

        if self.base_offset is None:
            rng = np.random.RandomState(self.seed)
            generated = rng.rand(3) * 4242.0
            self.base_offset = tuple(float(value) for value in generated)
        else:
            if len(self.base_offset) != 3 or not np.all(np.isfinite(self.base_offset)):
                raise ValueError("base_offset must contain three finite entries")
            self.base_offset = tuple(float(value) for value in self.base_offset)

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "sampler": "Release25PerlinPermeabilitySampler",
            "algorithm": "historical_perlin_v2_pnoise2_log10_minmax",
            "seed": self.seed,
            "frequency": [self.frequency[0], self.frequency[1]],
            "k_min": self.k_min,
            "k_max": self.k_max,
            "sample_count": self.n_samples,
            "shape": [self.shape[0], self.shape[1]],
            "domain_size_m": [self.domain_size_m[0], self.domain_size_m[1]],
            "base_start": self.base_start,
            "base_offset": list(self.base_offset),
            "historical_seed_note": (
                "DaRUS settings record seed_id=2907, but the historical generator's "
                "np.random.seed call was commented out. This UQ seed is explicit for reproducibility."
            ),
        }

    def __len__(self) -> int:
        return self.n_samples

    def _field(self, sample_index: int) -> Array:
        base = self.base_start + sample_index
        offset = (
            self.base_offset[0] + base,
            self.base_offset[1],
            self.base_offset[2],
        )
        field = historical_perlin_v2_field(
            shape=self.shape,
            domain_size_m=self.domain_size_m,
            k_min=self.k_min,
            k_max=self.k_max,
            frequency=self.frequency,
            offset=offset,
        )
        return field.astype(np.float32, copy=False)

    def __iter__(self) -> Iterator[Array]:
        for start in range(0, self.n_samples, self.batch_size):
            stop = min(start + self.batch_size, self.n_samples)
            yield np.stack([self._field(index) for index in range(start, stop)], axis=0)
