from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perlin import (
    RELEASE25_PERLIN_DOMAIN_SIZE_M,
    RELEASE25_PERLIN_FREQUENCY,
    RELEASE25_PERLIN_K_MAX,
    RELEASE25_PERLIN_K_MIN,
    RELEASE25_PERLIN_SHAPE,
    historical_perlin_v2_field,
)

Array = np.ndarray

RELEASE25_PERLIN_OFFSET_SPAN = 4242.0


@dataclass
class PerlinCoordinatePermeabilityMap:
    """Explicit two-coordinate map for the historical 2-D ``perlin_v2`` family.

    The original generator drew a three-component ``base_offset`` from
    ``np.random.rand(3) * 4242`` and then generated sample ``base`` with
    ``offset = base_offset + [base, 0, 0]``. For a 2-D field, however,
    ``make_grid(..., case='perlin_v2')`` only uses the first two offset
    components. Consequently the random starting location relevant to the 2-D
    permeability law is two-dimensional.

    This class standardizes those two random offsets as independent coordinates
    ``xi = (xi_x, xi_y)`` in ``[-1, 1]^2``. Each coordinate is mapped affinely
    to its configured Perlin offset interval. With the release25 defaults this
    corresponds to independent uniform starting offsets in ``[0, 4242]``.

    ``x_base_shift`` represents the deterministic integer x-shift used by the
    historical finite training sequence. It is *not* treated as a stochastic
    coordinate. Thus this map defines an iid continuous law from the same
    Perlin generator family; it does not claim to reproduce the exact joint
    dependence of the original finite training fields, which shared one random
    base offset and then used successive integer x-shifts.
    """

    shape: tuple[int, int] = RELEASE25_PERLIN_SHAPE
    domain_size_m: tuple[float, float] = RELEASE25_PERLIN_DOMAIN_SIZE_M
    frequency: tuple[float, float] = RELEASE25_PERLIN_FREQUENCY
    k_min: float = RELEASE25_PERLIN_K_MIN
    k_max: float = RELEASE25_PERLIN_K_MAX
    offset_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, RELEASE25_PERLIN_OFFSET_SPAN),
        (0.0, RELEASE25_PERLIN_OFFSET_SPAN),
    )
    x_base_shift: float = 0.0

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("shape must contain two positive entries")
        if len(self.domain_size_m) != 2 or any(
            not np.isfinite(value) or float(value) <= 0.0
            for value in self.domain_size_m
        ):
            raise ValueError("domain_size_m must contain two finite positive entries")
        if len(self.frequency) != 2 or any(
            not np.isfinite(value) or float(value) <= 0.0
            for value in self.frequency
        ):
            raise ValueError("frequency must contain two finite positive entries")
        if not (
            np.isfinite(self.k_min)
            and np.isfinite(self.k_max)
            and 0.0 < self.k_min < self.k_max
        ):
            raise ValueError("require finite permeability bounds with 0 < k_min < k_max")
        if len(self.offset_bounds) != 2:
            raise ValueError("offset_bounds must contain x and y intervals")

        normalized_bounds: list[tuple[float, float]] = []
        for bounds in self.offset_bounds:
            if len(bounds) != 2:
                raise ValueError("each offset interval must contain lower and upper bounds")
            lower, upper = float(bounds[0]), float(bounds[1])
            if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
                raise ValueError("each offset interval must be finite with lower < upper")
            normalized_bounds.append((lower, upper))

        if not np.isfinite(self.x_base_shift):
            raise ValueError("x_base_shift must be finite")

        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.domain_size_m = (
            float(self.domain_size_m[0]),
            float(self.domain_size_m[1]),
        )
        self.frequency = (float(self.frequency[0]), float(self.frequency[1]))
        self.k_min = float(self.k_min)
        self.k_max = float(self.k_max)
        self.offset_bounds = (normalized_bounds[0], normalized_bounds[1])
        self.x_base_shift = float(self.x_base_shift)

    @property
    def dimension(self) -> int:
        return 2

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.shape

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "map": "PerlinCoordinatePermeabilityMap",
            "algorithm": "historical_perlin_v2_pnoise2_log10_minmax",
            "coordinate_distribution": "iid_uniform_minus1_1",
            "natural_pce_basis": "Legendre",
            "coordinate_dimension": self.dimension,
            "shape": [self.shape[0], self.shape[1]],
            "domain_size_m": [self.domain_size_m[0], self.domain_size_m[1]],
            "frequency": [self.frequency[0], self.frequency[1]],
            "k_min": self.k_min,
            "k_max": self.k_max,
            "offset_bounds": [
                [self.offset_bounds[0][0], self.offset_bounds[0][1]],
                [self.offset_bounds[1][0], self.offset_bounds[1][1]],
            ],
            "x_base_shift": self.x_base_shift,
            "historical_interpretation": (
                "For the historical 2-D perlin_v2 generator only offset[0] and "
                "offset[1] affect pnoise2. The original finite dataset shared one "
                "random base_offset and used successive deterministic x-shifts. "
                "This map instead exposes the two random starting offsets as iid "
                "continuous coordinates for MC/PCE experiments."
            ),
        }

    def _prepare_coordinates(self, coordinates: Array) -> tuple[Array, bool]:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        single = coordinates.ndim == 1
        if single:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2:
            raise ValueError(
                "coordinates must have shape [2] or [B,2], "
                f"got {coordinates.shape}"
            )
        if coordinates.shape[1] != self.dimension:
            raise ValueError(
                f"expected stochastic dimension {self.dimension}, "
                f"got {coordinates.shape[1]}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("stochastic coordinates must be finite")
        tolerance = 1e-12
        if np.any(coordinates < -1.0 - tolerance) or np.any(
            coordinates > 1.0 + tolerance
        ):
            raise ValueError("Perlin coordinates must lie in [-1, 1]")
        return np.clip(coordinates, -1.0, 1.0), single

    def coordinates_to_offsets(self, coordinates: Array) -> Array:
        """Map standardized coordinates in ``[-1,1]^2`` to Perlin offsets."""

        coordinates, single = self._prepare_coordinates(coordinates)
        unit = 0.5 * (coordinates + 1.0)
        lower = np.asarray(
            [self.offset_bounds[0][0], self.offset_bounds[1][0]],
            dtype=np.float64,
        )
        upper = np.asarray(
            [self.offset_bounds[0][1], self.offset_bounds[1][1]],
            dtype=np.float64,
        )
        offsets = lower[None, :] + unit * (upper - lower)[None, :]
        offsets[:, 0] += self.x_base_shift
        return offsets[0] if single else offsets

    def map_coordinates(self, coordinates: Array) -> Array:
        """Map standardized uniform coordinates to physical permeability fields."""

        coordinates, single = self._prepare_coordinates(coordinates)
        offsets = self.coordinates_to_offsets(coordinates)
        if offsets.ndim == 1:
            offsets = offsets[None, :]

        fields = [
            historical_perlin_v2_field(
                shape=self.shape,
                domain_size_m=self.domain_size_m,
                k_min=self.k_min,
                k_max=self.k_max,
                frequency=self.frequency,
                offset=offset,
            )
            for offset in offsets
        ]
        result = np.stack(fields, axis=0).astype(np.float32, copy=False)
        if not np.all(np.isfinite(result)):
            raise ValueError("Perlin coordinate map produced non-finite values")
        if np.any(result <= 0.0):
            raise ValueError("Perlin coordinate map produced non-positive permeability")
        return result[0] if single else result
