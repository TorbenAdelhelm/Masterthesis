from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class NewLGCNNDomain:
    """Projected square/rectangular domain for a newly generated LGCNN field."""

    west_edge_m: float
    south_edge_m: float
    cell_size_m: float
    nx: int
    ny: int
    selection_method: str
    conditioning_measurement_count: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.west_edge_m) or not np.isfinite(self.south_edge_m):
            raise ValueError("domain origin must be finite")
        if not np.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")
        if int(self.nx) <= 0 or int(self.ny) <= 0:
            raise ValueError("nx and ny must be positive")

    @property
    def east_edge_m(self) -> float:
        return float(self.west_edge_m + self.nx * self.cell_size_m)

    @property
    def north_edge_m(self) -> float:
        return float(self.south_edge_m + self.ny * self.cell_size_m)

    @property
    def first_cell_center_x_m(self) -> float:
        return float(self.west_edge_m + 0.5 * self.cell_size_m)

    @property
    def first_cell_center_y_m(self) -> float:
        return float(self.south_edge_m + 0.5 * self.cell_size_m)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.ny), int(self.nx))

    @property
    def size_m(self) -> tuple[float, float]:
        return (float(self.ny * self.cell_size_m), float(self.nx * self.cell_size_m))

    def contains_xy(self, x_m: Array, y_m: Array) -> Array:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        return (
            (x >= self.west_edge_m)
            & (x < self.east_edge_m)
            & (y >= self.south_edge_m)
            & (y < self.north_edge_m)
        )

    def projected_xy_to_local_yx(self, x_m: Array, y_m: Array) -> Array:
        x = np.asarray(x_m, dtype=np.float64)
        y = np.asarray(y_m, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x_m and y_m must have matching shapes")
        if not np.all(self.contains_xy(x, y)):
            raise ValueError("projected point lies outside the new LGCNN domain")
        return np.column_stack((y - self.south_edge_m, x - self.west_edge_m))

    def to_dict(self) -> dict[str, object]:
        return {
            "west_edge_m": float(self.west_edge_m),
            "south_edge_m": float(self.south_edge_m),
            "east_edge_m": self.east_edge_m,
            "north_edge_m": self.north_edge_m,
            "first_cell_center_x_m": self.first_cell_center_x_m,
            "first_cell_center_y_m": self.first_cell_center_y_m,
            "cell_size_m": float(self.cell_size_m),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "shape_yx": [int(self.ny), int(self.nx)],
            "domain_size_yx_m": [self.size_m[0], self.size_m[1]],
            "selection_method": self.selection_method,
            "conditioning_measurement_count": int(self.conditioning_measurement_count),
            "array_convention": "geographic_yx_axis0_y_axis1_x",
        }


def _aligned(value: float, cell_size_m: float) -> float:
    return float(np.round(float(value) / float(cell_size_m)) * float(cell_size_m))


def select_new_lgcnn_domain(
    coordinates_xy_m: Array,
    *,
    domain_size_m: float = 12800.0,
    cell_size_m: float = 5.0,
    west_edge_m: float | None = None,
    south_edge_m: float | None = None,
) -> tuple[NewLGCNNDomain, Array]:
    """Choose an explicit or measurement-dense projected LGCNN domain.

    Automatic selection searches cell-aligned square windows and first maximizes
    the number of real measurements inside the 12.8 km domain. Ties are broken
    by proximity of the window centre to the full measurement centroid and then
    deterministically by west/south origin.
    """

    coordinates = np.asarray(coordinates_xy_m, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates_xy_m must have shape [n,2] with n>0")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("measurement coordinates must be finite")
    domain_size_m = float(domain_size_m)
    cell_size_m = float(cell_size_m)
    if domain_size_m <= 0.0 or cell_size_m <= 0.0:
        raise ValueError("domain_size_m and cell_size_m must be positive")
    cells = domain_size_m / cell_size_m
    if not np.isclose(cells, round(cells), atol=1e-9):
        raise ValueError("domain_size_m must be an integer multiple of cell_size_m")
    n = int(round(cells))

    explicit = (west_edge_m is not None) or (south_edge_m is not None)
    if explicit:
        if west_edge_m is None or south_edge_m is None:
            raise ValueError("explicit domain requires both west_edge_m and south_edge_m")
        west = float(west_edge_m)
        south = float(south_edge_m)
        method = "explicit_projected_origin"
    else:
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        # Candidate edges are positions where a measurement just enters/leaves
        # a cell-aligned interval, plus a centroid-centred candidate.
        west_candidates = set()
        south_candidates = set()
        for value in x:
            west_candidates.add(_aligned(np.floor(value / cell_size_m) * cell_size_m, cell_size_m))
            west_candidates.add(_aligned(np.ceil((value - domain_size_m) / cell_size_m) * cell_size_m, cell_size_m))
        for value in y:
            south_candidates.add(_aligned(np.floor(value / cell_size_m) * cell_size_m, cell_size_m))
            south_candidates.add(_aligned(np.ceil((value - domain_size_m) / cell_size_m) * cell_size_m, cell_size_m))
        centroid = np.mean(coordinates, axis=0)
        west_candidates.add(_aligned(centroid[0] - 0.5 * domain_size_m, cell_size_m))
        south_candidates.add(_aligned(centroid[1] - 0.5 * domain_size_m, cell_size_m))

        best: tuple[int, float, float, float] | None = None
        for west_candidate in sorted(west_candidates):
            x_inside = (x >= west_candidate) & (x < west_candidate + domain_size_m)
            if not np.any(x_inside):
                continue
            for south_candidate in sorted(south_candidates):
                inside = x_inside & (y >= south_candidate) & (y < south_candidate + domain_size_m)
                count = int(np.count_nonzero(inside))
                if count == 0:
                    continue
                center = np.asarray(
                    [west_candidate + 0.5 * domain_size_m, south_candidate + 0.5 * domain_size_m]
                )
                centroid_distance2 = float(np.sum((center - centroid) ** 2))
                score = (-count, centroid_distance2, west_candidate, south_candidate)
                if best is None or score < best:
                    best = score
        if best is None:
            raise RuntimeError("automatic domain search found no measurement-containing window")
        west = float(best[2])
        south = float(best[3])
        method = "auto_max_measurement_count_then_centroid"

    inside = (
        (coordinates[:, 0] >= west)
        & (coordinates[:, 0] < west + domain_size_m)
        & (coordinates[:, 1] >= south)
        & (coordinates[:, 1] < south + domain_size_m)
    )
    count = int(np.count_nonzero(inside))
    if count == 0:
        raise ValueError("selected domain contains no real measurements")
    domain = NewLGCNNDomain(
        west_edge_m=west,
        south_edge_m=south,
        cell_size_m=cell_size_m,
        nx=n,
        ny=n,
        selection_method=method,
        conditioning_measurement_count=count,
    )
    return domain, inside
