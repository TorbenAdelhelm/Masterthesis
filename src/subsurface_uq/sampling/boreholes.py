from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class BoreholeObservationSet:
    indices: Array
    permeability: Array

    @property
    def count(self) -> int:
        return int(self.indices.shape[0])

    @property
    def log10_permeability(self) -> Array:
        return np.log10(self.permeability)

    def to_dict(self) -> dict[str, object]:
        return {
            "indices": self.indices.tolist(),
            "permeability_m2": self.permeability.tolist(),
            "log10_permeability": self.log10_permeability.tolist(),
        }


def sample_borehole_observations(
    field: Array,
    *,
    n_boreholes: int,
    seed: int,
    margin_cells: int = 0,
    min_spacing_cells: float = 0.0,
    max_attempts: int = 100000,
) -> BoreholeObservationSet:
    """Draw reproducible synthetic boreholes from one hidden permeability field.

    Boreholes are sampled on grid-cell centers. Optional margin and minimum
    Euclidean spacing constraints make it possible to avoid unrealistic
    clustering in controlled validation experiments.
    """

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError("field must have shape [H,W]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("field must contain finite positive permeability values")
    n_boreholes = int(n_boreholes)
    margin_cells = int(margin_cells)
    max_attempts = int(max_attempts)
    min_spacing_cells = float(min_spacing_cells)
    if n_boreholes <= 0:
        raise ValueError("n_boreholes must be positive")
    if margin_cells < 0:
        raise ValueError("margin_cells must be non-negative")
    if min_spacing_cells < 0.0:
        raise ValueError("min_spacing_cells must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    h, w = values.shape
    row_low, row_high = margin_cells, h - margin_cells
    col_low, col_high = margin_cells, w - margin_cells
    if row_low >= row_high or col_low >= col_high:
        raise ValueError("margin_cells removes the entire candidate domain")
    candidate_count = (row_high - row_low) * (col_high - col_low)
    if n_boreholes > candidate_count:
        raise ValueError("more boreholes requested than available grid cells")

    rng = np.random.default_rng(int(seed))
    selected: list[tuple[int, int]] = []
    selected_set: set[tuple[int, int]] = set()
    minimum_sq = min_spacing_cells * min_spacing_cells

    attempts = 0
    while len(selected) < n_boreholes and attempts < max_attempts:
        attempts += 1
        row = int(rng.integers(row_low, row_high))
        col = int(rng.integers(col_low, col_high))
        cell = (row, col)
        if cell in selected_set:
            continue
        if minimum_sq > 0.0 and any(
            (row - old_row) ** 2 + (col - old_col) ** 2 < minimum_sq
            for old_row, old_col in selected
        ):
            continue
        selected.append(cell)
        selected_set.add(cell)

    if len(selected) != n_boreholes:
        raise RuntimeError(
            "could not place the requested boreholes under the spacing constraint; "
            "reduce min_spacing_cells or increase max_attempts"
        )

    indices = np.asarray(selected, dtype=np.int64)
    permeability = values[indices[:, 0], indices[:, 1]].astype(np.float64, copy=False)
    return BoreholeObservationSet(indices=indices, permeability=permeability)
