from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

Array = np.ndarray


def _require_gstools():
    try:
        import gstools as gs
    except ImportError as exc:
        raise ImportError(
            "RadialExponentialPermeabilitySampler requires the optional "
            "'geostat' dependency: pip install 'subsurface-uq[geostat]'"
        ) from exc
    return gs


@dataclass
class RadialExponentialPermeabilitySampler:
    """Log-Gaussian permeability sampler with radial anisotropic exponential covariance.

    The modeled field is Y=log10(K). GSTools evaluates an exponential covariance
    in anisometrized coordinates, so for zero rotation the correlation is

        exp(-sqrt((dx/ell_x)^2 + (dy/ell_y)^2)).

    Optional point observations are imposed by simple kriging followed by
    conditioned random-field generation. The baseline implementation treats
    borehole observations as exact, matching the thesis task definition.

    This sampler is intended for Monte Carlo realism experiments. It does not
    expose finite independent Gaussian coordinates and therefore does not replace
    the KL map used for RQMC or Hermite-PCE experiments.
    """

    shape: tuple[int, int]
    cell_size_m: float
    mean_log10_k: float
    std_log10_k: float
    length_scale_y_m: float
    length_scale_x_m: float
    n_samples: int
    batch_size: int = 1
    seed: int = 2907
    angle_rad: float = 0.0
    observation_indices: Array | None = None
    observation_k: Array | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or any(int(v) <= 0 for v in self.shape):
            raise ValueError("shape must contain two positive entries")
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.cell_size_m = float(self.cell_size_m)
        self.mean_log10_k = float(self.mean_log10_k)
        self.std_log10_k = float(self.std_log10_k)
        self.length_scale_y_m = float(self.length_scale_y_m)
        self.length_scale_x_m = float(self.length_scale_x_m)
        self.n_samples = int(self.n_samples)
        self.batch_size = int(self.batch_size)
        self.seed = int(self.seed)
        self.angle_rad = float(self.angle_rad)
        if not np.isfinite(self.cell_size_m) or self.cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be finite and positive")
        if not np.isfinite(self.mean_log10_k):
            raise ValueError("mean_log10_k must be finite")
        if not np.isfinite(self.std_log10_k) or self.std_log10_k <= 0.0:
            raise ValueError("std_log10_k must be finite and positive")
        if self.length_scale_y_m <= 0.0 or self.length_scale_x_m <= 0.0:
            raise ValueError("length scales must be positive")
        if not np.isfinite(self.angle_rad):
            raise ValueError("angle_rad must be finite")
        if self.n_samples <= 0 or self.batch_size <= 0:
            raise ValueError("n_samples and batch_size must be positive")

        if self.observation_indices is None and self.observation_k is None:
            self.observation_indices = None
            self.observation_k = None
        elif self.observation_indices is None or self.observation_k is None:
            raise ValueError(
                "observation_indices and observation_k must either both be supplied or both omitted"
            )
        else:
            raw = np.asarray(self.observation_indices)
            if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
                raise ValueError("observation_indices must have shape [n,2]")
            if not np.all(np.isfinite(raw)) or not np.array_equal(raw, np.rint(raw)):
                raise ValueError("observation_indices must be finite integer grid cells")
            indices = np.rint(raw).astype(np.int64)
            if len({tuple(row) for row in indices.tolist()}) != indices.shape[0]:
                raise ValueError("duplicate observation cells are not supported")
            if np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= self.shape[0]):
                raise ValueError("observation row index is outside the field")
            if np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= self.shape[1]):
                raise ValueError("observation column index is outside the field")
            values = np.asarray(self.observation_k, dtype=np.float64).reshape(-1)
            if values.shape[0] != indices.shape[0]:
                raise ValueError("one observation_k value is required per observation cell")
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError("observation_k must be finite and positive")
            self.observation_indices = indices
            self.observation_k = values

    @property
    def field_shape(self) -> tuple[int, int]:
        return self.shape

    def __len__(self) -> int:
        return self.n_samples

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "sampler": "RadialExponentialPermeabilitySampler",
            "log_space": "log10",
            "covariance": "radial_anisotropic_exponential",
            "mean_log10_k": self.mean_log10_k,
            "std_log10_k": self.std_log10_k,
            "length_scale_y_m": self.length_scale_y_m,
            "length_scale_x_m": self.length_scale_x_m,
            "angle_rad": self.angle_rad,
            "shape": [self.shape[0], self.shape[1]],
            "cell_size_m": self.cell_size_m,
            "sample_count": self.n_samples,
            "seed": self.seed,
            "conditioning": (
                "none"
                if self.observation_indices is None
                else "exact_simple_kriging_conditioned_random_field"
            ),
            "n_observations": (
                0 if self.observation_indices is None else int(self.observation_indices.shape[0])
            ),
            "coordinate_note": (
                "GSTools random-field generation is used for scalable MC. This sampler "
                "does not expose explicit finite Gaussian coordinates for RQMC/PCE."
            ),
        }

    def _grid(self) -> tuple[Array, Array]:
        h, w = self.shape
        x = (np.arange(w, dtype=np.float64) + 0.5) * self.cell_size_m
        y = (np.arange(h, dtype=np.float64) + 0.5) * self.cell_size_m
        return x, y

    def _generator(self):
        gs = _require_gstools()
        model = gs.Exponential(
            dim=2,
            var=self.std_log10_k**2,
            len_scale=[self.length_scale_x_m, self.length_scale_y_m],
            angles=self.angle_rad,
        )
        if self.observation_indices is None:
            return gs.SRF(model, mean=self.mean_log10_k)

        assert self.observation_k is not None
        x, y = self._grid()
        rows = self.observation_indices[:, 0]
        cols = self.observation_indices[:, 1]
        cond_pos = [x[cols], y[rows]]
        cond_val = np.log10(self.observation_k)
        krige = gs.krige.Simple(
            model,
            cond_pos=cond_pos,
            cond_val=cond_val,
            mean=self.mean_log10_k,
            exact=True,
        )
        return gs.CondSRF(krige)

    def _seeds(self) -> Array:
        sequence = np.random.SeedSequence(self.seed)
        children = sequence.spawn(self.n_samples)
        return np.asarray(
            [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children],
            dtype=np.uint32,
        )

    def _one_log10_field(self, generator, seed: int) -> Array:
        x, y = self._grid()
        field_xy = generator(
            (x, y),
            seed=int(seed),
            mesh_type="structured",
            store=False,
        )
        field_xy = np.asarray(field_xy, dtype=np.float64)
        expected = (x.size, y.size)
        if field_xy.shape != expected:
            raise RuntimeError(
                f"GSTools returned field shape {field_xy.shape}, expected {expected}"
            )
        return field_xy.T

    def __iter__(self) -> Iterator[Array]:
        generator = self._generator()
        seeds = self._seeds()
        fields: list[Array] = []
        for seed in seeds:
            log10_k = self._one_log10_field(generator, int(seed))
            permeability = np.power(10.0, log10_k).astype(np.float32, copy=False)
            if not np.all(np.isfinite(permeability)) or np.any(permeability <= 0.0):
                raise ValueError("generated permeability contains invalid values")
            fields.append(permeability)
            if len(fields) == self.batch_size:
                yield np.stack(fields, axis=0)
                fields = []
        if fields:
            yield np.stack(fields, axis=0)
