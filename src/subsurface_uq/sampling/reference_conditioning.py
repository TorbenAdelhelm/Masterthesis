"""Synthetic reference observations, engineered designs and conditional FFT GRFs.

Locations live in the supplied array's local row/column coordinates. The fixed
reference does not identify an original kriging covariance or real borehole map.
Every fixed design has one Gaussian log-field; different designs form a mixture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import hashlib
import math

import numpy as np

from .gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec


def _positive_integer(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return int(value)


def _reference(values):
    if np.iscomplexobj(values):
        raise ValueError("reference must contain real permeability values")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 3 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("reference must be a finite positive 2-D grid, at least 3 cells per axis")
    return values


def _indices(values, shape):
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != 2 or not np.issubdtype(values.dtype, np.number)
            or not np.all(np.isfinite(values)) or not np.array_equal(values, np.rint(values))):
        raise ValueError("indices must have shape [n,2] and contain finite integer cells")
    values = values.astype(np.int64)
    if np.any(values < 0) or np.any(values >= np.asarray(shape)):
        raise ValueError("observation/query indices outside the reference grid")
    return values


def _covariance(spec, first, second, cell_size_m):
    return spec.sigma_R**2 * spec.correlation(
        (first[:, None, 0] - second[None, :, 0]) * cell_size_m,
        (first[:, None, 1] - second[None, :, 1]) * cell_size_m,
    )


class _ObservationSystem:
    """Whiten cross covariances without forming an ill-conditioned inverse."""

    def __init__(self, matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(.5 * (matrix + matrix.T))
        scale = float(np.max(eigenvalues))
        if scale <= 0 or np.min(eigenvalues) < -scale * 1e-10:
            raise ValueError("observation covariance is not positive semidefinite")
        retained = eigenvalues > scale * 1e-12
        self.vectors, self.values = eigenvectors[:, retained], eigenvalues[retained]
        self.factor = self.vectors / np.sqrt(self.values)
        self.rank = int(retained.sum())
        self.condition_number = float(self.values.max() / self.values.min())

    def whiten(self, cross):
        return cross @ self.factor

    def transform(self, innovation):
        return self.factor.T @ innovation

    def project(self, innovation):
        return self.vectors @ (self.vectors.T @ innovation)


def _variance(spec, queries, selected, noise_std, cell_size_m, chunk_size=8192):
    variance = np.full(len(queries), spec.sigma_R**2)
    if len(selected):
        system = _covariance(spec, selected, selected, cell_size_m) + np.diag(np.asarray(noise_std)**2)
        solver = _ObservationSystem(system)
        for start in range(0, len(queries), chunk_size):
            cross = _covariance(spec, queries[start:start+chunk_size], selected, cell_size_m)
            variance[start:start+chunk_size] -= np.sum(solver.whiten(cross)**2, axis=1)
    if np.min(variance, initial=0) < -spec.sigma_R**2 * 1e-8:
        raise ArithmeticError("negative posterior variance exceeds numerical tolerance")
    return np.maximum(variance, 0.)


@dataclass(frozen=True)
class LocationPriorSpec:
    """A declared sequential location-selection policy, not an inferred prior."""

    kind: str = "uniform"
    variance_power: float = 1.
    gradient_power: float = 1.
    exploration_fraction: float = .05
    candidate_stride_cells: int = 1
    margin_cells: int = 0
    min_spacing_m: float = 0.

    def __post_init__(self):
        if self.kind not in {"uniform", "variance", "gradient", "hybrid"}:
            raise ValueError("unknown location prior kind")
        _positive_integer(self.candidate_stride_cells, "candidate_stride_cells")
        _positive_integer(self.margin_cells, "margin_cells", allow_zero=True)
        for name in ("variance_power", "gradient_power", "min_spacing_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.exploration_fraction) or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must lie in [0,1]")


@dataclass(frozen=True)
class SyntheticReferenceMeasurements:
    indices: np.ndarray
    reference_log10_k: np.ndarray
    observed_log10_k: np.ndarray
    observation_std_log10_k: np.ndarray
    selection_probabilities: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        raw = np.asarray(self.indices)
        if (raw.ndim != 2 or raw.shape[1] != 2 or not len(raw)
                or not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw))
                or not np.array_equal(raw, np.rint(raw))):
            raise ValueError("measurement indices must be nonempty integer [n,2]")
        if len(np.unique(raw, axis=0)) != len(raw):
            raise ValueError("duplicate measurement cells are not supported")
        object.__setattr__(self, "indices", raw.astype(np.int64, copy=True))
        for name in ("reference_log10_k", "observed_log10_k", "observation_std_log10_k", "selection_probabilities"):
            arr = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if arr.shape != (len(raw),) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain one finite value per observation")
            object.__setattr__(self, name, arr)
        if np.any(self.observation_std_log10_k < 0):
            raise ValueError("observation noise must be nonnegative")
        if np.any(self.selection_probabilities <= 0) or np.any(self.selection_probabilities > 1):
            raise ValueError("selection probabilities must lie in (0,1]")
        # Array contents cannot be changed behind a frozen design's identity.
        for name in ("indices", "reference_log10_k", "observed_log10_k", "observation_std_log10_k", "selection_probabilities"):
            getattr(self, name).setflags(write=False)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))


def draw_reference_measurements(reference, *, covariance, n_measurements, location_seed,
                                observation_seed, cell_size_m=5., location_prior=None,
                                location_weights=None, measurement_mode="exact",
                                observation_std_log10_k=0.):
    """Draw cells under an engineered policy, then read the fixed reference.

    Noise, when requested, is simulated once and frozen as the observed data.
    Adaptive variance selection depends only on previous locations and declared
    likelihood accuracy, never on future values or accepted field realizations.
    """
    reference = _reference(reference)
    if not isinstance(covariance, GSToolsReferenceSpec):
        raise ValueError("covariance must be a GSToolsReferenceSpec")
    policy = LocationPriorSpec() if location_prior is None else location_prior
    if not isinstance(policy, LocationPriorSpec):
        raise ValueError("location_prior must be a LocationPriorSpec")
    count = _positive_integer(n_measurements, "n_measurements")
    location_seed = _positive_integer(location_seed, "location_seed", allow_zero=True)
    observation_seed = _positive_integer(observation_seed, "observation_seed", allow_zero=True)
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be finite and positive")
    std = float(observation_std_log10_k)
    if not math.isfinite(std) or std < 0:
        raise ValueError("observation_std_log10_k must be finite and nonnegative")
    if measurement_mode not in {"exact", "noisy"}:
        raise ValueError("measurement_mode must be exact or noisy")
    if (measurement_mode == "exact" and std != 0) or (measurement_mode == "noisy" and std <= 0):
        raise ValueError("exact measurements require zero noise; noisy measurements require positive noise")
    if np.iscomplexobj(location_weights):
        raise ValueError("location_weights must be real")
    weights = np.ones(reference.shape) if location_weights is None else np.asarray(location_weights, dtype=np.float64)
    if weights.shape != reference.shape or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("location_weights must be finite nonnegative weights matching the reference")
    margin, stride = policy.margin_cells, policy.candidate_stride_cells
    rows = np.arange(margin, reference.shape[0]-margin, stride)
    cols = np.arange(margin, reference.shape[1]-margin, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    candidates = np.column_stack([rr.ravel(), cc.ravel()])
    if not len(candidates):
        raise ValueError("margin removes the candidate domain")
    base = weights[tuple(candidates.T)].copy()
    available = base > 0
    if available.sum() < count:
        raise ValueError("more measurements requested than eligible candidate cells")
    base /= base.max()
    structure = np.ones(len(candidates))
    if policy.kind in {"gradient", "hybrid"}:
        gy, gx = np.gradient(np.log10(reference), cell_size_m)
        gradient = np.hypot(gy, gx)[tuple(candidates.T)]
        if gradient.max() > 0:
            structure = (gradient / gradient.max())**policy.gradient_power
    rng = np.random.default_rng(location_seed)
    selected, probabilities, selected_variances = [], [], []
    for _ in range(count):
        if not np.any(available):
            raise ValueError("cannot place requested measurements under spacing/weight constraints")
        chosen = np.asarray(selected, dtype=np.int64).reshape(-1, 2)
        variance = None
        if policy.kind in {"variance", "hybrid"}:
            variance = _variance(covariance, candidates, chosen, np.full(len(chosen), std), cell_size_m)
        score = base * structure
        if policy.kind in {"variance", "hybrid"}:
            score = score * (variance / covariance.sigma_R**2)**policy.variance_power
        score = np.where(available, score, 0.)
        baseline = np.where(available, base, 0.)
        baseline /= baseline.sum()
        if score.sum() > 0:
            score /= score.sum()
        else:
            score = baseline.copy()  # Constant gradient/fully observed candidates.
        distribution = (1-policy.exploration_fraction)*score + policy.exploration_fraction*baseline
        # Renormalize accumulated floating point error, not the realized field.
        distribution /= distribution.sum()
        index = int(rng.choice(len(candidates), p=distribution))
        if variance is None:
            preselection_variance = _variance(covariance, candidates[index:index+1], chosen, np.full(len(chosen), std), cell_size_m)[0]
        else:
            preselection_variance = variance[index]
        selected.append(candidates[index].tolist())
        probabilities.append(float(distribution[index]))
        selected_variances.append(float(preselection_variance))
        distance_sq = np.sum((candidates-candidates[index])**2, axis=1) * cell_size_m**2
        available &= distance_sq >= policy.min_spacing_m**2
        available[index] = False
    indices = np.asarray(selected, dtype=np.int64)
    values = np.log10(reference[tuple(indices.T)])
    noise = np.random.default_rng(observation_seed).standard_normal(count)*std
    return SyntheticReferenceMeasurements(
        indices, values, values + noise, np.full(count, std), np.asarray(probabilities),
        {"kind": "synthetic_reference_measurements", "location_prior": asdict(policy),
         "location_seed": location_seed, "observation_seed": observation_seed,
         "measurement_mode": measurement_mode, "candidate_count": int(len(candidates)),
         "eligible_candidate_count": int(np.count_nonzero(base)),
         "location_weights_sha256": hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest(),
         "preselection_variance_log10": selected_variances,
         "coordinate_order": "row,column; local cell centers (index+0.5)*cell_size_m",
         "assumption_status": "engineered location policy; not inferred historical boreholes",
         "likelihood": "independent additive Gaussian noise in log10 permeability"},
    )


class ConditionalReferenceFFTSampler:
    """Full matched radial Gaussian conditioning by Matheron's update.

    For fixed locations and readings, generate m+u+C_xD A^-1(d-m_D-u_D-e),
    u~N(0,C), e~N(0,R), A=C_DD+R. Omitting e for noisy data is incorrect.
    Covariance is the declared matched kernel, with the FFT's recorded numerical
    covariance tolerance; no empirical realization normalization is performed.
    """

    def __init__(self, reference, covariance, observations, *, cell_size_m=5.,
                 mean_mode="reference_median", workers=1, chunk_size=8192, prior_sampler=None):
        self.reference = np.array(_reference(reference), copy=True)
        self.reference.setflags(write=False)
        if not isinstance(covariance, GSToolsReferenceSpec) or covariance.center != "median":
            raise ValueError("conditional mean modes require a median-centered GSToolsReferenceSpec")
        if not isinstance(observations, SyntheticReferenceMeasurements):
            raise ValueError("observations must be SyntheticReferenceMeasurements")
        self.indices = _indices(observations.indices, self.reference.shape)
        self.observations, self.covariance = observations, covariance
        if not math.isfinite(cell_size_m) or cell_size_m <= 0:
            raise ValueError("cell_size_m must be finite and positive")
        self.cell_size_m = float(cell_size_m)
        self.chunk_size = _positive_integer(chunk_size, "chunk_size")
        _positive_integer(workers, "workers")
        if mean_mode not in {"reference_median", "global_log_mean"}:
            raise ValueError("unknown mean_mode")
        self.mean_mode = mean_mode
        logs = np.log10(self.reference)
        self._mean = logs if mean_mode == "reference_median" else np.full(logs.shape, logs.mean())
        if not np.allclose(observations.reference_log10_k, logs[tuple(self.indices.T)], rtol=0, atol=1e-12):
            raise ValueError("observations do not belong to the supplied reference")
        self._mean_d = self._mean[tuple(self.indices.T)]
        self._system = _covariance(covariance, self.indices, self.indices, cell_size_m) + np.diag(observations.observation_std_log10_k**2)
        self._solver = _ObservationSystem(self._system)
        self._innovation = observations.observed_log10_k-self._mean_d
        projected = self._solver.project(self._innovation)
        if np.max(np.abs(projected-self._innovation)) > max(covariance.sigma_R*1e-6, 1e-10):
            raise ValueError("observations incompatible with numerical covariance rank; revise spacing/noise/kernel")
        self._whitened_innovation = self._solver.transform(self._innovation)
        # Exact data should be resolved numerically, not silently weakened to
        # constraints on a smaller eigenspace. Positive-noise designs also must
        # retain their declared independent noise directions.
        if self._solver.rank < len(self.indices):
            raise ValueError("observation covariance loses numerical rank; increase spacing, use fewer observations, or declare positive measurement noise")
        if prior_sampler is None:
            prior_sampler = GSToolsReferenceFFTSampler(covariance, self.reference.shape, cell_size_m=cell_size_m, workers=workers)
        if (not isinstance(prior_sampler, GSToolsReferenceFFTSampler) or prior_sampler.spec != covariance
                or prior_sampler.shape != self.reference.shape or prior_sampler.cell_size_m != cell_size_m):
            raise ValueError("prior_sampler must match reference grid and covariance exactly")
        self.prior_sampler = prior_sampler
        self.metadata = {"sampler": "ConditionalReferenceFFTSampler", "mean_mode": mean_mode,
                         "field_shape": list(self.reference.shape), "cell_size_m": cell_size_m,
                         "conditional_law": "Gaussian log10 field for one fixed observation design",
                         "physical_law": "untruncated pointwise lognormal; units as reference",
                         "conditioning": "Matheron update including independent likelihood-noise realization",
                         "observation_count": len(self.indices), "observation_covariance_rank": self._solver.rank,
                         "observation_covariance_condition_number": self._solver.condition_number,
                         "rank_relative_tolerance": 1e-12, "added_nugget": None,
                         "location_design": observations.metadata, "prior": prior_sampler.metadata,
                         "sample_filtering": None, "histogram_normalization": None,
                         "assumption_status": "reference-guided covariance; original kriging uncertainty not recovered"}

    @property
    def field_shape(self):
        return self.reference.shape

    def _blocks(self):
        for start in range(0, self.reference.size, self.chunk_size):
            flat = np.arange(start, min(start+self.chunk_size, self.reference.size))
            yield start, np.column_stack(np.unravel_index(flat, self.reference.shape))

    def posterior_at_indices(self, indices):
        queries = _indices(indices, self.reference.shape)
        if len(queries) > 4096:
            raise ValueError("dense query covariance limited to 4096 points; use posterior_moments for grids")
        cross = _covariance(self.covariance, queries, self.indices, self.cell_size_m)
        whitened = self._solver.whiten(cross)
        mean = self._mean[tuple(queries.T)] + whitened @ self._whitened_innovation
        covariance = _covariance(self.covariance, queries, queries, self.cell_size_m) - whitened @ whitened.T
        covariance = .5*(covariance+covariance.T)
        diagonal = np.diag(covariance).copy()
        if np.min(diagonal, initial=0) < -self.covariance.sigma_R**2*1e-8:
            raise ArithmeticError("negative posterior variance exceeds numerical tolerance")
        np.fill_diagonal(covariance, np.maximum(diagonal, 0.))
        return mean, covariance

    def posterior_moments(self):
        """Return analytic mean and marginal variance of log10 permeability."""
        mean = self._mean.copy().ravel()
        variance = np.empty(self.reference.size)
        for start, queries in self._blocks():
            cross = _covariance(self.covariance, queries, self.indices, self.cell_size_m)
            whitened = self._solver.whiten(cross)
            mean[start:start+len(queries)] += whitened @ self._whitened_innovation
            values = self.covariance.sigma_R**2-np.sum(whitened**2, axis=1)
            if np.min(values, initial=0) < -self.covariance.sigma_R**2*1e-8:
                raise ArithmeticError("negative posterior variance exceeds numerical tolerance")
            variance[start:start+len(queries)] = np.maximum(values, 0.)
        return mean.reshape(self.field_shape), variance.reshape(self.field_shape)

    def sample_log10(self, seed):
        seed = _positive_integer(seed, "seed", allow_zero=True)
        prior_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
        seed_value = int(prior_seed.generate_state(1, dtype=np.uint64)[0])
        residual = self.prior_sampler.sample_residual(seed_value)
        noise = np.random.default_rng(noise_seed).standard_normal(len(self.indices))*self.observations.observation_std_log10_k
        innovation = self._innovation-residual[tuple(self.indices.T)]-noise
        transformed = self._solver.transform(innovation)
        sample = self._mean.copy().ravel()
        sample += residual.ravel()
        for start, queries in self._blocks():
            cross = _covariance(self.covariance, queries, self.indices, self.cell_size_m)
            sample[start:start+len(queries)] += self._solver.whiten(cross) @ transformed
        sample = sample.reshape(self.field_shape)
        exact = self.observations.observation_std_log10_k == 0
        if np.any(exact):
            discrepancy = sample[tuple(self.indices[exact].T)] - self.observations.observed_log10_k[exact]
            if np.max(np.abs(discrepancy)) > max(self.covariance.sigma_R*1e-7, 1e-10):
                raise ArithmeticError("exact conditioning exceeds numerical residual tolerance; revise spacing/kernel/noise; no draw replacement")
        return sample

    def sample_permeability(self, seed):
        with np.errstate(over="ignore", under="ignore"):
            sample = np.power(10., self.sample_log10(seed))
        if not np.all(np.isfinite(sample)) or np.any(sample <= 0):
            raise ArithmeticError("conditional draw outside finite positive range; no resampling applied")
        return sample
