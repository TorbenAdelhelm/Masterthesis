"""Local input uncertainty around one observed/interpolated realistic reference.

No covariance is estimated between pixels of different RUN cutouts. Residual
amplitude and covariance describe a stated uncertainty scenario, not an inferred
geological truth. Coordinates remain finite iid standard-normal variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import yaml

from .kl import KLLogGaussianPermeabilityMap
from .kriging import ContinuousPointConditionalKLLogGaussianPermeabilityMap


@dataclass
class ReferenceFieldLogGaussianPermeabilityMap:
    reference_permeability: np.ndarray
    gaussian_map: object
    center: str = "median"

    def __post_init__(self):
        self.reference_permeability = np.asarray(self.reference_permeability, dtype=np.float64).copy()
        if (self.reference_permeability.shape != self.gaussian_map.field_shape
                or not np.all(np.isfinite(self.reference_permeability))
                or np.any(self.reference_permeability <= 0)):
            raise ValueError("reference must be a finite positive field matching the Gaussian map")
        if self.center not in {"median", "arithmetic-mean"}:
            raise ValueError("center must be median or arithmetic-mean")
        self.prior = getattr(self.gaussian_map, "prior", self.gaussian_map)
        if not isinstance(self.prior, KLLogGaussianPermeabilityMap) or self.prior.mean_log10_k != 0:
            raise ValueError("the residual KL prior must have zero log10 mean")
        self._mean_log10 = np.log10(self.reference_permeability)
        if self.center == "arithmetic-mean":
            self._mean_log10 -= 0.5 * np.log(10.0) * self.prior.pointwise_log10_variance()

    @property
    def dimension(self):
        return self.gaussian_map.dimension

    @property
    def field_shape(self):
        return self.prior.field_shape

    @property
    def domain_size_m(self):
        return self.prior.domain_size_m

    @property
    def metadata(self):
        return {
            "map": type(self).__name__,
            "input_law": "reference-centered-lognormal",
            "scenario": getattr(self, "scenario", None),
            "coordinate_distribution": "iid_standard_normal",
            "physical_variable": "intrinsic_permeability_m2",
            "center": self.center,
            "center_semantics": "reference is the unconditional pointwise center; conditioning updates it",
            "reference_array_sha256": sha256(self.reference_permeability.tobytes()).hexdigest(),
            "residual": self.gaussian_map.metadata,
            "uncertainty_status": "residual covariance/amplitude require evidence or an explicit sensitivity assumption",
        }

    def mean_at_points(self, points_yx_m):
        """Bilinear interpolation of the same deterministic grid mean used in G.

        Constant edge extension spans the half-cell margin inside the domain.
        Observations outside the physical domain are rejected.
        """
        points = np.asarray(points_yx_m, dtype=np.float64)
        domain = np.asarray(self.domain_size_m)
        if (points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points))
                or np.any(points < 0) or np.any(points > domain)):
            raise ValueError("points must be finite local [y,x] inside the reference domain")
        axes = [(np.arange(n) + 0.5) * length / n
                for n, length in zip(self.field_shape, domain)]
        clipped = np.clip(points, [a[0] for a in axes], [a[-1] for a in axes])
        return RegularGridInterpolator(axes, self._mean_log10)(clipped)

    def map_log10_coordinates(self, coordinates):
        return self._mean_log10 + self.gaussian_map.map_log10_coordinates(coordinates)

    def map_log10_coordinates_at_points(self, coordinates, points_yx_m):
        return (self.mean_at_points(points_yx_m)
                + self.gaussian_map.map_log10_coordinates_at_points(coordinates, points_yx_m))

    def map_coordinates(self, coordinates):
        with np.errstate(over="ignore", under="ignore"):
            field = np.power(10.0, self.map_log10_coordinates(coordinates)).astype(np.float32)
        if not np.all(np.isfinite(field)) or np.any(field <= 0):
            raise ValueError("reference-field map produced non-finite/non-positive float32 permeability")
        return field


def build_reference_field_maps(reference_permeability, *, cell_size_m, residual_std_log10_k,
                               length_scale_m, covariance_model="matern32", n_modes=None,
                               energy_threshold=0.95, center="median",
                               observation_coordinates_yx_m=None, observation_log10_k=None,
                               observation_std_log10_k=None):
    """Build unconditional and noisy conditional versions of one residual law."""
    reference = np.asarray(reference_permeability)
    if reference.ndim != 2 or not np.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("2-D reference and positive finite cell size required")
    prior = KLLogGaussianPermeabilityMap(
        shape=reference.shape, domain_size_m=tuple(n * cell_size_m for n in reference.shape),
        mean_log10_k=0.0, std_log10_k=residual_std_log10_k,
        length_scale_m=length_scale_m, covariance_model=covariance_model,
        n_modes=n_modes, energy_threshold=energy_threshold,
    )
    unconditional = ReferenceFieldLogGaussianPermeabilityMap(reference, prior, center)
    observations = (observation_coordinates_yx_m, observation_log10_k, observation_std_log10_k)
    if all(value is None for value in observations):
        return unconditional, unconditional
    if any(value is None for value in observations):
        raise ValueError("conditioning requires coordinates, log10 values and positive noise std")
    residual_observations = (np.asarray(observation_log10_k)
                             - unconditional.mean_at_points(observation_coordinates_yx_m))
    posterior = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
        prior=prior, observation_coordinates_yx_m=observation_coordinates_yx_m,
        observation_log10_k=residual_observations,
        observation_std_log10_k=observation_std_log10_k,
    )
    return unconditional, ReferenceFieldLogGaussianPermeabilityMap(reference, posterior, center)


def save_reference_field_input_model(path, unconditional, *, observation_coordinates_yx_m=None,
                                     observation_log10_k=None, observation_std_log10_k=None,
                                     source_metadata=None):
    """Store a portable reference artifact and Gaussian residual law for RQ1/PCE.

    Input measurements must already be converted to the declared intrinsic-
    permeability convention. Reusing parent-generating measurements for a second
    independent update needs a separate justification.
    """
    path = Path(path).resolve()
    observations = (observation_coordinates_yx_m, observation_log10_k, observation_std_log10_k)
    if any(v is not None for v in observations) and any(v is None for v in observations):
        raise ValueError("conditioning requires coordinates, values and noise std")
    if unconditional.gaussian_map is not unconditional.prior:
        raise ValueError("save requires the unconditional reference-field map")
    path.parent.mkdir(parents=True, exist_ok=True)
    reference_path = path.with_suffix(".reference.npz")
    np.savez_compressed(reference_path, permeability_m2=unconditional.reference_permeability)
    conditioning = {}
    if observation_coordinates_yx_m is not None:
        conditioning = {
            "observation_coordinates_yx_m": np.asarray(observation_coordinates_yx_m).tolist(),
            "observation_log10_intrinsic_permeability": np.asarray(observation_log10_k).tolist(),
            "observation_std_log10_k": np.asarray(observation_std_log10_k).tolist(),
        }
    payload = {
        "schema_version": 3, "input_law": "reference-centered-lognormal",
        "scenario": getattr(unconditional, "scenario", None),
        "coordinate_distribution": "iid_standard_normal", "coordinate_dimension": unconditional.dimension,
        "field_shape": list(unconditional.field_shape), "center": unconditional.center,
        "prior": unconditional.prior.metadata, "conditioning": conditioning,
        "reference_field": {"path": reference_path.name,
                            "sha256": sha256(reference_path.read_bytes()).hexdigest()},
        "source_metadata": source_metadata or {},
        "training_reference": {"minimum_k_m2": float(np.min(unconditional.reference_permeability)),
                               "maximum_k_m2": float(np.max(unconditional.reference_permeability))},
        "source_policy": {"covariance": "explicit_residual_sensitivity_assumption",
                          "pixelwise_covariance_between_RUNs": False, "sample_filtering": None},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def load_reference_field_maps(payload, source_path):
    source_path = Path(source_path)
    reference_path = source_path.parent / payload["reference_field"]["path"]
    if sha256(reference_path.read_bytes()).hexdigest() != payload["reference_field"]["sha256"]:
        raise ValueError("reference artifact checksum mismatch")
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = np.asarray(archive["permeability_m2"])
    prior = payload["prior"]
    if prior["mean_log10_k"] != 0 or prior.get("global_mean_std_log10_k", 0) != 0:
        raise ValueError("reference-field artifact requires a zero-mean spatial residual prior")
    shape = tuple(prior["shape"])
    domain = np.asarray(prior["domain_size_m"])
    cell_sizes = domain / np.asarray(shape)
    if reference.shape != shape or not np.allclose(cell_sizes, cell_sizes[0]):
        raise ValueError("reference shape and uniform grid geometry must match the input law")
    conditioning = payload.get("conditioning", {})
    maps = build_reference_field_maps(
        reference, cell_size_m=float(cell_sizes[0]), residual_std_log10_k=prior["std_log10_k"],
        length_scale_m=prior["length_scale_m"], covariance_model=prior["covariance"].removeprefix("separable_"),
        n_modes=prior["requested_n_modes"], energy_threshold=prior["energy_threshold_requested"],
        center=payload["center"],
        observation_coordinates_yx_m=conditioning.get("observation_coordinates_yx_m"),
        observation_log10_k=conditioning.get("observation_log10_intrinsic_permeability"),
        observation_std_log10_k=conditioning.get("observation_std_log10_k"),
    )
    if payload.get("scenario") is not None:
        from .scenarios import ReferenceScenario
        scenario = ReferenceScenario(**payload["scenario"]["config"])
        if scenario.scenario_id != payload["scenario"]["scenario_id"]:
            raise ValueError("scenario ID/config mismatch")
        if (scenario.center != payload["center"]
                or scenario.covariance_family != prior["covariance"].removeprefix("separable_")
                or not np.isclose(scenario.sigma_R, prior["std_log10_k"], rtol=1e-10, atol=0)
                or not np.allclose((scenario.ell_y, scenario.ell_x), prior["length_scale_m"], rtol=1e-10, atol=0)
                or scenario.n_modes != prior["requested_n_modes"]
                or (scenario.energy_threshold is not None
                    and scenario.energy_threshold != prior["energy_threshold_requested"])):
            raise ValueError("scenario config conflicts with serialized residual law")
        for field_map in maps:
            field_map.scenario = scenario.manifest()
    return maps
