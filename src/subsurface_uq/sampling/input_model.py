from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .kl import KLLogGaussianPermeabilityMap
from .kriging import ContinuousPointConditionalKLLogGaussianPermeabilityMap


@dataclass(frozen=True)
class ConditionalKLInputModel:
    """Reconstructed training-informed conditional KL input law."""

    source_path: Path
    prior: KLLogGaussianPermeabilityMap
    conditional: ContinuousPointConditionalKLLogGaussianPermeabilityMap
    payload: dict[str, object]

    @property
    def training_k_range(self) -> tuple[float, float] | None:
        reference = self.payload.get("training_reference")
        if not isinstance(reference, dict):
            return None
        low = reference.get("minimum_k_m2")
        high = reference.get("maximum_k_m2")
        if low is None or high is None:
            return None
        low = float(low)
        high = float(high)
        if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
            return None
        return (low, high)


def load_conditional_kl_input_model(
    path: str | Path,
) -> ConditionalKLInputModel:
    """Load the Gaussian-coordinate input law written by new-domain-generate."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("stochastic input model must contain a YAML mapping")
    if int(raw.get("schema_version", -1)) != 1:
        raise ValueError("unsupported stochastic input model schema_version")
    if raw.get("coordinate_distribution") != "iid_standard_normal":
        raise ValueError(
            "RQ1/PCE conditional KL input model requires iid_standard_normal coordinates"
        )

    prior_raw = raw.get("prior")
    conditioning_raw = raw.get("conditioning")
    if not isinstance(prior_raw, dict) or not isinstance(conditioning_raw, dict):
        raise ValueError("stochastic input model must contain prior and conditioning mappings")

    covariance = str(prior_raw.get("covariance", ""))
    if not covariance.startswith("separable_"):
        raise ValueError("input-model prior covariance must be separable for the KL path")
    covariance_model = covariance.removeprefix("separable_")
    requested_n_modes = prior_raw.get("requested_n_modes")
    n_modes = None if requested_n_modes is None else int(requested_n_modes)
    energy_threshold = float(prior_raw.get("energy_threshold_requested", 0.95))

    prior = KLLogGaussianPermeabilityMap(
        shape=tuple(int(v) for v in prior_raw["shape"]),
        domain_size_m=tuple(float(v) for v in prior_raw["domain_size_m"]),
        mean_log10_k=float(prior_raw["mean_log10_k"]),
        std_log10_k=float(prior_raw["std_log10_k"]),
        global_mean_std_log10_k=float(
            prior_raw.get("global_mean_std_log10_k", 0.0)
        ),
        length_scale_m=tuple(float(v) for v in prior_raw["length_scale_m"]),
        covariance_model=covariance_model,
        n_modes=n_modes,
        energy_threshold=energy_threshold,
    )

    conditional = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
        prior=prior,
        observation_coordinates_yx_m=np.asarray(
            conditioning_raw["observation_coordinates_yx_m"],
            dtype=np.float64,
        ),
        observation_log10_k=np.asarray(
            conditioning_raw["observation_log10_intrinsic_permeability"],
            dtype=np.float64,
        ),
        observation_std_log10_k=np.asarray(
            conditioning_raw["observation_std_log10_k"],
            dtype=np.float64,
        ),
    )

    declared_shape = tuple(int(v) for v in raw.get("field_shape", prior.field_shape))
    if declared_shape != prior.field_shape:
        raise ValueError(
            "stochastic input model field_shape does not match reconstructed KL prior"
        )
    declared_dimension = int(raw.get("coordinate_dimension", conditional.dimension))
    if declared_dimension != conditional.dimension:
        raise ValueError(
            "stochastic input model coordinate_dimension does not match reconstruction"
        )

    return ConditionalKLInputModel(
        source_path=source,
        prior=prior,
        conditional=conditional,
        payload=dict(raw),
    )
