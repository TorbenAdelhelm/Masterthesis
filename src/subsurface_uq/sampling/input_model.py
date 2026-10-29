from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .kl import KLLogGaussianPermeabilityMap
from .kriging import ContinuousPointConditionalKLLogGaussianPermeabilityMap
from .normal_score import (
    EmpiricalNormalScoreTransform,
    NormalScoreConditionalPermeabilityMap,
)


def validate_explicit_training_reference(reference):
    """Require declared training sources, distinct from a nominal scenario field.

    Source checksums record the explicitly selected support data; this validates
    provenance structure and does not independently verify a model's training split.
    """
    if not isinstance(reference, dict):
        raise ValueError("training_reference must be a profile mapping")
    provenance = reference.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") != "explicit_training_fields":
        raise ValueError("training_reference requires explicit training-field provenance")
    sources = provenance.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("training_reference requires non-empty training sources")
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("path"), str) or not source["path"].strip():
            raise ValueError("training source requires a path")
        checksum = source.get("sha256")
        if (not isinstance(checksum, str) or len(checksum) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in checksum)):
            raise ValueError("training source requires a SHA-256 checksum")
    try:
        low, high = (float(reference[key]) for key in ("minimum_k_m2", "maximum_k_m2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("training_reference requires permeability range in m2") from exc
    if not (np.isfinite(low) and np.isfinite(high) and 0.0 < low < high):
        raise ValueError("training_reference permeability range must be positive and finite")
    return reference


@dataclass(frozen=True)
class ConditionalKLInputModel:
    """Reconstructed training-informed conditional KL input law."""

    source_path: Path
    prior: KLLogGaussianPermeabilityMap
    conditional: object
    payload: dict[str, object]
    unconditional_map: object | None = None

    @property
    def unconditional(self):
        """Physical permeability under the same marginal law as conditional."""
        return self.prior if self.unconditional_map is None else self.unconditional_map

    @property
    def training_k_range(self) -> tuple[float, float] | None:
        reference = self.payload.get("training_reference")
        if not isinstance(reference, dict):
            return None
        if int(self.payload.get("schema_version", -1)) == 3:
            # Older schema-3 artifacts stored the nominal field range here. Keep
            # their physical law reproducible without relabeling it as training data.
            if "provenance" not in reference:
                return None
            validate_explicit_training_reference(reference)
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
    schema_version = int(raw.get("schema_version", -1))
    if schema_version not in {1, 2, 3}:
        raise ValueError("unsupported stochastic input model schema_version")
    if raw.get("coordinate_distribution") != "iid_standard_normal":
        raise ValueError(
            "RQ1/PCE conditional KL input model requires iid_standard_normal coordinates"
        )

    if schema_version == 3:
        if raw.get("input_law") not in {"reference-centered-lognormal", "reference-centered-lognormal-candidate"}:
            raise ValueError("unsupported schema-3 input law")
        reference_support = raw.get("training_reference")
        if isinstance(reference_support, dict) and "provenance" in reference_support:
            validate_explicit_training_reference(reference_support)
        from .reference_field import load_reference_field_maps
        unconditional, conditional = load_reference_field_maps(raw, source)
        if (tuple(raw["field_shape"]) != unconditional.field_shape
                or int(raw["coordinate_dimension"]) != conditional.dimension):
            raise ValueError("reference-field declared shape/dimension mismatch")
        return ConditionalKLInputModel(
            source_path=source, prior=unconditional.prior, conditional=conditional,
            payload=dict(raw), unconditional_map=unconditional,
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

    input_law = str(raw.get("input_law", "legacy-lognormal"))
    unconditional_map = prior
    if schema_version >= 2 and input_law == "normal-score-copula":
        transform_raw = raw.get("normal_score_transform")
        if not isinstance(transform_raw, dict):
            raise ValueError(
                "normal-score input model must contain normal_score_transform"
            )
        transform = EmpiricalNormalScoreTransform.from_dict(transform_raw)
        unconditional_map = NormalScoreConditionalPermeabilityMap(
            transform=transform, gaussian_map=prior,
        )
        latent_values = np.asarray(
            conditioning_raw["observation_latent_values"],
            dtype=np.float64,
        )
        latent_std = np.asarray(
            conditioning_raw["observation_std_latent"],
            dtype=np.float64,
        )
        gaussian_conditional = ContinuousPointConditionalKLLogGaussianPermeabilityMap(
            prior=prior,
            observation_coordinates_yx_m=np.asarray(
                conditioning_raw["observation_coordinates_yx_m"],
                dtype=np.float64,
            ),
            observation_log10_k=latent_values,
            observation_std_log10_k=latent_std,
        )
        conditional: object = NormalScoreConditionalPermeabilityMap(
            transform=transform,
            gaussian_map=gaussian_conditional,
        )
        reconstructed_dimension = gaussian_conditional.dimension
    else:
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
        reconstructed_dimension = conditional.dimension

    declared_shape = tuple(int(v) for v in raw.get("field_shape", prior.field_shape))
    if declared_shape != prior.field_shape:
        raise ValueError(
            "stochastic input model field_shape does not match reconstructed KL prior"
        )
    declared_dimension = int(raw.get("coordinate_dimension", reconstructed_dimension))
    if declared_dimension != reconstructed_dimension:
        raise ValueError(
            "stochastic input model coordinate_dimension does not match reconstruction"
        )

    return ConditionalKLInputModel(
        source_path=source,
        prior=prior,
        conditional=conditional,
        payload=dict(raw),
        unconditional_map=unconditional_map,
    )
