"""Run an explicit YAML reference-scenario matrix, with optional RQ1 propagation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import yaml

from ..io import sha256_file
from ..sampling.scenarios import (
    reference_scenario_from_config,
    scenario_matrix,
    scenario_weights,
)
from ..sampling.reference_field import save_reference_field_input_model
from ..sampling.coordinates import GaussianCoordinatePermeabilitySampler
from ..sampling.geostatistical_target import (
    Base10LognormalTarget,
    HydraulicConductivityValidationAccumulator,
    validate_hydraulic_conductivity_marginal,
)
from ..sampling.hydraulic_conductivity import (
    FluidProperties,
    hydraulic_conversion_metadata,
    permeability_to_hydraulic_conductivity,
)
from ..sampling.spatial_diagnostics import compare_reference_ensemble
from ..sampling.training_compatibility import (
    characterize_training_patch_distribution, TrainingPatchCompatibilityDiagnostics)
from .reference_field_permeability import coarsen_reference


def _scenario_instance_identity(scenario, *, reference_sha256, reference_shape,
                                cell_size_m, permeability_convention):
    """Identify one scientific scenario instance, including its reference bytes/grid."""
    payload = {
        "scenario_id": scenario.scenario_id,
        "reference_sha256": str(reference_sha256),
        "reference_shape": [int(v) for v in reference_shape],
        "cell_size_m": float(cell_size_m),
        "permeability_convention": str(permeability_convention),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "ref-instance-" + sha256(canonical.encode()).hexdigest()[:20], payload


def _guard_scenario_directory(directory, instance_id):
    """Do not silently replace outputs produced from a different reference/grid."""
    if not directory.exists():
        return
    entries = list(directory.iterdir())
    if not entries:
        return
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"scenario directory {directory} is non-empty but has no manifest; refusing to overwrite it"
        )
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"scenario directory {directory} has an unreadable manifest; refusing to overwrite it"
        ) from exc
    if existing.get("scenario_instance_id") != instance_id:
        raise ValueError(
            f"scenario directory {directory} belongs to a different scientific instance; "
            "use a new output directory or restore the original reference/grid"
        )


def _fluid_from_config(payload):
    if payload is None:
        return FluidProperties()
    if not isinstance(payload, dict):
        raise ValueError("fluid_properties must be a mapping")
    allowed = {"density_kg_m3", "dynamic_viscosity_pa_s", "gravity_m_s2"}
    if not set(payload) <= allowed:
        raise ValueError("unknown fluid_properties keys")
    return FluidProperties(**payload)


def run_matrix(config_path, output_dir, *, rq1_config=None):
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    allowed = {"base", "axes", "references", "cell_size_m", "permeability_convention",
               "n_samples", "seed", "diagnostic_factor", "scenario_weights", "training_support",
               "storage", "distribution_target", "distribution_validation", "fluid_properties"}
    if not isinstance(config, dict) or not set(config) <= allowed:
        raise ValueError("unknown matrix config keys")
    scenarios = scenario_matrix(
        reference_scenario_from_config(config["base"]), config.get("axes", {})
    )
    weights = scenario_weights([s.scenario_id for s in scenarios], config.get("scenario_weights"))
    if config["permeability_convention"] not in {"historical-training", "physical"}:
        raise ValueError("explicit permeability convention required")
    permeability_convention = config["permeability_convention"]
    target = Base10LognormalTarget.from_mapping(config.get("distribution_target"))
    fluid = _fluid_from_config(config.get("fluid_properties"))
    validation_tolerances = config.get("distribution_validation")
    if validation_tolerances is not None and not isinstance(validation_tolerances, dict):
        raise ValueError("distribution_validation must be a mapping of heuristic tolerances")
    cell = float(config["cell_size_m"])
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError("cell_size_m must be positive and finite")
    count, seed = config.get("n_samples", 32), config.get("seed", 4901)
    factor = config.get("diagnostic_factor", 1)
    if (not isinstance(count, int) or count < 2 or isinstance(factor, bool)
            or not isinstance(factor, int) or factor <= 0):
        raise ValueError("n_samples >= 2 and positive integer diagnostic_factor required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    storage = config.get("storage", {})
    if not isinstance(storage, dict) or not set(storage) <= {"retain_generated_fields"}:
        raise ValueError("storage accepts only retain_generated_fields")
    retain_generated_fields = storage.get("retain_generated_fields", True)
    if not isinstance(retain_generated_fields, bool):
        raise ValueError("storage.retain_generated_fields must be boolean")
    references = config["references"]
    def resolve(path):
        return (source.parent / path).resolve()
    paths = {s.reference_run: resolve(references[s.reference_run]) for s in scenarios}
    reference_shapes = set()
    for path in paths.values():
        reference = np.load(path, allow_pickle=False)
        reference_shapes.add(reference.shape)
        coarse_reference = coarsen_reference(reference, factor)
        if min(coarse_reference.shape) < 3:
            raise ValueError("diagnostic grid must have at least three cells in each direction")
        if any(s.n_modes is not None and s.n_modes > reference.size for s in scenarios
               if paths[s.reference_run] == path):
            raise ValueError("n_modes exceeds reference grid dimension")
    support = config.get("training_support")
    profile = None
    if support:
        if not set(support) <= {"reference_npy", "box_size", "skip_per_dir", "feature_stride", "max_patches"}:
            raise ValueError("unknown training_support config keys")
        fields = np.stack([np.load(resolve(p), allow_pickle=False) for p in support["reference_npy"]])
        if any(shape != fields.shape[1:] for shape in reference_shapes):
            raise ValueError("training_support and scenario references must share the full field shape")
        profile = characterize_training_patch_distribution(
            fields, cell_size_m=cell, box_size=support.get("box_size", 1280),
            skip_per_dir=support.get("skip_per_dir", 8), feature_stride=support.get("feature_stride", 8),
            max_patches=support.get("max_patches", 192), field_names=support["reference_npy"])
    root = Path(output_dir).resolve()
    rq1_template = None
    if rq1_config:
        from ..rq1.config import load_rq1_config
        rq1_template = load_rq1_config(
            rq1_config, input_model_override=root / "stochastic_input_model.yaml")
    root.mkdir(parents=True, exist_ok=True)
    conversion = hydraulic_conversion_metadata(permeability_convention, fluid=fluid)
    report = {"schema_version": 3, "config_sha256": sha256_file(source),
              "seed": seed, "n_samples": count, "scenario_weights": weights,
              "weight_status": "explicit scenario assumptions, never inferred probabilities" if weights else "absent; results per scenario",
              "pooling": False,
              "distribution_target": target.metadata,
              "hydraulic_conversion": conversion,
              "storage": {"retain_generated_fields": retain_generated_fields},
              "scenarios": []}
    for scenario in scenarios:
        reference_path = paths[scenario.reference_run]
        reference = np.load(reference_path, allow_pickle=False)
        reference_sha = sha256_file(reference_path)
        source_metadata = {"reference_run": scenario.reference_run,
                           "reference_npy_sha256": reference_sha,
                           "permeability_convention": permeability_convention,
                           "hydraulic_conductivity_validation_target": target.metadata,
                           "hydraulic_conversion": conversion}
        instance_id, instance_identity = _scenario_instance_identity(
            scenario, reference_sha256=reference_sha, reference_shape=reference.shape,
            cell_size_m=cell, permeability_convention=permeability_convention)
        directory = root / scenario.scenario_id
        _guard_scenario_directory(directory, instance_id)
        directory.mkdir(exist_ok=True)
        unconditional, field_map = scenario.build_maps(reference, cell_size_m=cell)
        artifact = directory / "stochastic_input_model.yaml"
        support_metadata = ({"training_reference": {
            **profile.to_dict(), "provenance": {"kind": "explicit_training_fields", "sources": [
                {"path": str(resolve(p)), "sha256": sha256_file(resolve(p))}
                for p in support["reference_npy"]]}}} if profile else {})
        save_reference_field_input_model(artifact, unconditional, source_metadata=source_metadata,
                                         **support_metadata)
        diagnostics = TrainingPatchCompatibilityDiagnostics(profile) if profile else None
        sampler = GaussianCoordinatePermeabilitySampler(field_map=field_map, n_samples=count, batch_size=1, seed=seed)
        generated_path = directory / "generated_fields.npy"
        ensemble = (np.lib.format.open_memmap(generated_path, mode="w+",
                                              dtype=np.float32, shape=(count, *reference.shape))
                    if retain_generated_fields else None)
        if not retain_generated_fields and generated_path.exists():
            generated_path.unlink()
        distribution = HydraulicConductivityValidationAccumulator(
            target, tolerances=validation_tolerances
        )
        coarse = []
        start = 0
        for batch in sampler:
            if ensemble is not None:
                ensemble[start:start + len(batch)] = batch
            for field in batch:
                kh = permeability_to_hydraulic_conductivity(
                    field, convention=permeability_convention, fluid=fluid
                )
                distribution.update(kh)
                coarse.append(coarsen_reference(field, factor))
            if diagnostics:
                diagnostics.update(batch)
            start += len(batch)
        if ensemble is not None:
            ensemble.flush()
        fidelity = compare_reference_ensemble(coarsen_reference(reference, factor), np.asarray(coarse),
                                              cell_size_m=cell * factor)
        patch = diagnostics.finalize() if diagnostics else None
        outside = ([key for key, value in patch["metrics"].items()
                    if value["fraction_generated_patches_outside_training_envelope"] > 0] if patch else [])
        fidelity["training_patch_support"] = patch
        fidelity["compatibility_assessment"] = {
            "status": ("descriptor_deviations_observed" if outside else "within_sampled_descriptor_envelopes") if patch else "unassessed_training_support",
            "interpretation": "Inspect patch descriptor deviations; no calibrated universal threshold or guarantee of in-distribution predictions",
            "sample_filtering": None, "flagged_metrics": outside}
        (directory / "diagnostics.json").write_text(json.dumps(fidelity, indent=2, allow_nan=False), encoding="utf-8")

        reference_kh = permeability_to_hydraulic_conductivity(
            reference, convention=permeability_convention, fluid=fluid
        )
        distribution_report = distribution.finalize()
        distribution_report.update({
            "reference_marginal": validate_hydraulic_conductivity_marginal(
                reference_kh, target, tolerances=validation_tolerances
            ),
            "residual_law": {
                "sigma_R_log10": float(scenario.sigma_R),
                "interpretation": (
                    "sigma_R is the stochastic residual amplitude around the heterogeneous reference; "
                    "it is not the same quantity as the pooled full-field log10 standard deviation."
                ),
            },
            "hydraulic_conversion": conversion,
            "spatial_support_deviation": fidelity["compatibility_assessment"],
            "assumption_coverage": distribution_report["pooled"]["fraction_inside_interval"],
            "log10_sigma_error": distribution_report["pooled"]["log10_std_error"],
            "normality_deviation": distribution_report["pooled"]["normality_deviation"],
            "sample_filtering": None,
        })
        (directory / "distribution_validation.json").write_text(
            json.dumps(distribution_report, indent=2, allow_nan=False), encoding="utf-8"
        )

        manifest = scenario.manifest()
        variance = unconditional.prior.pointwise_log10_variance()
        manifest.update(scenario_instance_id=instance_id, instance_identity=instance_identity,
                        source_metadata=source_metadata, cell_size_m=cell, seed=seed, n_samples=count,
                        dimension=field_map.dimension, retained_energy=unconditional.prior.retained_energy_fraction,
                        retained_log10_variance={"min": float(variance.min()), "mean": float(variance.mean()), "max": float(variance.max())},
                        distribution_validation=str(directory / "distribution_validation.json"),
                        generated_full_field_log10_std=distribution_report["pooled"]["log10_std"],
                        generated_full_field_log10_std_error=distribution_report["pooled"]["log10_std_error"],
                        generated_interval_fraction=distribution_report["pooled"]["fraction_inside_interval"],
                        input_model_sha256=sha256_file(artifact), input_model=str(artifact),
                        storage={"retain_generated_fields": retain_generated_fields,
                                 "generated_fields": str(generated_path) if retain_generated_fields else None})
        if rq1_config:
            from ..rq1.workflow import run_rq1
            rq1 = replace(rq1_template, input_model=artifact, output_root=directory / "rq1",
                          experiment_id=scenario.scenario_id)
            run_rq1(rq1)
            manifest["rq1_output"] = str(rq1.output_root)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        report["scenarios"].append(manifest)
    (root / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rq1-config", help="Fixed-model/QoI template; the matrix overrides its entire grf block, and reference RUN does not change fixed sources")
    args = parser.parse_args(argv)
    report = run_matrix(args.config, args.output_dir, rq1_config=args.rq1_config)
    print(f"Saved {len(report['scenarios'])} separate reference scenarios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
