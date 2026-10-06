"""Run an explicit YAML reference-scenario matrix, with optional RQ1 propagation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import yaml

from ..io import sha256_file
from ..sampling.scenarios import ReferenceScenario, scenario_matrix, scenario_weights
from ..sampling.reference_field import save_reference_field_input_model
from ..sampling.coordinates import GaussianCoordinatePermeabilitySampler
from ..sampling.spatial_diagnostics import compare_reference_ensemble
from ..sampling.training_compatibility import (
    characterize_training_patch_distribution, TrainingPatchCompatibilityDiagnostics)
from .reference_field_permeability import coarsen_reference


def run_matrix(config_path, output_dir, *, rq1_config=None):
    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    allowed = {"base", "axes", "references", "cell_size_m", "permeability_convention",
               "n_samples", "seed", "diagnostic_factor", "scenario_weights", "training_support"}
    if not isinstance(config, dict) or not set(config) <= allowed:
        raise ValueError("unknown matrix config keys")
    scenarios = scenario_matrix(ReferenceScenario(**config["base"]), config.get("axes", {}))
    weights = scenario_weights([s.scenario_id for s in scenarios], config.get("scenario_weights"))
    if config["permeability_convention"] not in {"historical-training", "physical"}:
        raise ValueError("explicit permeability convention required")
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
    references = config["references"]
    def resolve(path):
        return (source.parent / path).resolve()
    # Preflight all scenario configurations and references before writing outputs.
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
        # A matrix owns its input law. Validate only the fixed-model, sampling
        # and QoI template; discard any legacy/manual stochastic-input block.
        rq1_template = load_rq1_config(
            rq1_config, input_model_override=root / "stochastic_input_model.yaml")
    root.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "config_sha256": sha256_file(source),
              "seed": seed, "n_samples": count, "scenario_weights": weights,
              "weight_status": "explicit scenario assumptions, never inferred probabilities" if weights else "absent; results per scenario",
              "pooling": False, "scenarios": []}
    for scenario in scenarios:
        reference = np.load(paths[scenario.reference_run], allow_pickle=False)
        unconditional, field_map = scenario.build_maps(reference, cell_size_m=cell)
        directory = root / scenario.scenario_id
        directory.mkdir(exist_ok=True)
        source_metadata = {"reference_run": scenario.reference_run,
                           "reference_npy_sha256": sha256_file(paths[scenario.reference_run]),
                           "permeability_convention": config["permeability_convention"]}
        artifact = directory / "stochastic_input_model.yaml"
        support_metadata = ({"training_reference": {
            **profile.to_dict(), "provenance": {"kind": "explicit_training_fields", "sources": [
                {"path": str(resolve(p)), "sha256": sha256_file(resolve(p))}
                for p in support["reference_npy"]]}}} if profile else {})
        save_reference_field_input_model(artifact, unconditional, source_metadata=source_metadata,
                                         **support_metadata)
        diagnostics = TrainingPatchCompatibilityDiagnostics(profile) if profile else None
        sampler = GaussianCoordinatePermeabilitySampler(field_map=field_map, n_samples=count, batch_size=1, seed=seed)
        ensemble = np.lib.format.open_memmap(directory / "generated_fields.npy", mode="w+",
                                            dtype=np.float32, shape=(count, *reference.shape))
        coarse = []
        for index, batch in enumerate(sampler):
            ensemble[index] = batch[0]
            coarse.append(coarsen_reference(batch[0], factor))
            if diagnostics:
                diagnostics.update(batch)
        ensemble.flush()
        fidelity = compare_reference_ensemble(coarsen_reference(reference, factor), np.asarray(coarse),
                                              cell_size_m=cell * factor)
        patch = diagnostics.finalize() if diagnostics else None
        # Descriptor-envelope comparison is descriptive, not a calibrated acceptance test.
        outside = ([key for key, value in patch["metrics"].items()
                    if value["fraction_generated_patches_outside_training_envelope"] > 0] if patch else [])
        fidelity["training_patch_support"] = patch
        fidelity["compatibility_assessment"] = {
            "status": ("descriptor_deviations_observed" if outside else "within_sampled_descriptor_envelopes") if patch else "unassessed_training_support",
            "interpretation": "Inspect patch descriptor deviations; no calibrated universal threshold or guarantee of in-distribution predictions",
            "sample_filtering": None, "flagged_metrics": outside}
        (directory / "diagnostics.json").write_text(json.dumps(fidelity, indent=2, allow_nan=False), encoding="utf-8")
        manifest = scenario.manifest()
        variance = unconditional.prior.pointwise_log10_variance()
        manifest.update(source_metadata=source_metadata, cell_size_m=cell, seed=seed, n_samples=count,
                        dimension=field_map.dimension, retained_energy=unconditional.prior.retained_energy_fraction,
                        retained_log10_variance={"min": float(variance.min()), "mean": float(variance.mean()), "max": float(variance.max())},
                        input_model_sha256=sha256_file(artifact), input_model=str(artifact))
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
