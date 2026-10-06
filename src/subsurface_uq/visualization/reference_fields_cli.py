"""Export selected permeability input realizations and reference agreement PNGs.

Replay a completed temperature pilot with its exact Gaussian coordinate design,
or draw a subset from one portable reference-field input artifact. No LGCNN
prediction is rerun. Samples are selected by index, never agreement metrics.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from ..io import sha256_file, runtime_versions
from ..sampling.input_model import load_conditional_kl_input_model
from ..sampling.reference_field import ReferenceFieldLogGaussianPermeabilityMap
from .reference_fields import reference_field_agreement, plot_reference_field_comparison


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _coordinates(dimension, count, seed, method):
    if method == "MC":
        return np.random.default_rng(seed).normal(size=(count, dimension))
    if method != "RQMC" or count & (count - 1):
        raise ValueError("RQMC requires a power-of-two design size")
    uniforms = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(int(np.log2(count)))
    return ndtri(np.clip(uniforms, np.nextafter(0., 1.), np.nextafter(1., 0.)))


def pilot_preview_sources(pilot_output, *, reference_run=None, scenario_ids=None, include_kl=False):
    """Read frozen coordinate designs and expected model-input hashes."""
    root = Path(pilot_output).resolve()
    context_path, summary_path = root / "run_context.json", root / "summary.json"
    recorded = json.loads(context_path.read_text(encoding="utf-8"))
    context = recorded["context"]
    if sha256(_canonical(context).encode()).hexdigest() != recorded["context_id"]:
        raise ValueError("pilot run-context checksum mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["context_id"] != recorded["context_id"]:
        raise ValueError("pilot summary belongs to a different run context")
    artifacts = {row["scenario_id"]: row for row in context["artifacts"]}
    paired_path = root / "qoi_truncation.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired["context_id"] != recorded["context_id"]:
        raise ValueError("paired KL record belongs to a different run context")
    paired_ids = set(paired["scenario_ids"])
    paired_dimension = summary["paired_anchor_reuse"]["coordinate_dimension"]
    selected = set(scenario_ids or [])
    known = {row["scenario_id"] for row in summary["scenarios"]}
    if selected - known:
        raise ValueError(f"unknown scenario IDs: {sorted(selected - known)}")
    sources = []
    for row in summary["scenarios"]:
        sid = row["scenario_id"]
        if selected and sid not in selected:
            continue
        if not selected and not include_kl and row["role"] != "primary":
            continue
        if reference_run and row["config"]["reference_run"] != reference_run:
            continue
        artifact = artifacts[sid]["input_model"]
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"pilot input-artifact checksum mismatch: {sid}")
        diagnostics_path = root / sid / "sample_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if any(r["context_id"] != recorded["context_id"] for r in diagnostics):
            raise ValueError(f"sample diagnostics belong to a different pilot context: {sid}")
        indices = [r["sample_index"] for r in diagnostics]
        if sorted(indices) != list(range(context["n_samples"])):
            raise ValueError(f"missing or duplicated pilot sample indices: {sid}")
        sources.append({
            "scenario_id": sid, "input_model": str(path),
            "input_model_sha256": artifact["sha256"],
            "config": row["config"], "method": context["method"],
            "seed": context["seed"], "n_samples": context["n_samples"],
            "coordinate_dimension": paired_dimension if sid in paired_ids else row["dimension"],
            "expected_fields": {r["sample_index"]: r["physical_field_sha256"] for r in diagnostics},
            "pilot_context_id": recorded["context_id"],
            "source_diagnostics_sha256": sha256_file(diagnostics_path),
        })
    if not sources:
        raise ValueError("no pilot scenarios match the selection")
    return sources


def export_reference_previews(sources, output_dir, *, sample_indices=(0, 1), save_arrays=False):
    """Store only a predetermined subset; use common scales per reference/grid.

    Pilot hashes must match bitwise before any PNG is presented as a queried
    LGCNN input. Optional NPY files preserve the exact float32 physical inputs;
    PNGs are illustrations. Temporary selected arrays bound memory usage.
    """
    indices = list(sample_indices)
    if (not indices or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("sample_indices must be distinct non-negative integers (zero-based)")
    if not sources or len({s["scenario_id"] for s in sources}) != len(sources):
        raise ValueError("distinct nonempty scenario sources required")
    for source in sources:
        if source["n_samples"] < 1 or max(indices) >= source["n_samples"] or source["seed"] < 0:
            raise ValueError("sample index outside the original design or invalid design size/seed")
        sid = source["scenario_id"]
        if not sid or sid in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in sid):
            raise ValueError("scenario IDs must be safe directory names")
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("preview output is nonempty; choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    records, groups = [], {}
    with TemporaryDirectory(prefix=".selected-fields-", dir=root) as temporary:
        temp = Path(temporary)
        for source in sources:
            path = Path(source["input_model"]).resolve()
            if source.get("input_model_sha256") and sha256_file(path) != source["input_model_sha256"]:
                raise ValueError("input artifact changed after preview-source selection")
            model = load_conditional_kl_input_model(path)
            field_map = model.conditional
            if not isinstance(field_map, ReferenceFieldLogGaussianPermeabilityMap):
                raise ValueError("previews require a reference-centered permeability artifact")
            dimension = source.get("coordinate_dimension")
            dimension = field_map.dimension if dimension is None else dimension
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < field_map.dimension:
                raise ValueError("coordinate design is smaller than the selected map")
            coordinates = _coordinates(dimension, source["n_samples"], source["seed"], source["method"])
            reference = field_map.reference_permeability
            cell = np.asarray(field_map.domain_size_m) / np.asarray(reference.shape)
            if not np.allclose(cell, cell[0]):
                raise ValueError("preview currently requires uniform square cells")
            reference_sha = sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()
            group_id = sha256(_canonical({"reference_sha256": reference_sha, "shape": list(reference.shape),
                                          "cell_size_m": float(cell[0])}).encode()).hexdigest()[:20]
            if group_id not in groups:
                ref_path = (root if save_arrays else temp) / f"reference-{group_id}.npy"
                np.save(ref_path, reference, allow_pickle=False)
                groups[group_id] = {"reference": ref_path, "low": np.inf, "high": -np.inf, "residual": 0.}
            directory = root / source["scenario_id"]
            directory.mkdir()
            cfg = source.get("config") or (model.payload.get("scenario") or {}).get("config", {})
            artifact_config = (model.payload.get("scenario") or {}).get("config")
            if artifact_config is not None and cfg != artifact_config:
                raise ValueError("preview scenario config disagrees with the input artifact")
            run_label = cfg.get("reference_run", model.payload.get("source_metadata", {}).get("reference_run", "reference"))
            label = f"{run_label} | {field_map.center} centering"
            if cfg:
                truncation = f"energy={cfg['energy_threshold']:g}" if cfg.get("energy_threshold") is not None else f"modes={field_map.dimension}"
                sigma = cfg.get("sigma_R", field_map.prior.std_log10_k)
                covariance = cfg.get("covariance_family", field_map.prior.covariance_model)
                label += f" | sigma_R={sigma:g} | {covariance} | {truncation}"
            for index in indices:
                field = np.ascontiguousarray(field_map.map_coordinates(coordinates[index, :field_map.dimension]), dtype=np.float32)
                digest = sha256(field.tobytes()).hexdigest()
                expected = source.get("expected_fields")
                if expected is not None and digest != expected[index]:
                    raise ValueError(f"regenerated field does not match the queried pilot input: {source['scenario_id']} sample {index}")
                agreement = reference_field_agreement(reference, field)
                group = groups[group_id]
                group["low"] = min(group["low"], *agreement["reference_log10_range"], *agreement["generated_log10_range"])
                group["high"] = max(group["high"], *agreement["reference_log10_range"], *agreement["generated_log10_range"])
                group["residual"] = max(group["residual"], agreement["log10_max_abs_difference"])
                field_path = (directory if save_arrays else temp) / f"{source['scenario_id']}-sample-{index:04d}.npy"
                np.save(field_path, field, allow_pickle=False)
                records.append({"scenario_id": source["scenario_id"], "sample_index": index,
                                "input_model_sha256": sha256_file(path), "config": cfg,
                                "permeability_convention": model.payload.get("source_metadata", {}).get("permeability_convention", "artifact-declared"),
                                "sampling": {key: source[key] for key in ("method", "seed", "n_samples")},
                                "coordinate_dimension": dimension, "map_dimension": field_map.dimension,
                                "pilot_context_id": source.get("pilot_context_id"),
                                "pilot_input_hash_verified": expected is not None,
                                "source_diagnostics_sha256": source.get("source_diagnostics_sha256"),
                                "physical_field_sha256": digest, "reference_array_sha256": reference_sha,
                                "group_id": group_id, "cell_size_m": float(cell[0]),
                                "agreement": agreement, "_field": field_path, "_label": label})
                print(f"Reconstructed {source['scenario_id']} sample {index}", flush=True)
            del model, field_map, reference, field, coordinates
        for record in records:
            group = groups[record["group_id"]]
            low, high = group["low"], group["high"]
            limits = (low - .05, high + .05) if low == high else (low, high)
            residual_limit = max(group["residual"], 1e-6)
            destination = root / record["scenario_id"] / f"sample-{record['sample_index']:04d}.comparison.png"
            reference = np.load(group["reference"], allow_pickle=False)
            field = np.load(record["_field"], allow_pickle=False)
            plot_reference_field_comparison(reference, field, destination,
                cell_size_m=record["cell_size_m"], title=f"{record['_label']}\nGenerated input sample {record['sample_index']} (zero-based)",
                log10_limits=limits, residual_limit=residual_limit, agreement=record["agreement"])
            record.update(png=str(destination.relative_to(root)), png_sha256=sha256_file(destination),
                          generated_npy=str(record["_field"].relative_to(root)) if save_arrays else None,
                          reference_npy=str(group["reference"].relative_to(root)) if save_arrays else None,
                          shared_log10_limits=list(limits), shared_log10_ratio_limit=residual_limit)
            record.pop("_field")
            record.pop("_label")
        report = {"schema_version": 1, "sample_indices_zero_based": indices, "save_arrays": save_arrays,
                  "selection": "explicit indices only; no agreement-based selection, filtering or rejection",
                  "scales": "shared full-range permeability and symmetric log-ratio scales per identical reference/grid within this export",
                  "output_type": "generated permeability inputs; not LGCNN temperature outputs",
                  "runtime": runtime_versions(("numpy", "scipy", "matplotlib")), "samples": records}
        (root / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pilot-output", help="Completed temperature pilot directory; reproduce and hash-check its exact queried inputs")
    source.add_argument("--input-model", help="One reference-field artifact for a standalone coordinate design")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1], help="Zero-based indices, chosen before inspecting agreement")
    parser.add_argument("--save-arrays", action="store_true", help="Also retain selected float32 NPY inputs and their nominal reference")
    parser.add_argument("--reference-run", help="Pilot scenario filter, e.g. RUN_2")
    parser.add_argument("--scenario-ids", nargs="+", help="Explicit pilot scenario IDs; may include KL variants")
    parser.add_argument("--include-kl", action="store_true", help="Include extra KL variants when selecting pilot scenarios by reference RUN")
    parser.add_argument("--n-samples", type=int, help="Original full standalone design size, not the number of exported images")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=("MC", "RQMC"))
    parser.add_argument("--coordinate-dimension", type=int, help="Standalone shared-design dimension; defaults to the map dimension")
    args = parser.parse_args(argv)
    if args.pilot_output:
        if any(v is not None for v in (args.n_samples, args.seed, args.method, args.coordinate_dimension)):
            parser.error("pilot coordinate design is frozen; do not override seed/method/design size/dimension")
        sources = pilot_preview_sources(args.pilot_output, reference_run=args.reference_run,
                                        scenario_ids=args.scenario_ids, include_kl=args.include_kl)
    else:
        if args.reference_run or args.scenario_ids or args.include_kl:
            parser.error("pilot scenario filters require --pilot-output")
        if args.n_samples is None or args.seed is None or args.method is None:
            parser.error("standalone mode requires explicit --n-samples, --seed and --method")
        path = Path(args.input_model).resolve()
        sources = [{"scenario_id": "standalone", "input_model": str(path),
                    "input_model_sha256": sha256_file(path), "n_samples": args.n_samples,
                    "seed": args.seed, "method": args.method, "coordinate_dimension": args.coordinate_dimension}]
    report = export_reference_previews(sources, args.output_dir, sample_indices=args.sample_indices,
                                      save_arrays=args.save_arrays)
    print(f"Saved {len(report['samples'])} input comparison PNGs to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
