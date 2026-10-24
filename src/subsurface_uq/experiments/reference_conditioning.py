"""Study synthetic observation-location priors on separate fixed references.

The observation designs, fixed-design conditional Gaussian ensembles and
across-design mixture diagnostics have separate identities and output records.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from ..sampling.reference_conditioning import (
    ConditionalReferenceFFTSampler, LocationPriorSpec, draw_reference_measurements,
)
from ..sampling.reference_coherence import native_short_range_features, reference_ensemble_coherence
from ..sampling.training_compatibility import permeability_field_features
from ..visualization.reference_fields import reference_field_agreement


_KEYS = {
    "references", "covariance", "covariance_by_reference", "cell_size_m", "location_priors",
    "n_measurements", "n_designs", "n_samples", "measurement_mode", "observation_std_log10_k",
    "mean_mode", "seed", "workers", "chunk_size", "retained_sample_indices", "location_weights",
    "diagnostic_factor", "max_lag_cells", "permeability_units",
}
_DEFAULTS = {
    "cell_size_m": 5., "n_measurements": 16, "n_designs": 4, "n_samples": 8,
    "measurement_mode": "exact", "observation_std_log10_k": 0., "mean_mode": "reference_median",
    "seed": 4901, "workers": 1, "chunk_size": 8192, "retained_sample_indices": [0],
    "diagnostic_factor": 8, "max_lag_cells": 24, "permeability_units": "m2",
}


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _identity(value, prefix):
    encoded = json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return prefix + sha256(encoded).hexdigest()[:20]


def _name(value, what):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value) is None:
        raise ValueError(f"{what} must be a short alphanumeric name without path separators")
    return value


def _positive_integer(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")
    return value


def _relative_path(raw, root):
    if not isinstance(raw, str) or not raw:
        raise ValueError("reference and weight paths must be nonempty strings")
    path = Path(raw).expanduser()
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    """Validate scientific choices and resolve paths before creating output."""
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("conditional-study configuration must be a mapping")
    if not set(raw) <= _KEYS:
        raise ValueError(f"unknown conditional-study configuration keys: {sorted(set(raw) - _KEYS)}")
    cfg = {**_DEFAULTS, **raw}
    refs = cfg.get("references")
    priors = cfg.get("location_priors")
    if not isinstance(refs, dict) or not refs:
        raise ValueError("references must be a nonempty name-to-NPY mapping")
    if not isinstance(priors, dict) or not priors:
        raise ValueError("location_priors must explicitly declare one or more location policies")
    for key in ("n_measurements", "n_designs", "n_samples", "workers", "chunk_size", "diagnostic_factor", "max_lag_cells"):
        _positive_integer(cfg[key], key)
    _positive_integer(cfg["seed"], "seed", zero=True)
    if (isinstance(cfg["cell_size_m"], bool) or not np.isfinite(cfg["cell_size_m"]) or cfg["cell_size_m"] <= 0):
        raise ValueError("cell_size_m must be positive and finite")
    if cfg["permeability_units"] != "m2":
        raise ValueError("this LGCNN workflow requires reference permeability in m2; convert hydraulic conductivity explicitly first")
    if cfg["mean_mode"] not in {"reference_median", "global_log_mean"}:
        raise ValueError("mean_mode must be reference_median or global_log_mean")
    noise = cfg["observation_std_log10_k"]
    if isinstance(noise, bool) or not np.isfinite(noise) or noise < 0:
        raise ValueError("observation_std_log10_k must be finite and nonnegative")
    if cfg["measurement_mode"] not in {"exact", "noisy"}:
        raise ValueError("measurement_mode must be exact or noisy")
    if ((cfg["measurement_mode"] == "exact" and noise != 0) or
            (cfg["measurement_mode"] == "noisy" and noise <= 0)):
        raise ValueError("exact readings require zero noise; noisy readings require positive noise")
    retained = cfg["retained_sample_indices"]
    if not isinstance(retained, list) or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < cfg["n_samples"] for v in retained):
        raise ValueError("retained_sample_indices must be valid zero-based sample indices")
    if len(set(retained)) != len(retained):
        raise ValueError("retained_sample_indices must not contain duplicates")
    covariance = cfg.get("covariance", {})
    by_reference = cfg.get("covariance_by_reference", {})
    if not isinstance(covariance, dict) or not isinstance(by_reference, dict) or not set(by_reference) <= set(refs):
        raise ValueError("covariance and covariance_by_reference must be mappings; overrides must name supplied references")
    policies = {_name(name, "prior name"): LocationPriorSpec(**parameters) for name, parameters in priors.items()}
    weights = cfg.get("location_weights")
    if isinstance(weights, dict) and not set(weights) <= set(refs):
        raise ValueError("location_weights contains an unknown reference")
    prepared = {}
    for name, reference_path in refs.items():
        _name(name, "reference name")
        ref_path = _relative_path(reference_path, source.parent)
        reference = np.load(ref_path, allow_pickle=False)
        if (np.iscomplexobj(reference) or reference.ndim != 2 or min(reference.shape) < 3
                or not np.all(np.isfinite(reference)) or np.any(reference <= 0)):
            raise ValueError(f"{name}: reference must contain positive finite real 2-D permeability")
        factor = cfg["diagnostic_factor"]
        if min(reference[::factor, ::factor].shape) < 3:
            raise ValueError(f"{name}: diagnostic_factor leaves fewer than 3x3 cells")
        override = by_reference.get(name, {})
        if not isinstance(override, dict):
            raise ValueError("each covariance override must be a mapping")
        spec = GSToolsReferenceSpec(**{**covariance, **override})
        if spec.center != "median":
            raise ValueError("conditional workflow requires covariance.center=median; mean_mode declares the prior log-mean")
        weight_path = weights.get(name) if isinstance(weights, dict) else weights
        weight_array = None
        if weight_path is not None:
            weight_path = _relative_path(weight_path, source.parent)
            weight_array = np.load(weight_path, allow_pickle=False)
            if (np.iscomplexobj(weight_array) or weight_array.shape != reference.shape or
                    not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0) or not np.any(weight_array > 0)):
                raise ValueError("location weight maps must be finite nonnegative reference-shaped arrays with positive support")
        prepared[name] = {"path": ref_path, "reference": reference, "sha256": sha256_file(ref_path),
                          "covariance": spec, "weights": weight_array,
                          "weights_path": weight_path,
                          "weights_sha256": None if weight_path is None else sha256_file(weight_path)}
    return source, cfg, policies, prepared


def _seed(root, *labels):
    # Name-derived spawn keys make unchanged cases insensitive to configuration order.
    key = sha256(json.dumps(labels, separators=(",", ":")).encode()).digest()
    entropy = [root, *np.frombuffer(key, dtype="<u4").tolist()]
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])


def _patch_descriptors(field, cell_size):
    h, w = field.shape
    box = min(1280, h, w)
    positions = sorted({(0, 0), (0, w-box), (h-box, 0), (h-box, w-box), ((h-box)//2, (w-box)//2)})
    return [{"row_start": r, "column_start": c, "box_size_cells": box,
             "features": permeability_field_features(field[r:r+box, c:c+box], cell_size_m=cell_size,
                                                      spatial_stride=max(1, box//96))} for r,c in positions]


def _write_observations(path, observations, cell_size):
    import csv
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        columns = ["step", "row", "column", "local_row_m", "local_column_m", "reference_log10_k",
                   "observed_log10_k", "observation_std_log10_k", "selection_probability"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, (r,c) in enumerate(observations.indices):
            writer.writerow(dict(zip(columns, [i, int(r), int(c), (r+.5)*cell_size, (c+.5)*cell_size,
                                                observations.reference_log10_k[i], observations.observed_log10_k[i],
                                                observations.observation_std_log10_k[i], observations.selection_probabilities[i]])))


def _plot_design(path, reference, mean, variance, sample_log, observations, *, cell_size, title):
    logs = np.log10(reference)
    height, width = reference.shape
    extent = (0, width*cell_size/1000, 0, height*cell_size/1000)
    stride = max(1, int(np.ceil(max(reference.shape)/640)))
    show = lambda value: value[::stride, ::stride]
    lo, hi = min(logs.min(), mean.min(), sample_log.min()), max(logs.max(), mean.max(), sample_log.max())
    if lo == hi:
        lo, hi = lo-.05, hi+.05
    fig, axes = plt.subplots(2,3,figsize=(14,8.2),layout="constrained")
    for ax, data, label in zip(axes.flat[:3], (logs,mean,sample_log),
                               ("Fixed reference", "Conditional log-mean / log-median", "First conditional realization")):
        image = ax.imshow(show(data),origin="lower",extent=extent,cmap="viridis",vmin=lo,vmax=hi)
        ax.set_title(label)
        fig.colorbar(image,ax=ax,shrink=.8,label="log10 permeability [m²]")
    delta = sample_log-logs
    limit = max(float(abs(delta).max()),1e-7)
    image = axes[1,0].imshow(show(delta),origin="lower",extent=extent,cmap="RdBu_r",vmin=-limit,vmax=limit)
    axes[1,0].set_title("Realization minus reference")
    fig.colorbar(image,ax=axes[1,0],shrink=.8,label="log10(K/Kref)")
    image = axes[1,1].imshow(show(np.sqrt(variance)),origin="lower",extent=extent,cmap="magma",vmin=0)
    axes[1,1].set_title("Conditional log-space standard deviation")
    fig.colorbar(image,ax=axes[1,1],shrink=.8,label="log10 standard deviation")
    axes[1,2].imshow(show(logs),origin="lower",extent=extent,cmap="Greys",vmin=lo,vmax=hi)
    coords = (observations.indices+.5)*cell_size/1000
    image = axes[1,2].scatter(coords[:,1],coords[:,0],c=np.arange(len(coords)),cmap="plasma",s=32,edgecolor="black")
    axes[1,2].set_title("Synthetic measurement draw order")
    fig.colorbar(image,ax=axes[1,2],shrink=.8,label="Selection step")
    for ax in axes.flat:
        ax.set(xlabel="Array-column distance [km]",ylabel="Array-row distance [km]")
        ax.set_aspect("equal")
    fig.suptitle(title)
    try:
        fig.savefig(path,dpi=160)
    finally:
        plt.close(fig)


def run_study(config_path, output_dir):
    """Execute frozen synthetic designs and conditional field ensembles."""
    source, cfg, policies, references = load_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a fresh directory to preserve the frozen study")
    output.mkdir(parents=True)
    package = Path(__file__).parents[1]
    implementation_files = [Path(__file__),package/"sampling/reference_conditioning.py",
                            package/"sampling/gstools_reference.py"]
    manifest = {"schema_version":1,"config_path":str(source),"config_sha256":sha256_file(source),
                "configuration":cfg,"code_commit":git_head(Path(__file__).parents[3]),
                "runtime":runtime_versions(("numpy","scipy","gstools","matplotlib")),
                "implementation_sha256":{str(p.relative_to(package)):sha256_file(p) for p in implementation_files},
                "synthetic_observations":True,"reference_pooling":None,"sample_filtering":None,
                "prior_weights":None,"references":{},"status":"running"}
    _write(output/"manifest.json",manifest)
    all_summaries = {}
    for reference_name, item in sorted(references.items()):
        reference, spec = item["reference"], item["covariance"]
        reference_id = _identity({"reference_sha256":item["sha256"],"covariance":asdict(spec),
                                  "mean_mode":cfg["mean_mode"],"cell_size_m":cfg["cell_size_m"]},"reference-")
        reference_dir = output/reference_name
        reference_dir.mkdir()
        _write(reference_dir/"reference_descriptors.json",{
            "native_short_range":native_short_range_features(reference,cell_size_m=cfg["cell_size_m"]),
            "patch_descriptors":_patch_descriptors(reference,cfg["cell_size_m"]),
            "interpretation":"fixed nominal-reference descriptors; training membership is unverified"})
        manifest["references"][reference_name] = {"reference_id":reference_id,"path":str(item["path"]),
            "sha256":item["sha256"],"shape":list(reference.shape),"covariance":asdict(spec),
            "location_weights_path":None if item["weights_path"] is None else str(item["weights_path"]),
            "location_weights_sha256":item["weights_sha256"]}
        _write(output/"manifest.json",manifest)
        prior_sampler = GSToolsReferenceFFTSampler(spec,reference.shape,cell_size_m=cfg["cell_size_m"],workers=cfg["workers"])
        _write(reference_dir/"prior_sampler.json",prior_sampler.metadata)
        reference_summaries = {}
        ref_log = np.log10(reference.astype(float))
        probe_axes = [np.unique(np.linspace(0,n-1,min(5,n),dtype=int)) for n in reference.shape]
        probe_grid = np.meshgrid(*probe_axes,indexing="ij")
        probes = np.column_stack([v.ravel() for v in probe_grid])
        for prior_name, policy in sorted(policies.items()):
            directory = reference_dir/prior_name
            directory.mkdir()
            sum_variance = np.zeros(reference.shape,float)
            mean_of_means = np.zeros(reference.shape,float)
            m2_means = np.zeros(reference.shape,float)
            designs = []
            for design_index in range(cfg["n_designs"]):
                seed_labels = (reference_name,prior_name,design_index)
                location_seed = _seed(cfg["seed"],*seed_labels,"locations")
                observation_seed = _seed(cfg["seed"],*seed_labels,"observations")
                observations = draw_reference_measurements(reference,covariance=spec,n_measurements=cfg["n_measurements"],
                    location_seed=location_seed,observation_seed=observation_seed,cell_size_m=cfg["cell_size_m"],
                    location_prior=policy,location_weights=item["weights"],measurement_mode=cfg["measurement_mode"],
                    observation_std_log10_k=cfg["observation_std_log10_k"])
                design_id = _identity({"reference_id":reference_id,"prior":asdict(policy),
                    "weights_sha256":item["weights_sha256"],"indices":observations.indices,
                    "observed_log10_k":observations.observed_log10_k,"std":observations.observation_std_log10_k,
                    "measurement_mode":cfg["measurement_mode"]},"design-")
                design_dir = directory/f"{design_index:03d}-{design_id}"
                design_dir.mkdir()
                _write_observations(design_dir/"measurements.csv",observations,cfg["cell_size_m"])
                sampler = ConditionalReferenceFFTSampler(reference,spec,observations,cell_size_m=cfg["cell_size_m"],
                    mean_mode=cfg["mean_mode"],workers=cfg["workers"],chunk_size=cfg["chunk_size"],prior_sampler=prior_sampler)
                mean, variance = sampler.posterior_moments()
                sum_variance += variance
                difference = mean-mean_of_means
                mean_of_means += difference/(design_index+1)
                m2_means += difference*(mean-mean_of_means)
                np.save(design_dir/"posterior_log10_mean.npy",mean.astype(np.float32))
                np.save(design_dir/"posterior_log10_variance.npy",variance.astype(np.float32))
                np.save(design_dir/"posterior_log10_std.npy",np.sqrt(variance).astype(np.float32))
                analytic_probe_mean, analytic_probe_covariance = sampler.posterior_at_indices(probes)
                running_mean, running_m2 = np.zeros(reference.shape,float),np.zeros(reference.shape,float)
                coarse, probe_samples, records = [],[],[]
                first_log = None
                for sample_index in range(cfg["n_samples"]):
                    field_seed = _seed(cfg["seed"],*seed_labels,"field",sample_index)
                    logs = np.asarray(sampler.sample_log10(field_seed),dtype=float)
                    with np.errstate(over="raise",invalid="raise",under="ignore"):
                        field = np.power(10.,logs).astype(np.float32)
                    if not np.all(np.isfinite(field)) or np.any(field<=0):
                        raise ValueError("conditional field is outside positive finite float32 permeability range")
                    delta = logs-running_mean
                    running_mean += delta/(sample_index+1)
                    running_m2 += delta*(logs-running_mean)
                    coarse.append(field[::cfg["diagnostic_factor"],::cfg["diagnostic_factor"]].copy())
                    probe_samples.append(logs[tuple(probes.T)])
                    record = {"sample_index":sample_index,"field_seed":field_seed,
                              "reference_agreement":reference_field_agreement(reference,field),
                              "native_short_range":native_short_range_features(field,cell_size_m=cfg["cell_size_m"]),
                              "patch_descriptors":_patch_descriptors(field,cfg["cell_size_m"])}
                    if sample_index in cfg["retained_sample_indices"]:
                        name = f"sample-{sample_index:04d}.npy"
                        np.save(design_dir/name,field)
                        record.update(retained_field=name,sha256=sha256_file(design_dir/name))
                    records.append(record)
                    if first_log is None:
                        first_log=logs.copy()
                empirical_variance = running_m2/(cfg["n_samples"]-1) if cfg["n_samples"]>1 else None
                coherence = reference_ensemble_coherence(reference[::cfg["diagnostic_factor"],::cfg["diagnostic_factor"]],
                    np.asarray(coarse),cell_size_m=cfg["cell_size_m"]*cfg["diagnostic_factor"],max_lag_cells=cfg["max_lag_cells"])
                _write(design_dir/"reference_diagnostics.json",coherence)
                _plot_design(design_dir/"conditional_fields.png",reference,mean,variance,first_log,observations,
                             cell_size=cfg["cell_size_m"],title=f"{reference_name} | {prior_name} | design {design_index}")
                probe_samples=np.asarray(probe_samples)
                np.save(design_dir/"log10_probe_samples.npy",probe_samples)
                design = {"design_id":design_id,"design_index":design_index,"location_seed":location_seed,
                    "observation_seed":observation_seed,"location_prior":asdict(policy),
                    "observations":observations.metadata,"conditional_sampler":sampler.metadata,
                    "mean_mode":cfg["mean_mode"],"measurement_mode":cfg["measurement_mode"],"sample_count":cfg["n_samples"],
                    "diagnostic_support":{"method":"point subsampling only for descriptive diagnostics",
                        "factor":cfg["diagnostic_factor"],"native_field_shape":list(reference.shape),"training_support":"unassessed"},
                    "posterior":{"spatial_mean_variance_log10":float(variance.mean()),
                        "mean_log10_rmse_to_reference":float(np.sqrt(np.mean((mean-ref_log)**2))),
                        "variance_reduction_fraction":float(1-variance.mean()/spec.sigma_R**2)},
                    "monte_carlo":{"mean_field_rmse_to_analytic_log10_mean":float(np.sqrt(np.mean((running_mean-mean)**2))),
                        "mean_standard_error_rms_log10":None if empirical_variance is None else float(np.sqrt(np.mean(empirical_variance)/cfg["n_samples"])),
                        "spatial_mean_empirical_variance_log10":None if empirical_variance is None else float(empirical_variance.mean()),
                        "interpretation":"finite-ensemble numerical error; does not reduce geological uncertainty"},
                    "probes":{"indices":probes,"analytic_log10_mean":analytic_probe_mean,"analytic_log10_covariance":analytic_probe_covariance},
                    "records":records,"sample_filtering":None,"interpretation":"one fixed synthetic location/reading design; conditional Gaussian log-field"}
                _write(design_dir/"design.json",design)
                designs.append({"design_id":design_id,"directory":design_dir.name,"posterior":design["posterior"],"monte_carlo":design["monte_carlo"]})
                print(f"{reference_name} / {prior_name}: design {design_index+1}/{cfg['n_designs']} complete",flush=True)
                del sampler
            within = sum_variance/cfg["n_designs"]
            between = m2_means/cfg["n_designs"]
            np.save(directory/"within_design_variance_log10.npy",within.astype(np.float32))
            np.save(directory/"between_design_variance_log10.npy",between.astype(np.float32))
            np.save(directory/"design_mixture_variance_log10.npy",(within+between).astype(np.float32))
            np.save(directory/"design_mixture_log10_mean.npy",mean_of_means.astype(np.float32))
            summary={"reference_id":reference_id,"location_prior":asdict(policy),"design_count":cfg["n_designs"],
                "design_weight_assumption":"equal weights over these iid replicates of the declared synthetic design/reading policy; empirical design-mixture diagnostic, not inferred geological probabilities",
                "within_design_mean_variance_log10":float(within.mean()),"between_design_mean_variance_log10":float(between.mean()),
                "design_mixture_mean_variance_log10":float((within+between).mean()),
                "decomposition":"Var(Y)=E_design[Var(Y|design)]+Var_design(E[Y|design]); pointwise log-space variance averaged spatially",
                "between_design_interpretation":"location variability and, in noisy mode, simulated-reading variability; not separated by this experiment",
                "mixture_distribution":"generally non-Gaussian; every fixed-design log-field remains Gaussian",
                "reference_pooling":None,"prior_pooling":None,"designs":designs}
            _write(directory/"summary.json",summary)
            reference_summaries[prior_name]=summary
        all_summaries[reference_name]=reference_summaries
        del prior_sampler
    manifest["status"]="complete"
    _write(output/"summary.json",{"schema_version":1,"references":all_summaries,"reference_pooling":None,"prior_pooling":None})
    _write(output/"manifest.json",manifest)
    lines=["# Synthetic reference-conditioning study","","Each RUN and location prior remains separate. All readings are synthetic.",
           "No field filtering, inferred scenario probabilities, surrogate query or training-support certification is performed.","",
           "| Reference | Location prior | Within-design mean variance | Between-design mean variance | Mixture mean variance |",
           "|---|---|---:|---:|---:|"]
    for name, priors in all_summaries.items():
        for prior, summary in priors.items():
            lines.append(f"| {name} | {prior} | {summary['within_design_mean_variance_log10']:.6g} | {summary['between_design_mean_variance_log10']:.6g} | {summary['design_mixture_mean_variance_log10']:.6g} |")
    lines += ["","All variances are in log10-permeability squared. Equal replicate weights are an explicit design-mixture summary assumption.",
              "With reference_median and exact reference readings, every design retains the reference log-mean: between-design mean variance is zero although conditional covariance changes.",
              "One- or few-design/sample pilots do not establish convergence. Inspect each design's conditional_fields.png, measurements.csv, design.json and reference_diagnostics.json."]
    (output/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return {"manifest":manifest,"references":all_summaries}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,help="YAML configuration; input paths are relative to this file")
    parser.add_argument("--output-dir",required=True,help="new output directory; existing directories are refused")
    args=parser.parse_args(argv)
    run_study(args.config,args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
