"""Nested synthetic-measurement density study through a frozen temperature model.

Independent MC within fixed designs; repeated designs remain separate. This is
not RQMC, neural-weight uncertainty, a real-well calibration or an RQ5 EVR ranker.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..rq1.analysis import (global_uncertainty_metrics, prefix_field_convergence,
                            scalar_summary, write_temperature_field_products)
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler
from ..sampling.reference_conditioning import ConditionalReferenceFFTSampler, draw_reference_measurements
from ..sampling.reference_coherence import native_short_range_features, reference_ensemble_coherence
from ..statistics.online import OnlineFieldStatistics
from ..statistics.temperature_risk import scalar_temperature_qois, summarize_risk_store, validate_output_grid
from ..surrogates.frozen_temperature import load_frozen_temperature_model
from ..visualization.reference_fields import reference_field_agreement
from .reference_conditioning import (_identity, _positive_integer, _relative_path, _seed,
                                     _patch_descriptors, _write, _write_observations, load_config as load_conditioning_config)


def load_config(path):
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    allowed = {"conditioning_config", "model_config", "model_asset_root", "measurement_counts",
               "sample_budgets", "risk_thresholds_C", "threshold_comparison", "quantile_chunk_rows",
               "retain_temperature_ensemble", "torch_threads"}
    if not isinstance(raw, dict) or not set(raw) <= allowed:
        raise ValueError("temperature study must be a mapping with known configuration keys")
    cfg = {"threshold_comparison": ">", "quantile_chunk_rows": 16,
           "retain_temperature_ensemble": False, "torch_threads": 8, **raw}
    for key in ("conditioning_config", "model_config"):
        cfg[key] = _relative_path(cfg.get(key), source.parent)
    cfg["model_asset_root"] = (_relative_path(cfg["model_asset_root"], source.parent)
                               if cfg.get("model_asset_root") is not None else None)
    for name, minimum in (("measurement_counts", 0), ("sample_budgets", 2)):
        values = cfg.get(name)
        if (not isinstance(values, list) or not values or
                any(isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values)
                or values != sorted(set(values))):
            raise ValueError(f"{name} must be a strictly increasing integer list with minimum {minimum}")
    if cfg["measurement_counts"][0] != 0 or len(cfg["measurement_counts"]) < 2:
        raise ValueError("measurement_counts must include zero and a positive conditioning budget")
    thresholds = cfg.get("risk_thresholds_C")
    if (not isinstance(thresholds, list) or not thresholds or
            any(isinstance(v, bool) or not isinstance(v, (int,float)) or not np.isfinite(v) or v <= 0 for v in thresholds)
            or len(set(thresholds)) != len(thresholds)):
        raise ValueError("risk_thresholds_C must explicitly declare distinct positive finite warming thresholds")
    if cfg["threshold_comparison"] not in {">", ">="}:
        raise ValueError("threshold_comparison must be > or >=")
    for key in ("quantile_chunk_rows", "torch_threads"):
        _positive_integer(cfg[key], key)
    if not isinstance(cfg["retain_temperature_ensemble"], bool):
        raise ValueError("retain_temperature_ensemble must be boolean")
    base_source, base, policies, references = load_conditioning_config(cfg["conditioning_config"])
    if cfg["measurement_counts"][-1] != base["n_measurements"]:
        raise ValueError("largest measurement count must equal conditioning_config.n_measurements")
    if cfg["sample_budgets"][-1] != base["n_samples"]:
        raise ValueError("largest sample budget must equal conditioning_config.n_samples")
    return source, cfg, base_source, base, policies, references


def _mean_sem(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "design_estimate_sd": float(values.std(ddof=1)) if len(values)>1 else None,
            "design_estimate_sem": float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            "n_designs": len(values)}


def _figure(path, arrays, titles, *, symmetric=(), shared_pairs=(), cell_size_m=5., limits=None):
    fig, axes = plt.subplots(2,3,figsize=(13,7.5),layout="constrained")
    for i,(ax,values,title) in enumerate(zip(axes.flat,arrays,titles)):
        stride = max(1,int(np.ceil(max(values.shape)/640)))
        options = {"cmap":"viridis"}
        for pair in shared_pairs:
            if i in pair:
                options.update(vmin=min(float(arrays[j].min()) for j in pair),
                               vmax=max(float(arrays[j].max()) for j in pair))
        if i in symmetric:
            limit = max(float(np.abs(values).max()),1e-7)
            options = {"cmap":"RdBu_r", "vmin":-limit, "vmax":limit}
        if limits is not None and i in limits:
            options.update(vmin=limits[i][0],vmax=limits[i][1])
        extent = (0,values.shape[1]*cell_size_m/1000,0,values.shape[0]*cell_size_m/1000)
        artist = ax.imshow(values[::stride,::stride],origin="lower",extent=extent,**options)
        ax.set_title(title,fontsize=10)
        ax.set(xlabel="Local column distance [km]",ylabel="Local row distance [km]")
        fig.colorbar(artist,ax=ax,shrink=.8)
    try:
        fig.savefig(path,dpi=150)
    finally:
        plt.close(fig)


def _summarize_case(directory, store, accumulator, reference_temperature, model, cfg):
    stats = accumulator.finalize()
    products = write_temperature_field_products(temperature_store=store, final_mean=stats.mean,
        final_std=stats.std, output_directory=directory, chunk_rows=cfg["quantile_chunk_rows"])
    risk = summarize_risk_store(store, directory/"risk", thresholds=cfg["risk_thresholds_C"],
        background=model.background_temperature, receptors=model.receptors, cell_size_m=model.cell_size_m,
        budgets=cfg["sample_budgets"], comparison=cfg["threshold_comparison"], chunk_rows=cfg["quantile_chunk_rows"])
    _write(directory/"risk.json",risk)
    convergence = prefix_field_convergence(temperature_store=store, checkpoints=cfg["sample_budgets"],
        reference_mean=stats.mean, reference_std=stats.std, chunk_rows=cfg["quantile_chunk_rows"])
    probability = np.load(directory/"risk"/risk["thresholds"][0]["probability_map"])
    width = np.load(products["temperature_width90"])
    difference = stats.mean-reference_temperature
    _figure(directory/"temperature_uncertainty.png",
        (reference_temperature,stats.mean,difference,stats.std,width,probability),
        ("Reference temperature [°C]","MC mean [°C]","Mean − reference [°C]",
         "Conditional temperature SD [°C]","Temperature q95 − q05 [°C]",
         f"P(ΔT {cfg['threshold_comparison']} {cfg['risk_thresholds_C'][0]} °C)"),symmetric=(2,),
        limits={3:(0.,max(float(stats.std.max()),1e-12)),4:(0.,max(float(width.max()),1e-12)),5:(0.,1.)})
    return {"field_statistics": global_uncertainty_metrics(stats.std), "field_convergence":convergence,
            "mean_temperature_shift_rmse_C":float(np.sqrt(np.mean(difference**2))),
            "quantile_probabilities":[.05,.5,.95], "risk_summary":"risk.json",
            "convergence_reference":"largest nested-prefix empirical MC estimate for this same fixed design; not truth",
            "sampling_method":"iid MC; common prior/noise draws across measurement counts"}


def _aggregate(cases, budgets):
    rows = []
    for reference, policy, count in sorted({(c["reference"],c["policy"],c["n_measurements"]) for c in cases}):
        group = sorted([c for c in cases if (c["reference"],c["policy"],c["n_measurements"]) == (reference,policy,count)],
                       key=lambda c:c["design_index"])
        baseline = {c["design_index"]:c for c in cases if c["reference"]==reference and c["policy"]==policy and c["n_measurements"]==0}
        for budget in budgets:
            targets = ["spatial_mean_temperature_variance", *group[0]["qoi_checkpoints"][str(budget)]]
            for target in targets:
                def variance(case):
                    if target == "spatial_mean_temperature_variance":
                        return next(r["U_RMS"]**2 for r in case["field_convergence"] if r["budget"]==budget)
                    return case["qoi_checkpoints"][str(budget)][target]["variance"]
                values = [variance(c) for c in group]
                reductions = [variance(baseline[c["design_index"]])-variance(c) for c in group]
                rows.append({"reference":reference,"policy":policy,"n_measurements":count,
                    "n_samples":budget,"target":target,"units":"°C²",
                    "expected_conditional_variance":_mean_sem(values),
                    "paired_variance_reduction_from_zero":_mean_sem(reductions)})
    return rows


def _density_figures(output, rows):
    for reference, policy in sorted({(r["reference"],r["policy"]) for r in rows}):
        group = [r for r in rows if r["reference"]==reference and r["policy"]==policy]
        targets = sorted({r["target"] for r in group})
        fig, axes = plt.subplots(1,len(targets),figsize=(4.3*len(targets),3.9),squeeze=False,layout="constrained")
        for ax,target in zip(axes.flat,targets):
            for budget in sorted({r["n_samples"] for r in group}):
                selected = sorted([r for r in group if r["target"]==target and r["n_samples"]==budget],key=lambda r:r["n_measurements"])
                x = [r["n_measurements"] for r in selected]
                y = [r["expected_conditional_variance"]["mean"] for r in selected]
                errors = [r["expected_conditional_variance"]["design_estimate_sem"] for r in selected]
                ax.errorbar(x,y,yerr=None if any(e is None for e in errors) else errors, marker="o",label=f"L={budget}")
            ax.set(title=target,xlabel="Conditioning locations",ylabel="Estimated conditional variance [°C²]")
            ax.legend()
        fig.suptitle(f"{reference} / {policy}; bars: SEM across design estimates (includes MC error)")
        try:
            fig.savefig(output/f"{reference}.{policy}.density.png",dpi=150)
        finally:
            plt.close(fig)


def run_study(config_path, output_dir, *, model_factory=load_frozen_temperature_model):
    """Run direct full-covariance MC; injection is for independent model tests."""
    source,cfg,base_source,base,policies,references = load_config(config_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise ValueError("output directory already exists; use a fresh directory")
    model = model_factory(cfg["model_config"],asset_root=cfg["model_asset_root"],torch_threads=cfg["torch_threads"])
    if model.cell_size_m != base["cell_size_m"] or any(tuple(v["reference"].shape)!=tuple(model.input_shape) for v in references.values()):
        raise ValueError("reference grid/spacing must match the fixed model context; prepare cutout-specific inputs explicitly")
    package = Path(__file__).parents[1]
    files = sorted(package.rglob("*.py"))
    source_hashes = {str(p):sha256_file(p) for p in (source,base_source,cfg["model_config"],*files)}
    for item in references.values():
        source_hashes[str(item["path"])] = item["sha256"]
        if item["weights_path"] is not None:
            source_hashes[str(item["weights_path"])] = item["weights_sha256"]
    # Feasibility and all frozen reading designs are checked before any output.
    designs = {}
    for name,item in sorted(references.items()):
        law_id = _identity({"reference_sha256":item["sha256"],"covariance":asdict(item["covariance"]),
                            "mean_mode":base["mean_mode"],"cell_size_m":base["cell_size_m"]},"reference-")
        for policy_name,policy in sorted(policies.items()):
            for design_index in range(base["n_designs"]):
                labels = (law_id,policy_name,design_index)
                observations = draw_reference_measurements(item["reference"],covariance=item["covariance"],
                    n_measurements=base["n_measurements"],location_seed=_seed(base["seed"],*labels,"location"),
                    observation_seed=_seed(base["seed"],*labels,"reading"),cell_size_m=base["cell_size_m"],
                    location_prior=policy,location_weights=item["weights"],measurement_mode=base["measurement_mode"],
                    observation_std_log10_k=base["observation_std_log10_k"])
                designs[name,policy_name,design_index] = (law_id,observations)
    output.mkdir(parents=True)
    expected_evaluations = len(references)*len(policies)*base["n_designs"]*len(cfg["measurement_counts"])*base["n_samples"]
    print(f"Planned: {expected_evaluations} generated-input evaluations + {len(references)} references; direct MC",flush=True)
    manifest = {"schema_version":1,"status":"running","configuration":{k:str(v) if isinstance(v,Path) else v for k,v in cfg.items()},
        "conditioning_configuration":base,"code_commit":git_head(package.parents[1]),
        "source_sha256":source_hashes,"runtime":runtime_versions(("numpy","scipy","gstools","torch","matplotlib")),
        "model":model.metadata,"reference_pooling":None,"policy_pooling":None,"sample_filtering":None,
        "design_weights":"equal iid replicates of each declared synthetic location/reading policy; not inferred geological probabilities",
        "method":"direct full-covariance conditional FFT MC; no KL approximation or RQMC",
        "information_state":"reference_median contains every reference pixel; global_log_mean also uses a reference-derived global statistic",
        "measurement_density":"nested prefixes of one frozen maximum design; zero is the unconditioned law",
        "RQ5_status":"input-variance placement baseline; no predictive-measurement EVR or output-informed location ranking"}
    _write(output/"manifest.json",manifest)
    cases = []
    for name,item in sorted(references.items()):
        reference,spec = item["reference"],item["covariance"]
        directory = output/name
        directory.mkdir()
        nominal = np.asarray(model.predict_temperature(reference))
        if nominal.dtype != np.float32:
            nominal = nominal.astype(np.float32)
        validate_output_grid(nominal.shape,model.receptors,model.mean_anomaly_roi)
        if not np.all(np.isfinite(nominal)) or any(n>m for n,m in zip(nominal.shape,reference.shape)):
            raise ValueError("nominal temperature must be finite and fit the fixed input grid")
        np.save(directory/"reference_temperature.npy",nominal)
        nominal_qoi = scalar_temperature_qois(nominal,receptors=model.receptors,roi=model.mean_anomaly_roi,
                                             background=model.background_temperature)
        _write(directory/"reference_temperature.json",{"qoi":nominal_qoi,"source_sha256":item["sha256"],
            "normalization":model.normalization_diagnostics(reference),"output_shape":list(nominal.shape),
            "output_offset_in_input_cells":[(m-n)//2 for m,n in zip(reference.shape,nominal.shape)],
            "coordinate_convention":"local array row/column; output is the model's valid center crop"})
        _write(directory/"reference_input_descriptors.json",{
            "native_short_range":native_short_range_features(reference,cell_size_m=base["cell_size_m"]),
            "patch_descriptors":_patch_descriptors(reference,base["cell_size_m"]),
            "training_support":"unverified; reference patches are nominal structural evidence only"})
        prior = GSToolsReferenceFFTSampler(spec,reference.shape,cell_size_m=base["cell_size_m"],workers=base["workers"])
        for policy_name in sorted(policies):
            for design_index in range(base["n_designs"]):
                law_id,maximum = designs[name,policy_name,design_index]
                design_dir = directory/policy_name/f"design-{design_index:03d}"
                design_dir.mkdir(parents=True)
                _write_observations(design_dir/"maximum_measurements.csv",maximum,base["cell_size_m"])
                _write(design_dir/"maximum_design.json",{"metadata":maximum.metadata,"covariance":asdict(spec),
                    "reference_sha256":item["sha256"],"n_measurements":len(maximum.indices)})
                for count in cfg["measurement_counts"]:
                    case_dir = design_dir/f"n-{count:04d}"
                    case_dir.mkdir()
                    observations = None if count==0 else maximum.prefix(count)
                    if observations is not None:
                        _write_observations(case_dir/"measurements.csv",observations,base["cell_size_m"])
                    sampler = ConditionalReferenceFFTSampler(reference,spec,observations,cell_size_m=base["cell_size_m"],
                        mean_mode=base["mean_mode"],workers=base["workers"],chunk_size=base["chunk_size"],prior_sampler=prior)
                    mean,variance = sampler.posterior_moments()
                    np.save(case_dir/"posterior_log10_std.npy",np.sqrt(variance).astype(np.float32))
                    input_variance = float(variance.mean())
                    del mean,variance
                    store = case_dir/"temperature_ensemble.npy"
                    temperatures = open_memmap(store,mode="w+",dtype=np.float32,shape=(base["n_samples"],*nominal.shape))
                    accumulator = OnlineFieldStatistics()
                    qoi_samples,records = [],[]
                    factor = base["diagnostic_factor"]
                    diagnostic_fields = np.empty((base["n_samples"],*reference[::factor,::factor].shape))
                    for index in range(base["n_samples"]):
                        seed = _seed(base["seed"],law_id,policy_name,design_index,"field",index)
                        field = sampler.sample_permeability(seed).astype(np.float32)
                        if not np.all(np.isfinite(field)) or np.any(field<=0):
                            raise ArithmeticError("generated field exceeds positive finite float32 range; no replacement")
                        temperature = np.asarray(model.predict_temperature(field),dtype=np.float32)
                        if temperature.shape!=nominal.shape or not np.all(np.isfinite(temperature)):
                            raise ValueError("temperature output changed shape or contains nonfinite values")
                        temperatures[index] = temperature
                        diagnostic_fields[index] = field[::factor,::factor]
                        accumulator.update(temperature)
                        qoi = scalar_temperature_qois(temperature,receptors=model.receptors,roi=model.mean_anomaly_roi,
                                                     background=model.background_temperature)
                        qoi_samples.append(list(qoi.values()))
                        record = {"sample_index":index,"seed":seed,"qoi":qoi,
                                  "field_sha256":sha256(np.ascontiguousarray(field).view(np.uint8)).hexdigest(),
                                  "field_dtype":str(field.dtype),"field_shape":list(field.shape),
                                  "normalization":model.normalization_diagnostics(field),
                                  "reference_agreement":reference_field_agreement(reference,field),
                                  "native_short_range":native_short_range_features(field,cell_size_m=base["cell_size_m"]),
                                  "patch_descriptors":_patch_descriptors(field,base["cell_size_m"])}
                        if index in base["retained_sample_indices"]:
                            np.save(case_dir/f"permeability-{index:04d}.npy",field)
                            np.save(case_dir/f"temperature-{index:04d}.npy",temperature)
                            _figure(case_dir/f"sample-{index:04d}.png",
                                (np.log10(reference),np.log10(field),np.log10(field/reference),nominal,temperature,temperature-nominal),
                                ("Reference log10 K [m²]","Generated log10 K [m²]","log10(K/Kref)",
                                 "Reference temperature [°C]","Generated temperature [°C]","Temperature difference [°C]"),
                                symmetric=(2,5),shared_pairs=((0,1),(3,4)),cell_size_m=model.cell_size_m)
                        records.append(record)
                    temperatures.flush()
                    del temperatures
                    qoi_samples = np.asarray(qoi_samples)
                    np.save(case_dir/"qoi_samples.npy",qoi_samples)
                    coherence = reference_ensemble_coherence(reference[::factor,::factor],diagnostic_fields,
                        cell_size_m=base["cell_size_m"]*factor,max_lag_cells=base["max_lag_cells"])
                    _write(case_dir/"reference_diagnostics.json",{
                        "diagnostics":coherence,"sampling":"deterministic point stride; no block averaging",
                        "diagnostic_factor":factor,"native_cell_size_m":base["cell_size_m"],
                        "diagnostic_cell_size_m":base["cell_size_m"]*factor,
                        "interpretation":"distribution, directional correlations/variograms and radial/angular/2-D spectra at declared diagnostic resolution; native gradients and patch descriptors in sample records",
                        "sample_filtering":None,"training_support":"unverified"})
                    del diagnostic_fields,coherence
                    case = _summarize_case(case_dir,store,accumulator,nominal,model,cfg)
                    case.update({"reference":name,"policy":policy_name,"design_index":design_index,"n_measurements":count,
                        "directory":str(case_dir.relative_to(output)),"qoi_names":list(nominal_qoi),
                        "analytic_mean_input_variance_log10":input_variance,"sampler":sampler.metadata,"records":records})
                    case["qoi_checkpoints"] = {str(n):{target:{**scalar_summary(qoi_samples[:n,j]),
                        "variance":float(qoi_samples[:n,j].var(ddof=1)),
                        "mean_standard_error":float(qoi_samples[:n,j].std(ddof=1)/np.sqrt(n)),
                        "mean_shift_from_reference_C":float(qoi_samples[:n,j].mean()-nominal_qoi[target])}
                        for j,target in enumerate(nominal_qoi)} for n in cfg["sample_budgets"]}
                    if not cfg["retain_temperature_ensemble"]:
                        store.unlink()
                    case["artifacts_sha256"] = {str(p.relative_to(case_dir)):sha256_file(p) for p in sorted(case_dir.rglob("*")) if p.is_file()}
                    _write(case_dir/"summary.json",case)
                    cases.append(case)
                    print(f"{name}/{policy_name}/design-{design_index:03d}/n={count}: {base['n_samples']} LGCNN draws complete",flush=True)
                    del sampler,accumulator,field,temperature
        del prior
    rows = _aggregate(cases,cfg["sample_budgets"])
    _density_figures(output,rows)
    _write(output/"density_summary.json",{"rows":rows,"reference_pooling":None,"policy_pooling":None,
        "interpretation":"mean conditional temperature/QoI variance across repeated synthetic designs; no pooled posterior",
        "error_bars":"SEM across independent design estimates, combining design variability and finite MC error; absent for one design",
        "paired_reductions":"differences of variance estimates under common FFT/noise draws; can be negative, never clipped",
        "convergence":"prefix-to-largest comparisons are dependent; zero final difference does not prove convergence"})
    with (output/"density_summary.csv").open("w",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=["reference","policy","n_measurements","n_samples","target",
            "variance_C2","variance_sem_C2","paired_reduction_C2","paired_reduction_sem_C2","n_designs"])
        writer.writeheader()
        for row in rows:
            variance,reduction = row["expected_conditional_variance"],row["paired_variance_reduction_from_zero"]
            writer.writerow({**{k:row[k] for k in ("reference","policy","n_measurements","n_samples","target")},
                "variance_C2":variance["mean"],"variance_sem_C2":variance["design_estimate_sem"],
                "paired_reduction_C2":reduction["mean"],"paired_reduction_sem_C2":reduction["design_estimate_sem"],"n_designs":variance["n_designs"]})
    for path,digest in {**source_hashes,**model.metadata.get("files_sha256",{})}.items():
        if sha256_file(path)!=digest:
            raise ValueError("source/model context changed during execution; results are not frozen")
    manifest.update({"status":"complete","case_count":len(cases),
        "generated_field_evaluations":len(cases)*base["n_samples"],"reference_evaluations":len(references)})
    manifest["artifacts_sha256"] = {str(p.relative_to(output)):sha256_file(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p!=output/"manifest.json"}
    _write(output/"manifest.json",manifest)
    return {"manifest":manifest,"rows":rows,"cases":cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,help="density-study YAML; paths resolve relative to this file")
    parser.add_argument("--output-dir",required=True,help="fresh output directory; existing outputs are refused")
    args = parser.parse_args(argv)
    run_study(args.config,args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
