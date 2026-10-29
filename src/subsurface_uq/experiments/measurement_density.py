"""Native sparse-reconstruction measurement-density propagation for RQ1/RQ2.

Stages screen input covariance choices, prepare every native conditional input,
query a frozen LGCNN, then summarize temperature uncertainty and regional risk.
No output-informed placement, reference-mean prior or sample filtering is used.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.special import ndtri
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from ..sampling.covariance_reconstruction import OrdinaryKrigingSystem
from ..sampling.measurement_density import coverage_locations, gaussian_crps, observation_condition_number, native_ensembles
from ..sampling.reference_coherence import native_short_range_features, reference_ensemble_coherence
from ..statistics.online import OnlineFieldStatistics
from ..statistics.regional_temperature import empirical_tail_risk, regional_deviation_samples
from ..statistics.temperature_risk import receptor_risk, scalar_temperature_qois, validate_output_grid
from ..rq1.analysis import scalar_summary
from ..surrogates.frozen_temperature import load_frozen_temperature_model
from .conditioning_temperature import _figure, _summarize_case
from .covariance_reconstruction import quantile_interval
from .reference_conditioning import _identity, _name, _positive_integer, _relative_path, _seed, _write


_KEYS={"references","kernel_fits","fit_max_lag_m","families","sigma_log10","screen_sigma_log10",
       "measurement_counts","coverage_scenarios","cluster_scale_fraction","candidate_stride_cells",
       "screen_stride_cells","cell_size_m","n_samples","sample_budgets","seed","workers","chunk_size",
       "model_config","model_asset_root","torch_threads","risk_thresholds_C","deviation_limits_C","quantile_chunk_rows"}
_LEGACY_KEYS = {"kernel_fits", "fit_max_lag_m", "families", "sigma_log10", "screen_sigma_log10"}
_VALIDATION_KEYS = (_KEYS - _LEGACY_KEYS) | {"validation", "covariance_candidates"}


def load_config(path):
    source=Path(path).resolve();cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or set(cfg) not in (_KEYS, _VALIDATION_KEYS):
        raise ValueError("declare either the legacy texture benchmark or the observations-only validation configuration")
    validated = "validation" in cfg
    if validated:
        from .measurement_density_validation import validate_settings
        validate_settings(cfg)
    for key in ("candidate_stride_cells","screen_stride_cells","n_samples","workers","chunk_size","torch_threads","quantile_chunk_rows"):
        _positive_integer(cfg[key],key)
    _positive_integer(cfg["seed"],"seed",zero=True)
    if cfg["cell_size_m"]!=5. or cfg["n_samples"]<4 or cfg["n_samples"]%2:
        raise ValueError("native 5 m cells and an even sample count >=4 required")
    if cfg["candidate_stride_cells"]%cfg["screen_stride_cells"]:
        raise ValueError("candidate stride must be a multiple of screening stride")
    for key,minimum in (("measurement_counts",1),("sample_budgets",2)):
        v=cfg[key]
        if not isinstance(v,list) or not v or v!=sorted(set(v)) or any(isinstance(n,bool) or not isinstance(n,int) or n<minimum for n in v):
            raise ValueError(f"{key} must contain increasing integer budgets")
    if cfg["sample_budgets"][-1]!=cfg["n_samples"]:
        raise ValueError("largest sample budget must equal n_samples")
    for key in (("cluster_scale_fraction",) if validated else ("sigma_log10","fit_max_lag_m","cluster_scale_fraction")):
        if isinstance(cfg[key],bool) or not np.isfinite(cfg[key]) or cfg[key]<=0:
            raise ValueError(f"{key} must be positive")
    if cfg["cluster_scale_fraction"]>=1:
        raise ValueError("cluster_scale_fraction must be below 1")
    for key in (("risk_thresholds_C","deviation_limits_C") if validated else ("screen_sigma_log10","risk_thresholds_C","deviation_limits_C")):
        v=cfg[key]
        if not isinstance(v,list) or not v or len(set(v))!=len(v) or any(isinstance(x,bool) or not np.isfinite(x) or x<=0 for x in v):
            raise ValueError(f"{key} must contain distinct positive finite numbers")
    if not validated and cfg["sigma_log10"] not in cfg["screen_sigma_log10"]:
        raise ValueError("primary amplitude must occur in screening amplitudes")
    if not cfg["coverage_scenarios"] or len(set(cfg["coverage_scenarios"]))!=len(cfg["coverage_scenarios"]) or not set(cfg["coverage_scenarios"])<={"balanced","clustered"}:
        raise ValueError("declare distinct balanced/clustered coverage scenarios")
    if not validated and (not cfg["families"] or len(set(cfg["families"]))!=len(cfg["families"]) or not set(cfg["families"])<={"gaussian","matern32_radial","exponential_radial"}):
        raise ValueError("unsupported covariance families")
    if not isinstance(cfg["references"],dict) or not cfg["references"]:
        raise ValueError("reference mapping required")
    fits = None if validated else json.loads(_relative_path(cfg["kernel_fits"],source.parent).read_text(encoding="utf-8"))
    refs={}
    for name,raw in cfg["references"].items():
        _name(name,"reference");p=_relative_path(raw,source.parent)
        values=np.load(p,mmap_mode="r",allow_pickle=False)
        if np.iscomplexobj(values) or values.ndim!=2 or min(values.shape)<3 or not np.all(np.isfinite(values)) or np.any(values<=0):
            raise ValueError("positive finite two-dimensional reference arrays required")
        if validated:
            refs[name] = {"path": p, "shape": tuple(values.shape), "sha256": sha256_file(p)}
            continue
        record=fits["references"][name]
        if sha256_file(p)!=record["reference_sha256"]:
            raise ValueError("reference does not match frozen texture-fit source")
        specs={}
        for family in cfg["families"]:
            selected=[f for f in record["fits"] if f["family"]==family and f["fit_max_lag_m"]==cfg["fit_max_lag_m"]]
            if len(selected)!=1:raise ValueError("one declared fit per RUN/family/window required")
            f=selected[0]
            specs[family]=GSToolsReferenceSpec(family,f["ell_row_m"],f["ell_col_m"],cfg["sigma_log10"])
        refs[name]={"path":p,"shape":tuple(values.shape),"sha256":record["reference_sha256"],"specs":specs}
    return source,cfg,refs


def _sources(source,cfg,refs):
    package=Path(__file__).parents[1]
    fits = [] if "validation" in cfg else [_relative_path(cfg["kernel_fits"],source.parent)]
    files=[source,*fits,*sorted(package.rglob("*.py")),*[r["path"] for r in refs.values()]]
    return {str(p.resolve()):sha256_file(p) for p in files}


def _check(digests):
    for p,digest in digests.items():
        if sha256_file(p)!=digest:raise ValueError(f"frozen study source changed: {p}")


def _layout(directory):
    return np.load(directory/"measurement_indices.npy")


def screen(source,cfg,refs,output):
    if "validation" in cfg:
        from .measurement_density_validation import screen_validation
        return screen_validation(source, cfg, refs, output, source_hashes=_sources(source, cfg, refs))
    if output.exists():raise ValueError("screening requires a fresh output directory")
    output.mkdir(parents=True)
    manifest={"status":"screening","configuration":cfg,"configuration_path":str(source),"source_sha256":_sources(source,cfg,refs),
        "base_git_commit":git_head(Path(__file__).parents[3]),"runtime":runtime_versions(("numpy","scipy","gstools","torch")),
        "information_state":"unknown constant mean estimated only from selected exact readings; no full-reference mean",
        "reference_pooling":None,"scenario_weights":None,"sample_filtering":None,"measurement_error_variance":0.,
        "kernel_prior_information":"frozen full-reference texture-fit lengths; not independently calibrated uncertainty",
        "selection":"lowest average withheld-point CRPS at declared amplitude across equally weighted benchmark counts/layouts; weights are not geological probabilities"}
    manifest["validation_status"] = "legacy full-reference-informed benchmark; independent predictive adequacy unassessed"
    manifest["study_id"]=_identity({"configuration":cfg,"sources":manifest["source_sha256"]},"density-")
    _write(output/"manifest.json",manifest)
    all_results=[];primary={}
    for name,record in refs.items():
        truth=np.load(record["path"],mmap_mode="r")
        root=output/name;root.mkdir()
        layouts={}
        for coverage in cfg["coverage_scenarios"]:
            d=root/coverage;d.mkdir()
            indices,metadata=coverage_locations(record["shape"],max(cfg["measurement_counts"]),stride=cfg["candidate_stride_cells"],
                seed=_seed(cfg["seed"],name,coverage,"locations"),kind=coverage,cluster_scale=cfg["cluster_scale_fraction"])
            layouts[coverage]=indices
            np.save(d/"measurement_indices.npy",indices)
            _write(d/"design.json",metadata)
            np.savetxt(d/"measurements.csv",np.column_stack([np.arange(1,len(indices)+1),indices,np.log10(truth[tuple(indices.T)].astype(float))]),
                delimiter=",",header="order,native_row,native_column,log10_K",comments="",fmt=["%d","%d","%d","%.17g"])
        factor=cfg["screen_stride_cells"]
        small=truth[::factor,::factor]
        queries=np.column_stack(np.unravel_index(np.arange(small.size),small.shape))*factor
        withheld=np.ones(small.shape,dtype=bool)
        for indices in layouts.values():withheld[tuple((indices//factor).T)]=False
        np.save(root/"screen_withheld_mask.npy",withheld)
        for family,spec in record["specs"].items():
            for coverage,indices in layouts.items():
                for n in cfg["measurement_counts"]:
                    base={"reference":name,"family":family,"coverage":coverage,"n_measurements":n}
                    print(f"screen {name}/{family}/{coverage}/n={n}",flush=True)
                    try:
                        cond=observation_condition_number(spec,indices[:n],cfg["cell_size_m"])
                        readings=np.log10(truth[tuple(indices[:n].T)].astype(float))
                        system=OrdinaryKrigingSystem(record["shape"],spec,indices[:n],readings,cell_size_m=cfg["cell_size_m"])
                        mean=np.empty(len(queries));variance=np.empty(len(queries))
                        for start in range(0,len(queries),cfg["chunk_size"]):
                            stop=min(start+cfg["chunk_size"],len(queries))
                            mean[start:stop],variance[start:stop]=system.moments(system.features(queries[start:stop]))
                        target=np.log10(np.asarray(small,dtype=float)).ravel()[withheld.ravel()]
                        m=mean[withheld.ravel()];v=variance[withheld.ravel()]
                        for sigma in cfg["screen_sigma_log10"]:
                            sd=sigma*np.sqrt(v)
                            all_results.append({**base,"status":"available","sigma_log10":sigma,"condition_number":cond,
                                "mean_crps_log10":float(gaussian_crps(m,sd,target).mean()),"mean_rmse_log10":float(np.sqrt(np.mean((m-target)**2))),
                                "reference_inside_pointwise_95_fraction":float(np.mean(np.abs(m-target)<=ndtri(.975)*sd)),
                                "average_posterior_variance_log10":float(np.mean(sd**2)),"estimated_mean_selected_only":system.estimated_mean})
                    except (ValueError,np.linalg.LinAlgError,ArithmeticError) as exc:
                        all_results.append({**base,"status":"numerically_unavailable","reason":str(exc),"sample_rejection":False})
        rankings=[]
        for family in cfg["families"]:
            rows=[r for r in all_results if r["reference"]==name and r["family"]==family and r.get("sigma_log10")==cfg["sigma_log10"] and r["status"]=="available"]
            if len(rows)==len(layouts)*len(cfg["measurement_counts"]):
                rankings.append({"family":family,"mean_crps_log10":float(np.mean([r["mean_crps_log10"] for r in rows])),"covariance":asdict(record["specs"][family])})
        rankings.sort(key=lambda r:r["mean_crps_log10"])
        if not rankings:raise ValueError(f"no covariance family is numerically available for every count of {name}")
        primary[name]={"selected":rankings[0],"ranking":rankings}
        print(f"input-only primary for {name}: {rankings[0]['family']}",flush=True)
    _write(output/"input_screen.json",{"rows":all_results,"primary":primary,
        "withheld_scope":"withheld from conditioning and common across counts/layouts; texture fits used the full reference",
        "interpretation":"average pointwise predictive scores, not an iid spatial likelihood or geological calibration"})
    _check(manifest["source_sha256"])
    manifest.update(status="screened",primary_covariance=primary)
    _write(output/"manifest.json",manifest)
    return manifest


def _designs(cfg, refs, output, manifest):
    if "validation" in cfg:
        for d in manifest["designs"]:
            yield d, refs[d["reference"]], output/d["directory"], GSToolsReferenceSpec(**d["covariance"])
    else:
        for name, record in refs.items():
            spec = GSToolsReferenceSpec(**manifest["primary_covariance"][name]["selected"]["covariance"])
            for coverage in cfg["coverage_scenarios"]:
                yield {"reference": name, "coverage": coverage, "design_index": None}, record, output/name/coverage, spec


def _require_adequacy(cfg, output, manifest):
    if "validation" in cfg:
        _check({str(output/p): digest for p, digest in manifest["validation_artifact_sha256"].items()})
        assessment = json.loads((output/"reconstruction_validation.json").read_text())
        if not manifest.get("adequate_for_exploratory_propagation") or not assessment["adequate_for_exploratory_propagation"]:
            raise ValueError("withheld reconstruction uncertainty fails declared checks; propagation is not justified by this study")


def _atomic_json(path, value):
    temporary = path.with_name(path.name+".partial")
    _write(temporary, value)
    temporary.replace(path)


def prepare_inputs(source,cfg,refs,output,manifest):
    _check(manifest["source_sha256"])
    _require_adequacy(cfg, output, manifest)
    manifest["status"] = "preparing_inputs"; _write(output/"manifest.json", manifest)
    for design, record, directory, spec in _designs(cfg, refs, output, manifest):
        completed = True
        for n in cfg["measurement_counts"]:
            target = directory/f"n-{n:04d}"
            ledger = target/"input_files.json"
            if ledger.exists():
                _check({str(target/p): h for p, h in json.loads(ledger.read_text()).items()})
            else:
                completed = False
        prior_path = directory/"temporary_unit_prior.npy"
        progress_path = directory/"temporary_prior_progress.json"
        if completed:
            # Both files belong to this verified study, and are no longer needed.
            for p in (prior_path, progress_path):
                if p.exists(): p.unlink()
            continue
        truth = np.load(record["path"], mmap_mode="r")
        prior = GSToolsReferenceFFTSampler(replace(spec, sigma_R=1.), record["shape"],
            cell_size_m=cfg["cell_size_m"], workers=cfg["workers"])
        prefix = (design["reference"], "unit-prior") if design["design_index"] is None else (
            design["reference"], design["coverage"], design["design_index"], "unit-prior")
        seeds = [_seed(cfg["seed"], *prefix, i) for i in range(cfg["n_samples"])]
        context = {"covariance": asdict(spec), "shape": list(record["shape"]), "seeds": seeds}
        if prior_path.exists():
            if not progress_path.exists(): raise ValueError("prior cache has no recovery context")
            progress = json.loads(progress_path.read_text())
            if progress["context"] != context: raise ValueError("prior recovery context changed")
            draws = np.load(prior_path, mmap_mode="r+")
            if draws.shape != (cfg["n_samples"], *record["shape"]) or draws.dtype != np.float64:
                raise ValueError("prior cache shape or dtype changed")
            for i, digest in enumerate(progress["draw_sha256"]):
                if sha256_file_bytes(draws[i]) != digest: raise ValueError("completed prior cache changed")
        else:
            draws = open_memmap(prior_path, mode="w+", dtype=np.float64, shape=(cfg["n_samples"], *record["shape"]))
            progress = {"context": context, "draw_sha256": []}
            _atomic_json(progress_path, progress)
        for i in range(len(progress["draw_sha256"]), len(seeds)):
            print(f"native prior {design}: {i+1}/{len(seeds)}", flush=True)
            draws[i] = prior.sample_residual(seeds[i]); draws.flush()
            progress["draw_sha256"].append(sha256_file_bytes(draws[i]))
            _atomic_json(progress_path, progress)
        metadata = {**prior.metadata, "spec": {**prior.metadata["spec"], "marginal": "zero-mean unit Gaussian residual"}}
        _write(directory/"prior.json", {"metadata": metadata, "seeds": seeds,
            "field_pairing": "shared across counts within this fixed design and covariance"})
        native_ensembles(truth, spec, _layout(directory), cfg["measurement_counts"], draws, directory,
                        cell_size_m=cfg["cell_size_m"], chunk_size=cfg["chunk_size"])
        del draws, prior
        prior_path.unlink(); progress_path.unlink()
    _check(manifest["source_sha256"])
    manifest["status"]="inputs_ready";_write(output/"manifest.json",manifest)


def _finish_temperature_case(directory,store,nominal,model,cfg,records):
    accumulator=OnlineFieldStatistics()
    ensemble=np.load(store,mmap_mode="r")
    for field in ensemble:accumulator.update(field)
    summary=_summarize_case(directory,store,accumulator,nominal,model,{**cfg,"threshold_comparison":">"})
    qois=[scalar_temperature_qois(t,receptors=model.receptors,roi=model.mean_anomaly_roi,background=model.background_temperature) for t in ensemble]
    names=list(qois[0]);values=np.array([[q[n] for n in names] for q in qois])
    np.save(directory/"qoi_samples.npy",values)
    regional=regional_deviation_samples(ensemble,nominal,model.mean_anomaly_roi)
    np.savez(directory/"regional_deviation_samples.npz",**regional)
    loss=regional["maximum_absolute_deviation_C"]
    region={"roi_output_cells":model.mean_anomaly_roi,"interpretation":"fixed ROI within valid output crop; maximum taken within each realization before quantiles",
        "maximum_absolute_deviation":{**empirical_tail_risk(loss),"VaR_ci95":quantile_interval(loss),
            "exceedance_by_limit":{str(t):receptor_risk(loss,background=0.,threshold=t) for t in cfg["deviation_limits_C"]}},
        "other_losses":{k:scalar_summary(v) for k,v in regional.items() if k!="maximum_absolute_deviation_C"}}
    _write(directory/"regional_risk.json",region)
    half=cfg["n_samples"]//2
    batches=[]
    for a,b in [(0,half),(half,cfg["n_samples"])]:
        stats=np.std(np.asarray(ensemble[a:b],dtype=float),axis=0,ddof=1)
        batches.append({"sample_indices":[a,b],"U_RMS_C":float(np.sqrt(np.mean(stats**2))),
            "qoi":{name:scalar_summary(values[a:b,j]) for j,name in enumerate(names)},
            "regional_maximum_risk":empirical_tail_risk(loss[a:b])})
    summary.update(qoi_names=names,qoi={name:scalar_summary(values[:,j]) for j,name in enumerate(names)},
        regional_risk=region,independent_half_batches=batches,prediction_records=records,
        sampling_limit="pointwise 95th-percentile CI may have an unbounded upper endpoint at this finite budget; no tail-convergence claim")
    return summary


def propagate(source,cfg,refs,output,manifest,model_factory=load_frozen_temperature_model):
    _check(manifest["source_sha256"])
    _require_adequacy(cfg, output, manifest)
    model=model_factory(_relative_path(cfg["model_config"],source.parent),asset_root=_relative_path(cfg["model_asset_root"],source.parent),torch_threads=cfg["torch_threads"])
    if any(tuple(model.input_shape)!=r["shape"] for r in refs.values()) or model.cell_size_m!=cfg["cell_size_m"]:
        raise ValueError("native reference grid must match the fixed model")
    if manifest.get("model") is not None and manifest["model"]!=model.metadata:
        raise ValueError("model context differs from the frozen propagation context")
    manifest.update(status="propagating",model=model.metadata)
    _write(output/"manifest.json",manifest)
    cases=[]
    for design, record, directory, spec in _designs(cfg, refs, output, manifest):
        name, coverage = design["reference"], design["coverage"]
        reference=np.load(record["path"],mmap_mode="r")
        root=output/name;nominal_path=root/"reference_temperature.npy"
        if nominal_path.exists():
            context = json.loads((root/"reference_temperature.json").read_text())
            if context.get("temperature_sha256") != sha256_file(nominal_path):
                raise ValueError("nominal temperature cache changed or has no verified hash")
            nominal=np.load(nominal_path)
        else:
            nominal=np.asarray(model.predict_temperature(reference),dtype=np.float32)
            validate_output_grid(nominal.shape,model.receptors,model.mean_anomaly_roi)
            np.save(nominal_path,nominal)
            _write(root/"reference_temperature.json",{"output_shape":list(nominal.shape),"output_offset_in_input_cells":[(m-n)//2 for m,n in zip(reference.shape,nominal.shape)],
                "normalization":model.normalization_diagnostics(reference),"reference_permeability_sha256":record["sha256"],
                "temperature_sha256": sha256_file(nominal_path)})
        validate_output_grid(nominal.shape, model.receptors, model.mean_anomaly_roi)
        for n in cfg["measurement_counts"]:
            target=directory/f"n-{n:04d}"
            files=json.loads((target/"input_files.json").read_text())
            _check({str(target/p):h for p,h in files.items()})
            inputs=np.load(target/"permeability_ensemble.npy",mmap_mode="r")
            store=target/"temperature_ensemble.npy"
            progress=target/"prediction_records.json"
            records=json.loads(progress.read_text()) if progress.exists() else []
            temperatures=open_memmap(store,mode="r+" if store.exists() else "w+",dtype=np.float32,shape=(cfg["n_samples"],*nominal.shape))
            if temperatures.shape!=(cfg["n_samples"],*nominal.shape) or temperatures.dtype!=np.float32:
                raise ValueError("prediction cache grid, budget or dtype differs from the frozen study")
            if len(records) > cfg["n_samples"] or any(r["sample_index"] != i for i,r in enumerate(records)):
                raise ValueError("invalid prediction recovery sequence")
            for i,r in enumerate(records):
                if sha256_file_bytes(inputs[i])!=r["input_sha256"] or sha256_file_bytes(temperatures[i])!=r["temperature_sha256"]:
                    raise ValueError("completed prediction cache changed; refusing resume")
            for i in range(len(records),cfg["n_samples"]):
                start=time.monotonic();field=np.asarray(inputs[i])
                temperature=np.asarray(model.predict_temperature(field),dtype=np.float32)
                if temperature.shape!=nominal.shape or not np.all(np.isfinite(temperature)):
                    raise ValueError("invalid LGCNN prediction; no field replacement")
                temperatures[i]=temperature;temperatures.flush()
                records.append({"sample_index":i,"input_sha256":sha256_file_bytes(field),"temperature_sha256":sha256_file_bytes(temperature),
                    "seconds":time.monotonic()-start,"normalization":model.normalization_diagnostics(field),
                    "native_short_range":native_short_range_features(field,cell_size_m=cfg["cell_size_m"])})
                _atomic_json(progress,records)
                print(f"LGCNN {design}/n={n}: {i+1}/{cfg['n_samples']} ({records[-1]['seconds']:.2f} s)",flush=True)
                if i==0:
                    _figure(target/"sample-0000.png",(np.log10(reference),np.log10(field),np.log10(field/reference),nominal,temperature,temperature-nominal),
                        ("Reference log10 K","Conditional log10 K","log10(K/Kref)","Reference temperature [°C]","Conditional temperature [°C]","Temperature difference [°C]"),
                        symmetric=(2,5),shared_pairs=((0,1),(3,4)),cell_size_m=model.cell_size_m)
            del temperatures
            summary=_finish_temperature_case(target,store,nominal,model,cfg,records)
            diagnostics=reference_ensemble_coherence(reference[::cfg["screen_stride_cells"],::cfg["screen_stride_cells"]],
                np.asarray(inputs[:,::cfg["screen_stride_cells"],::cfg["screen_stride_cells"]]),cell_size_m=cfg["cell_size_m"]*cfg["screen_stride_cells"],max_lag_cells=16)
            _write(target/"input_coherence.json",{"diagnostics":diagnostics,"resolution_m":cfg["cell_size_m"]*cfg["screen_stride_cells"],
                "interpretation":"descriptive reference comparison; native short-range features retained per draw; no training-support certificate","sample_filtering":None})
            summary.update(reference=name,coverage=coverage,design_index=design["design_index"],n_measurements=n,kernel=spec.family,
                sigma_log10=spec.sigma_R,input_summary=json.loads((target/"input_summary.json").read_text()))
            _write(target/"temperature_summary.json",summary);cases.append(summary)
            del inputs
    _check({**manifest["source_sha256"],**model.metadata.get("files_sha256",{})})
    manifest.update(status="propagated",generated_evaluations=len(cases)*cfg["n_samples"],reference_evaluations=len(refs))
    _write(output/"manifest.json",manifest)
    return cases


def sha256_file_bytes(values):
    from hashlib import sha256
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def report(cfg,refs,output,manifest):
    rows=[]
    fig,axes=plt.subplots(len(refs),3,figsize=(15,4.2*len(refs)),squeeze=False,layout="constrained")
    for i,name in enumerate(refs):
        for design, _, directory, _ in _designs(cfg, refs, output, manifest):
            if design["reference"] != name:
                continue
            coverage = design["coverage"]
            cases=[]
            for n in cfg["measurement_counts"]:
                c=json.loads((directory/f"n-{n:04d}"/"temperature_summary.json").read_text());cases.append(c)
                rows.append({"reference":name,"coverage":coverage,"design_index":design["design_index"],"kernel":c["kernel"],"n_measurements":n,"n_samples":cfg["n_samples"],
                    "input_average_variance_log10":c["input_summary"]["analytic_average_input_variance_log10"],
                    "input_reference_mean_rmse_log10":c["input_summary"]["kriging_mean_reference_rmse_log10"],
                    "temperature_U_RMS_C":c["field_statistics"]["U_RMS"],"temperature_mean_shift_rmse_C":c["mean_temperature_shift_rmse_C"],
                    "regional_maximum_VaR95_C":c["regional_risk"]["maximum_absolute_deviation"]["VaR"],
                    "regional_maximum_ES95_C":c["regional_risk"]["maximum_absolute_deviation"]["expected_shortfall"]})
            counts=cfg["measurement_counts"]
            label = coverage if design["design_index"] is None else f"{coverage}, design {design['design_index']+1}"
            axes[i,0].plot(counts,[np.sqrt(c["input_summary"]["analytic_average_input_variance_log10"]) for c in cases],"o-",label=label)
            axes[i,1].plot(counts,[c["field_statistics"]["U_RMS"] for c in cases],"o-",label=label)
            axes[i,2].plot(counts,[c["regional_risk"]["maximum_absolute_deviation"]["VaR"] for c in cases],"o-",label=label)
        for j,(title,unit) in enumerate((("Average input uncertainty","log10 RMS SD"),("Temperature uncertainty","RMS temperature SD [°C]"),("Regional maximum deviation","VaR95 [°C]"))):
            axes[i,j].set(xscale="log",xlabel="Known exact readings",ylabel=unit,title=f"{name}: {title}")
            axes[i,j].invert_xaxis();axes[i,j].grid(alpha=.2);axes[i,j].legend()
    fig.suptitle("Measurement-density propagation: information decreases from left to right\nNative conditional fields; selected-readings-only mean; fixed frozen LGCNN; separate coverage scenarios")
    fig.savefig(output/"density_summary.png",dpi=160);plt.close(fig)
    _write(output/"density_summary.json",{"rows":rows,"scenario_pooling":None,"sampling":"iid conditional MC; shared draws across counts within each fixed design/covariance",
        "limits":"finite-budget estimates; no guarantee of monotone realized temperature variance or calibrated geological probabilities"})
    with (output/"density_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    manifest.update(status="complete",artifact_sha256={str(p.relative_to(output)):sha256_file(p) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"})
    _write(output/"manifest.json",manifest)
    return rows


def run(config_path,output_dir,stage="all",*,model_factory=load_frozen_temperature_model):
    source,cfg,refs=load_config(config_path);output=Path(output_dir).resolve()
    if stage in {"screen","validate","all"}:
        manifest=screen(source,cfg,refs,output)
    else:
        manifest=json.loads((output/"manifest.json").read_text())
        if manifest["configuration"]!=cfg:raise ValueError("configuration changed since screening")
        _check(manifest["source_sha256"])
    if stage in {"inputs", "temperatures"}: _require_adequacy(cfg, output, manifest)
    if stage=="inputs" and manifest["status"] not in {"screened", "validated", "preparing_inputs", "inputs_ready"}:
        raise ValueError("input stage requires completed screening or validation")
    if stage=="temperatures" and manifest["status"] not in {"inputs_ready","propagating","propagated","complete"}:
        raise ValueError("native inputs must be complete before propagation")
    if stage=="report" and manifest["status"] not in {"propagated","complete"}:raise ValueError("propagation must be complete before reporting")
    if stage == "all" and manifest["status"] == "validation_failed":
        return manifest
    if stage in {"inputs","all"}:prepare_inputs(source,cfg,refs,output,manifest)
    if stage in {"temperatures","all"}:propagate(source,cfg,refs,output,manifest,model_factory=model_factory)
    if stage in {"report","all"}:report(cfg,refs,output,manifest)
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--stage",choices=("screen","validate","inputs","temperatures","report","all"),default="all")
    a=p.parse_args(argv);run(a.config,a.output_dir,a.stage)


if __name__=="__main__":main()
