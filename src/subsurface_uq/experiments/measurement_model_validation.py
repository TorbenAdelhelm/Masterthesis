"""Validate regional hydraulic-conductivity models against real held-out data.

Freeze a geometry-only spatial holdout, select models within buffered spatial
development CV, then evaluate the frozen choices. No raster, RUN, LGCNN or
temperature query is used. Sources and prior completed studies are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtr, ndtri
import yaml

from ..io import sha256_file, git_head, runtime_versions
from ..sampling.munich_measurements import load_munich_hydraulic_conductivity_measurements
from ..validation.measurement_model import (MODELS, spatial_partition, development_cv, fit_model,
    spatial_scores, paired_block_bootstrap, block_coordinates)
from ..validation.reconstruction import predictive_scores
from .reference_conditioning import _identity, _relative_path, _write, _positive_integer


_KEYS={"measurements","sheet_name","stratigraphy","groundwater_state","coordinate_assumption",
       "coordinate_bounds_xy_m","units_assumption","models","holdout","development","variogram",
       "reml","bootstrap","rules","distance_bins_m"}


def atomic(path,payload):
    path=Path(path); temp=path.with_name(path.name+".partial")
    _write(temp,payload);temp.replace(path)


def commit(directory,files,marker):
    atomic(directory/marker,{str(p):sha256_file(directory/p) for p in files})


def complete(directory,marker):
    path=directory/marker
    if not path.exists():
        return False
    for p,h in json.loads(path.read_text()).items():
        if sha256_file(directory/p)!=h:
            raise ValueError(f"completed validation artifact changed: {p}")
    return True


def load_config(path):
    source=Path(path).resolve();cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or set(cfg)!=_KEYS:
        raise ValueError("declare exactly the measurement-validation configuration keys")
    if (not isinstance(cfg["models"],list) or len(set(cfg["models"]))!=len(cfg["models"])
            or not set(cfg["models"])<=set(MODELS) or "iid_normal" not in cfg["models"]
            or not any(m.startswith("legacy_") for m in cfg["models"])):
        raise ValueError("known distinct candidates, iid baseline and >=1 legacy model required")
    for name,keys in {
        "holdout":{"block_size_m","fraction","buffer_m","seed"},
        "development":{"block_size_m","buffer_m","folds","seed"},
        "variogram":{"lag_bin_m","max_lag_m","angle_tolerance_deg","min_pairs_per_bin","fit_nugget","pair_count_weighted_fit"},
        "reml":{"length_bounds_m","structured_std_bounds_log10","nugget_std_bounds_log10","initial_lengths_m","maxiter"},
        "bootstrap":{"replicates","seed"},
        "rules":{"minimum_test_blocks","coverage90_min","coverage90_max","standardized_rmse_min","standardized_rmse_max",
                  "absolute_standardized_bias_max","minimum_relative_crps_improvement","require_paired_bootstrap_improvement"},
    }.items():
        if not isinstance(cfg[name],dict) or set(cfg[name])!=keys:
            raise ValueError(f"declare exactly the {name} settings")
    for name in ["holdout","development"]:
        for k in ["block_size_m","buffer_m"]:
            if isinstance(cfg[name][k],bool) or not np.isfinite(cfg[name][k]) or cfg[name][k]<(1 if k=="block_size_m" else 0):
                raise ValueError("positive blocks and nonnegative buffers required")
        _positive_integer(cfg[name]["seed"],"seed",zero=True)
    if not 0<cfg["holdout"]["fraction"]<.5:
        raise ValueError("holdout fraction must lie in (0,.5)")
    if _positive_integer(cfg["development"]["folds"],"folds")<2:
        raise ValueError("at least two development folds required")
    _positive_integer(cfg["bootstrap"]["replicates"],"bootstrap replicates")
    _positive_integer(cfg["bootstrap"]["seed"],"bootstrap seed",zero=True)
    _positive_integer(cfg["reml"]["maxiter"],"optimizer iterations")
    for key in ["length_bounds_m","structured_std_bounds_log10","nugget_std_bounds_log10"]:
        bounds=cfg["reml"][key]
        if not isinstance(bounds,list) or len(bounds)!=2 or not 0<bounds[0]<bounds[1] or not np.all(np.isfinite(bounds)):
            raise ValueError("finite positive increasing parameter bounds required")
    starts=cfg["reml"]["initial_lengths_m"]
    if not isinstance(starts,list) or not starts or any(not np.isfinite(v) or v<=0 for v in starts):
        raise ValueError("positive optimizer starting lengths required")
    bounds=np.asarray(cfg["coordinate_bounds_xy_m"],dtype=float)
    if bounds.shape!=(2,2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError("coordinate bounds must be [[xmin,xmax],[ymin,ymax]]")
    bins=cfg["distance_bins_m"]
    if not isinstance(bins,list) or not bins or bins[0]!=0 or sorted(set(bins))!=bins or not np.all(np.isfinite(bins)):
        raise ValueError("increasing nonnegative distance-bin edges beginning at zero required")
    if cfg["units_assumption"]!="hydraulic_conductivity_m_per_s":
        raise ValueError("validate original hydraulic conductivity in m/s, not permeability")
    r=cfg["rules"]
    _positive_integer(r["minimum_test_blocks"],"minimum test blocks")
    if not 0<r["coverage90_min"]<r["coverage90_max"]<=1 or not 0<r["standardized_rmse_min"]<r["standardized_rmse_max"]:
        raise ValueError("increasing positive diagnostic intervals required")
    if not 0<=r["minimum_relative_crps_improvement"]<1 or r["absolute_standardized_bias_max"]<0:
        raise ValueError("valid bias/skill tolerances required")
    if not isinstance(r["require_paired_bootstrap_improvement"],bool):
        raise ValueError("bootstrap requirement must be boolean")
    return source,cfg


def read_measurements(source,cfg):
    path=_relative_path(cfg["measurements"],source.parent)
    data,qc=load_munich_hydraulic_conductivity_measurements(path,sheet_name=cfg["sheet_name"],
        stratigraphy=cfg["stratigraphy"],groundwater_state=cfg["groundwater_state"])
    rows=data.to_rows();x=data.coordinates_xy_m;y=data.log10_hydraulic_conductivity
    limits=np.asarray(cfg["coordinate_bounds_xy_m"])
    keep=np.all((x>=limits[:,0])&(x<limits[:,1]),axis=1)
    excluded=[row for row,k in zip(rows,keep) if not k]
    rows=[row for row,k in zip(rows,keep) if k];x=x[keep];y=y[keep]
    _,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
    duplicates=[[rows[j] for j in np.flatnonzero(inverse==i)] for i in np.flatnonzero(counts>1)]
    return path,rows,x,y,{"source_loader":qc,"coordinate_excluded":excluded,"n_retained":len(x),
        "n_unique_locations":len(counts),"colocated_records":duplicates,
        "coordinate_policy":"declared plausibility bounds; source coordinates never repaired or rounded",
        "duplicate_policy":"retain readings, keep colocated data together in folds, fit positive residual nugget; not exact duplicate constraints",
        "high_values_policy":"retain all positive semantically eligible readings; no clipping or upper-value exclusion",
        "coordinate_assumption":cfg["coordinate_assumption"],"units_assumption":cfg["units_assumption"],
        "missing_metadata":["authoritative workbook CRS declaration","depth/screen interval/test support","instrument error","historical interpolation lineage"]}


def source_hashes(source,workbook):
    files=[source,workbook,*sorted(Path(__file__).parents[1].rglob("*.py"))]
    return {str(p):sha256_file(p) for p in files}


def plan(source,cfg,output):
    if output.exists():
        raise ValueError("output already exists; resume with a specific stage")
    workbook,rows,x,y,qc=read_measurements(source,cfg)
    h=cfg["holdout"]
    partition=spatial_partition(x,block_size_m=h["block_size_m"],test_fraction=h["fraction"],buffer_m=h["buffer_m"],seed=h["seed"])
    output.mkdir(parents=True)
    sources=source_hashes(source,workbook)
    manifest={"schema_version":1,"study_id":_identity({"cfg":cfg,"sources":sources},"measurement-model-"),
        "status":"planned","configuration":cfg,"source_sha256":sources,"code_commit":git_head(source.parent),
        "runtime":runtime_versions(),"target":"new observation log10(Kh / [m/s]) at held-out regional measurement locations",
        "data_history":"retrospective validation: workbook and marginal diagnostics were inspected in earlier project work",
        "model_selection":"development-only equal-block CRPS; legacy procedure separately selected by development point RMSE",
        "geological_field_sampling_validated":False,"lgcnn_evaluations":0,"raster_values_used":False}
    atomic(output/"manifest.json",manifest);atomic(output/"measurement_qc.json",qc)
    atomic(output/"measurements.json",rows);np.savez(output/"partition.npz",**partition)
    archive=output/"source_snapshot";archive.mkdir()
    index={}
    for p,hsh in sources.items():
        path=Path(p)
        if path.suffix==".py":
            rel=Path("subsurface_uq")/path.relative_to(Path(__file__).parents[1])
        elif path==source:
            rel=Path("configuration.yaml")
        else:
            continue  # Preserve the read-only workbook in place, never duplicate it into code snapshots.
        target=archive/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        index[p]={"relative_archive":str(rel),"sha256":hsh}
    atomic(archive/"index.json",index)
    commit(output,["measurement_qc.json","measurements.json","partition.npz"],"planning_files.json")
    return manifest


def load_data(output):
    if not complete(output,"planning_files.json"):
        raise ValueError("missing frozen measurements/partition")
    rows=json.loads((output/"measurements.json").read_text())
    x=np.array([[r["x_m"],r["y_m"]] for r in rows]);y=np.array([r["log10_hydraulic_conductivity"] for r in rows])
    with np.load(output/"partition.npz") as f:
        partition={k:f[k].copy() for k in f.files}
    return rows,x,y,partition


def develop(cfg,output,manifest):
    if complete(output,"development_files.json"):
        return
    _,x,y,partition=load_data(output);dev=partition["development"]
    manifest["status"]="developing";atomic(output/"manifest.json",manifest)
    history=[]
    def progress(record):
        history.append(record);atomic(output/"development_progress.json",history)
        print(f"measurement development fold {record['fold']+1}: {record['model']}"+
              (" unavailable" if "failure" in record else " complete"),flush=True)
    selection=development_cv(x[dev],y[dev],cfg,on_fold=progress)
    selection["development_original_indices"]=np.flatnonzero(dev).tolist()
    # Persist selection before fitting/scoring either outer held-out choice.
    atomic(output/"development.json",selection)
    commit(output,["development.json","development_progress.json"],"development_files.json")
    manifest.update(status="selected",selected_model=selection["selected_model"],selected_legacy=selection["selected_legacy"])
    atomic(output/"manifest.json",manifest)


def assess_observation_model(scores,baseline_scores,paired,rules):
    p=scores["point_weighted"]
    skill=1-scores["equal_block_mean_crps_log10"]/baseline_scores["equal_block_mean_crps_log10"]
    checks={"enough_spatial_blocks":len(scores["blocks"])>=rules["minimum_test_blocks"],
        "coverage90":rules["coverage90_min"]<=scores["equal_block_coverage90"]<=rules["coverage90_max"],
        "standardized_rmse":rules["standardized_rmse_min"]<=p["standardized_rmse"]<=rules["standardized_rmse_max"],
        "standardized_bias":abs(p["standardized_bias"])<=rules["absolute_standardized_bias_max"],
        "crps_skill":skill>=rules["minimum_relative_crps_improvement"],
        "paired_block_improvement":not rules["require_paired_bootstrap_improvement"] or paired["ci95"][1]<0.}
    return {"relative_equal_block_crps_improvement":float(skill),"checks":checks,"rules":rules,
            "passes_exploratory_observation_checks":all(checks.values()),
            "latent_field_calibrated":False,
            "limits":"descriptive predeclared tolerances on a retrospective spatial holdout, not a geological certification"}


def evaluate(cfg,output,manifest):
    if not complete(output,"development_files.json"):
        raise ValueError("freeze development selection before holdout evaluation")
    if complete(output,"evaluation_files.json"):
        return
    rows,x,y,partition=load_data(output);dev=partition["development"];test=partition["test"]
    selected=json.loads((output/"development.json").read_text())
    names=list(dict.fromkeys(["iid_normal",selected["selected_legacy"],selected["selected_model"]]))
    results={};fits={}
    for name in names:
        model=fit_model(name,x[dev],y[dev],cfg)
        mean,variance,latent=model.predict(x[test])
        if np.any(variance<=0) or not np.all(np.isfinite([mean,variance])):
            raise ArithmeticError("nonpositive/nonfinite holdout predictive variance")
        scores=spatial_scores(y[test],mean,np.sqrt(variance),partition["block_coordinates"][test])
        results[name]={"mean":mean.tolist(),"predictive_std":np.sqrt(variance).tolist(),
            "latent_std":np.sqrt(latent).tolist() if latent is not None else None,"scores":scores}
        fits[name]={"model":name,"parameters":model.parameters,"training_original_indices":np.flatnonzero(dev).tolist()}
        atomic(output/"holdout_fit_progress.json",fits)
        print(f"measurement outer holdout: {name} complete",flush=True)
    baseline=results["iid_normal"];distance=cdist(x[test],x[dev]).min(axis=1)
    for name in names:
        r=results[name]
        paired=paired_block_bootstrap(y[test],np.array(r["mean"]),np.array(r["predictive_std"]),
            np.array(baseline["mean"]),np.array(baseline["predictive_std"]),partition["block_coordinates"][test],
            replicates=cfg["bootstrap"]["replicates"],seed=cfg["bootstrap"]["seed"])
        r["paired_crps_vs_iid"]=paired
        r["assessment"]=assess_observation_model(r["scores"],baseline["scores"],paired,cfg["rules"])
        edges=np.array([*cfg["distance_bins_m"],np.inf]);bins=[]
        for a,b in zip(edges[:-1],edges[1:]):
            mask=(distance>=a)&(distance<b)
            if mask.any():
                bins.append({"lower_m":a,"upper_m":b,**predictive_scores(y[test][mask],np.array(r["mean"])[mask],np.array(r["predictive_std"])[mask])})
        r["distance_bin_scores"]=bins
    payload={"models":results,"selected_model":selected["selected_model"],"selected_legacy":selected["selected_legacy"],
        "test_original_indices":np.flatnonzero(test).tolist(),"truth_log10":y[test].tolist(),
        "test_coordinates_xy_m":x[test].tolist(),"nearest_training_distance_m":distance.tolist(),
        "n_development":int(dev.sum()),"n_test":int(test.sum()),"n_buffer_excluded":int(partition["buffer_excluded"].sum()),
        "test_block_coordinates":partition["block_coordinates"][test].tolist(),
        "fits":fits,"selection_after_test_evaluation":None,"geological_field_sampling_validated":False,
        "field_sampling_requirements":["decide observation support and physical meaning of nugget",
            "validate latitude/longitude/projected CRS and target-domain registration",
            "assess covariance-parameter and mean-model uncertainty",
            "validate the spatial joint law and LGCNN compatibility"]}
    atomic(output/"holdout.json",payload)
    # Per-observation output stays in the ignored study folder, not documentation.
    with (output/"heldout_predictions.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["model","excel_row","x_m","y_m","truth_log10_kh","mean_log10_kh","observation_sd_log10","latent_sd_log10","nearest_training_distance_m"])
        writer.writeheader()
        for name,r in results.items():
            for j,original in enumerate(np.flatnonzero(test)):
                writer.writerow({"model":name,"excel_row":rows[original]["excel_row"],"x_m":x[original,0],"y_m":x[original,1],
                    "truth_log10_kh":y[original],"mean_log10_kh":r["mean"][j],"observation_sd_log10":r["predictive_std"][j],
                    "latent_sd_log10":r["latent_std"][j] if r["latent_std"] is not None else None,"nearest_training_distance_m":distance[j]})
    commit(output,["holdout.json","holdout_fit_progress.json","heldout_predictions.csv"],"evaluation_files.json")
    manifest["status"]="evaluated";atomic(output/"manifest.json",manifest)


def figure(cfg,output):
    _,x,y,p=load_data(output);h=json.loads((output/"holdout.json").read_text());dev=json.loads((output/"development.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    for key,color,label in [("development","C0","Development"),("test","C1","Spatial test"),("buffer_excluded","0.65","Buffer exclusion")]:
        mask=p[key];axes[0,0].scatter(x[mask,0]/1000,x[mask,1]/1000,s=15,c=color,label=f"{label} ({mask.sum()})")
    axes[0,0].set(xlabel="Projected easting [km]",ylabel="Projected northing [km]",title="Frozen geographic partition; values not used for placement",aspect="equal")
    axes[0,0].legend(fontsize=8)
    selected=h["models"][h["selected_model"]];truth=np.array(h["truth_log10"]);mean=np.array(selected["mean"]);std=np.array(selected["predictive_std"])
    axes[0,1].errorbar(truth,mean,yerr=ndtri(.95)*std,fmt="o",ms=3,alpha=.6,capsize=0)
    limits=[min(truth.min(),mean.min()),max(truth.max(),mean.max())]
    axes[0,1].plot(limits,limits,"k--");axes[0,1].set(xlabel="Withheld log10 Kh [m/s]",ylabel="Predicted mean and 90% observation interval",title=h["selected_model"])
    levels=np.array([.5,.8,.9,.95,.99])
    for name,r in h["models"].items():
        axes[1,0].plot(levels,[r["scores"]["point_weighted"]["reliability"][str(float(level))] for level in levels],"o-",label=name)
    axes[1,0].plot([0,1],[0,1],"k:");axes[1,0].set(xlim=(0,1),ylim=(0,1.02),xlabel="Nominal observation coverage",ylabel="Empirical coverage",title="Held-out observations; correlated points, no iid CI")
    axes[1,0].legend(fontsize=8)
    names=sorted(dev["scores"],key=lambda n:dev["scores"][n]["equal_block_mean_crps_log10"])
    axes[1,1].barh(names,[dev["scores"][n]["equal_block_mean_crps_log10"] for n in names])
    axes[1,1].invert_yaxis();axes[1,1].set(xlabel="Equal-block mean CRPS [log10]",title="Model selection uses development predictions only")
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle("Measurement-derived input model validation\nReal conductivity measurements; no raster-derived mean; observation and latent uncertainty remain distinct")
    fig.savefig(output/"validation.png",dpi=150);plt.close(fig)


def report(cfg,output,manifest):
    if not complete(output,"evaluation_files.json"):
        raise ValueError("complete held-out evaluation before reporting")
    qc=json.loads((output/"measurement_qc.json").read_text());dev=json.loads((output/"development.json").read_text());h=json.loads((output/"holdout.json").read_text())
    figure(cfg,output)
    lines=["# Measurement-derived input model validation","",f"Study: `{manifest['study_id']}`.","",
        f"Retained {qc['n_retained']} real conductivity records at {qc['n_unique_locations']} unique locations.",
        f"Development: {h['n_development']}; spatial test: {h['n_test']}; buffer exclusions: {h['n_buffer_excluded']}.","",
        "All data are log10 hydraulic conductivity in m/s. Coordinates use the declared working interpretation; absolute CRS provenance is not independently confirmed.","",
        f"Development-selected uncertainty model: **{h['selected_model']}**. Legacy procedure: **{h['selected_legacy']}**.",
        "Choices were saved before scoring the outer spatial test.","",
        "| Model | RMSE log10 | Mean CRPS, equal blocks | Observation coverage90, equal blocks | Standardized RMSE | CRPS skill vs iid | Checks pass |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for name,r in h["models"].items():
        s=r["scores"];a=r["assessment"];p=s["point_weighted"]
        lines.append(f"| {name} | {p['rmse_log10']:.4f} | {s['equal_block_mean_crps_log10']:.4f} | {s['equal_block_coverage90']:.3f} | {p['standardized_rmse']:.3f} | {a['relative_equal_block_crps_improvement']:.1%} | {a['passes_exploratory_observation_checks']} |")
    lines += ["","![Validation diagnostics](validation.png)","",
        "The checks concern new observation predictions, including an estimated unresolved residual nugget. They do not validate the smooth latent field, independent 5 m geological noise, historical RUN locations or LGCNN accuracy.","",
        "The source workbook has been inspected in earlier project work; this is a frozen retrospective spatial assessment rather than a prospective external dataset.",
        "Block bootstrap intervals are exploratory and do not remove all spatial dependence or parameter-estimation uncertainty.","",
        "See measurement_qc.json, development.json, holdout.json and heldout_predictions.csv for the full audit trail. No fields or LGCNN queries were generated."]
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    manifest.update(status="complete",observation_model_checks_pass=h["models"][h["selected_model"]]["assessment"]["passes_exploratory_observation_checks"],
        geological_field_sampling_validated=False,report_sha256={f:sha256_file(output/f) for f in ["report.md","validation.png"]})
    atomic(output/"manifest.json",manifest)


def run(config_path,output_dir,stage="all"):
    source,cfg=load_config(config_path);output=Path(output_dir).resolve()
    if stage not in {"plan","develop","evaluate","report","all"}:
        raise ValueError("unknown stage")
    if stage in {"plan","all"}:
        manifest=plan(source,cfg,output)
    else:
        manifest=json.loads((output/"manifest.json").read_text())
        workbook=_relative_path(cfg["measurements"],source.parent)
        if manifest["configuration"]!=cfg or manifest["source_sha256"]!=source_hashes(source,workbook):
            raise ValueError("validation source/configuration/installation changed; use original context or a fresh run")
        if manifest.get("report_sha256"):
            for f,h in manifest["report_sha256"].items():
                if sha256_file(output/f)!=h:raise ValueError("completed validation report changed")
    if stage in {"develop","all"}:develop(cfg,output,manifest)
    if stage in {"evaluate","all"}:evaluate(cfg,output,manifest)
    if stage in {"report","all"}:report(cfg,output,manifest)
    workbook=_relative_path(cfg["measurements"],source.parent)
    if source_hashes(source,workbook)!=manifest["source_sha256"]:
        raise ValueError("source changed during validation")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--output-dir",required=True)
    parser.add_argument("--stage",choices=("plan","develop","evaluate","report","all"),default="all")
    args=parser.parse_args(argv);run(args.config,args.output_dir,args.stage)


if __name__=="__main__":main()
