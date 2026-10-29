"""Paired joint-profile sensitivity from a completed controlled benchmark.

Outer trials are independent; pixels and paired profiles are not replicates.
Each profile conditions on its own model-consistent synthetic readings.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reference_conditioning import _write


def estimate(values):
    a=np.asarray(values,dtype=float)
    return {"mean":float(a.mean()),"outer_trial_sem":float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None,
            "outer_trial_values":a.tolist()}


def _moments(path,budget,chunk_rows):
    store=np.load(path,mmap_mode="r");mean=np.empty(store.shape[1:]);variance=np.empty_like(mean)
    for start in range(0,len(mean),chunk_rows):
        block=np.asarray(store[:budget,start:start+chunk_rows],dtype=float)
        mean[start:start+chunk_rows]=block.mean(axis=0)
        variance[start:start+chunk_rows]=block.var(axis=0,ddof=1)
    return mean,variance


def report(cfg,output,manifest):
    output=Path(output);reference=cfg["sensitivity_reference_profile"]
    designs=[("prior",0),*[(f"{kind}-{i:03d}",n) for kind in cfg["coverage_scenarios"]
                for i in range(cfg["n_designs"]) for n in cfg["measurement_counts"][1:]]]
    groups=[];contrasts=[]
    for design,n in designs:
        for budget in cfg["sample_budgets"]:
            collected={}
            for name in cfg["scenarios"]:
                trials=[]
                for trial in range(cfg["n_truths"]):
                    root=output/name/f"truth-{trial:04d}"
                    case=root/"prior" if n==0 else root/design/f"n-{n:04d}"
                    summary=json.loads((case/"summary.json").read_text())
                    inputs=json.loads((case/"input_summary.json").read_text())
                    samples=np.load(case/"qoi_samples.npy")[:budget]
                    mean,var=_moments(case/"temperature_ensemble.npy",budget,cfg["quantile_chunk_rows"])
                    qois={q:{"mean":float(samples[:,j].mean()),"sd":float(samples[:,j].std(ddof=1))}
                          for j,q in enumerate(summary["qoi_names"])}
                    risk=json.loads((case/"risk.json").read_text())
                    probabilities={}
                    for threshold in risk["thresholds"]:
                        checkpoint=next(c for c in threshold["checkpoints"] if c["n_samples"]==budget)
                        for q,r in checkpoint["receptors"].items():
                            probabilities[f"{q}_warming_gt_{threshold['threshold_C']:g}C"]=r["probability"]
                    trials.append({"qois":qois,"probabilities":probabilities,"mean_field":mean,"variance_field":var,
                        "average_variance":float(var.mean()),"analytic_input_variance":inputs["analytic_average_variance_log10"]})
                collected[name]=trials
                groups.append({"profile":name,"design":design,"n_measurements":n,"n_samples":budget,
                    "temperature_average_variance_C2":estimate([t["average_variance"] for t in trials]),
                    "analytic_average_input_variance_log10":trials[0]["analytic_input_variance"],
                    "qois":{q:{stat:estimate([t["qois"][q][stat] for t in trials]) for stat in ["mean","sd"]}
                            for q in trials[0]["qois"]},
                    "receptor_exceedance_probabilities":{q:estimate([t["probabilities"][q] for t in trials])
                            for q in trials[0]["probabilities"]}})
            for name,trials in collected.items():
                if name==reference:continue
                base=collected[reference]
                contrasts.append({"profile":name,"reference_profile":reference,"design":design,"n_measurements":n,"n_samples":budget,
                    "paired_temperature_variance_difference_C2":estimate([t["average_variance"]-b["average_variance"] for t,b in zip(trials,base)]),
                    "mean_field_rms_profile_difference_C":estimate([np.sqrt(np.mean((t["mean_field"]-b["mean_field"])**2)) for t,b in zip(trials,base)]),
                    "sd_field_rms_profile_difference_C":estimate([np.sqrt(np.mean((np.sqrt(t["variance_field"])-np.sqrt(b["variance_field"]))**2)) for t,b in zip(trials,base)]),
                    "paired_qoi_differences":{q:{stat:estimate([t["qois"][q][stat]-b["qois"][q][stat] for t,b in zip(trials,base)])
                        for stat in ["mean","sd"]} for q in trials[0]["qois"]},
                    "paired_receptor_probability_differences":{q:estimate([t["probabilities"][q]-b["probabilities"][q] for t,b in zip(trials,base)])
                        for q in trials[0]["probabilities"]}})
    reductions=[]
    for g in groups:
        if g["n_measurements"]==0:continue
        prior=next(p for p in groups if p["profile"]==g["profile"] and p["n_measurements"]==0 and p["n_samples"]==g["n_samples"])
        before=prior["temperature_average_variance_C2"]["outer_trial_values"]
        after=g["temperature_average_variance_C2"]["outer_trial_values"]
        reductions.append({"profile":g["profile"],"design":g["design"],"n_measurements":g["n_measurements"],"n_samples":g["n_samples"],
            "analytic_input_variance_fraction_remaining":g["analytic_average_input_variance_log10"]/prior["analytic_average_input_variance_log10"],
            "temperature_variance_fraction_remaining":g["temperature_average_variance_C2"]["mean"]/prior["temperature_average_variance_C2"]["mean"] if prior["temperature_average_variance_C2"]["mean"] else None,
            "paired_temperature_variance_reduction_C2":estimate(np.asarray(before)-np.asarray(after))})
    payload={"study_id":manifest["study_id"],"reference_profile":reference,"independent_outer_trials":cfg["n_truths"],
        "profile_probability_weights":None,"profile_groups":groups,"paired_profile_contrasts":contrasts,"conditioning_reductions":reductions,
        "coupling":"same FFT white Gaussian modes, geometry and sample indices across profiles; each profile has its own exact synthetic readings",
        "interpretation":"joint-profile sensitivity of the frozen LGCNN under declared input laws; no causal attribution to individual parameters",
        "limits":"outer-trial SEM includes finite inner MC variability; correlated pixels/profiles/draws across budgets are not extra independent trials; no geological or simulator calibration; no tail-convergence claim"}
    _write(output/"profile_sensitivity.json",payload)
    _figure(cfg,payload,output/"profile_sensitivity.png")
    return payload


def _figure(cfg,payload,path):
    groups=[g for g in payload["profile_groups"] if g["n_samples"]==cfg["n_samples"]]
    designs=list(dict.fromkeys((g["design"],g["n_measurements"]) for g in groups))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
    profiles=list(cfg["scenarios"]);x=np.arange(len(profiles));width=.7/len(designs)
    for j,(design,n) in enumerate(designs):
        rows=[next(g for g in groups if g["profile"]==p and (g["design"],g["n_measurements"])==(design,n)) for p in profiles]
        pos=x+(j-(len(designs)-1)/2)*width
        def bars(ax,values):
            ax.bar(pos,[v["mean"] for v in values],width=width,label=f"{design}, n={n}",
                yerr=[v["outer_trial_sem"] or 0 for v in values],capsize=3,alpha=.85)
        bars(axes[0],[r["temperature_average_variance_C2"] for r in rows])
        qoi="receptor_001"
        bars(axes[1],[r["qois"][qoi]["sd"] for r in rows])
        area=next(q for q in rows[0]["qois"] if q.startswith("warming_area_gt_"))
        bars(axes[2],[r["qois"][area]["mean"] for r in rows])
    titles=[("Mean pointwise T variance","C²"),("Receptor 1 T standard deviation","C"),
            (f"Mean warming area >{cfg['risk_thresholds_C'][0]:g} C","m²")]
    for ax,(title,ylabel) in zip(axes,titles):
        ax.set(title=title,ylabel=ylabel);ax.set_xticks(x,[p.replace("_structural","") for p in profiles]);ax.grid(axis="y",alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Structural-profile sensitivity: {cfg['n_truths']} independent paired trials, {cfg['n_samples']} draws/case\nError bars: outer-trial SEM including finite MC variation; separate laws, no geological weights")
    fig.savefig(path,dpi=140);plt.close(fig)
