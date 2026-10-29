"""Controlled conditional-field benchmark: known-law truths -> exact data -> LGCNN.

The independent outer trial is a hidden field drawn from the declared prior.
Within a trial, nested data and common MC draws compare information budgets.
RUN reconstruction remains a separate empirical realism challenge.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.linalg import cho_solve
from scipy.stats import binomtest
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.controlled_fields import KnownMeanConditioner, physical_permeability
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from ..sampling.measurement_density import coverage_locations
from ..statistics.calibration import ensemble_truth_scores, calibration_summary
from ..statistics.online import OnlineFieldStatistics
from ..statistics.temperature_risk import scalar_temperature_qois, summarize_risk_store, validate_output_grid
from ..surrogates.frozen_temperature import load_frozen_temperature_model
from ..rq1.analysis import global_uncertainty_metrics, prefix_field_convergence, write_temperature_field_products
from ..validation.reconstruction import predictive_scores
from .conditioning_temperature import _figure
from .reference_conditioning import _identity, _name, _positive_integer, _relative_path, _write


_KEYS = {"scenarios", "shape", "cell_size_m", "permeability_units", "measurement_counts",
         "coverage_scenarios", "n_designs", "candidate_stride_cells", "cluster_scale_fraction",
         "n_truths", "n_samples", "sample_budgets", "seed", "workers", "chunk_size",
         "validation_replicates", "validation_probes", "diagnostic_stride_cells",
         "model_config", "model_asset_root", "torch_threads", "risk_thresholds_C", "quantile_chunk_rows"}
_PROFILE_KEYS = {"structural_profile_file", "profile_pairing_group", "paired_embedding_shape",
                 "sensitivity_reference_profile"}


def seed_for(seed, *labels):
    """Stable separate random streams; scenario ordering cannot change a draw."""
    return int.from_bytes(sha256(json.dumps([seed,*labels], separators=(",", ":")).encode()).digest()[:8], "little")


def array_hash(values):
    return sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".partial")
    _write(temporary, value)
    temporary.replace(path)


def _verify(root, ledger):
    for filename, digest in ledger.items():
        if sha256_file(Path(root)/filename) != digest:
            raise ValueError(f"completed benchmark artifact changed: {filename}")


def _commit(directory, filenames, marker="files.json"):
    _atomic(directory/marker, {str(f): sha256_file(directory/f) for f in filenames})


def _completed(directory, marker="files.json"):
    path = directory/marker
    if not path.exists():
        return False
    _verify(directory, json.loads(path.read_text()))
    return True


def load_config(path):
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or set(cfg) not in (_KEYS, _KEYS | _PROFILE_KEYS):
        raise ValueError("declare every controlled-benchmark key; unknown keys are forbidden")
    for key in ("n_truths", "n_samples", "n_designs", "candidate_stride_cells", "workers", "chunk_size",
                "validation_replicates", "validation_probes", "diagnostic_stride_cells", "torch_threads", "quantile_chunk_rows"):
        _positive_integer(cfg[key], key)
    _positive_integer(cfg["seed"], "seed", zero=True)
    if cfg["n_samples"] < 4 or cfg["validation_replicates"] < 20:
        raise ValueError(">=4 MC draws and >=20 independent dense validation trials required")
    shape = cfg["shape"]
    if not isinstance(shape, list) or len(shape) != 2 or any(_positive_integer(v,"shape") < 3 for v in shape):
        raise ValueError("shape must be two integer dimensions >=3")
    if cfg["cell_size_m"] != 5. or cfg["permeability_units"] != "m2":
        raise ValueError("native 5 m grid and permeability in m2 required")
    if "structural_profile_file" in cfg:
        _name(cfg["profile_pairing_group"], "profile pairing group")
        embedding=cfg["paired_embedding_shape"]
        if (not isinstance(embedding,list) or len(embedding)!=2
                or any(isinstance(n,bool) or not isinstance(n,int) or n<3*m or n&(n-1)
                       for n,m in zip(embedding,shape))):
            raise ValueError("paired embedding requires powers of two >=3 times the crop")
        from .structural_profiles import verify_artifact
        artifact,_=verify_artifact(_relative_path(cfg["structural_profile_file"],source.parent))
        expected={name:p["law"] for name,p in artifact["profiles"].items()}
        if (cfg["scenarios"]!=expected or shape!=artifact["native_shape"]
                or cfg["cell_size_m"]!=artifact["cell_size_m"]):
            raise ValueError("scenario laws/grid must exactly match the frozen structural profile artifact")
        if cfg["sensitivity_reference_profile"] not in expected:
            raise ValueError("declare a sensitivity reference from the structural profiles")
    counts = cfg["measurement_counts"]
    if (not isinstance(counts,list) or len(counts)<2 or counts[0]!=0
            or any(_positive_integer(v,"measurement count",zero=True)!=v for v in counts)
            or sorted(set(counts))!=counts):
        raise ValueError("strictly increasing counts beginning at zero required")
    budgets = cfg["sample_budgets"]
    if (not isinstance(budgets,list) or not budgets or sorted(set(budgets))!=budgets
            or any(_positive_integer(v,"sample budget") < 2 for v in budgets) or budgets[-1]!=cfg["n_samples"]):
        raise ValueError("nested sample budgets >=2 ending at n_samples required")
    kinds = cfg["coverage_scenarios"]
    if (not isinstance(kinds,list) or not kinds or len(set(kinds))!=len(kinds)
            or not set(kinds)<={"balanced","clustered"}):
        raise ValueError("distinct balanced/clustered coverage scenarios required")
    if not np.isfinite(cfg["cluster_scale_fraction"]) or not 0 < cfg["cluster_scale_fraction"] < 1:
        raise ValueError("cluster scale must lie in (0,1)")
    if not isinstance(cfg["scenarios"],dict) or not cfg["scenarios"]:
        raise ValueError("explicit named geological scenarios required")
    for name,law in cfg["scenarios"].items():
        _name(name,"scenario")
        if not isinstance(law,dict) or set(law)!={"mean_log10_k", "covariance"}:
            raise ValueError("each law requires mean_log10_k and covariance")
        if isinstance(law["mean_log10_k"],bool) or not np.isfinite(law["mean_log10_k"]):
            raise ValueError("known log10 mean must be finite")
        spec = GSToolsReferenceSpec(**law["covariance"])
        if spec.center != "median":
            raise ValueError("Y has the declared Gaussian mean; use center=median")
        # Validate geometry/rank before creating output or loading a model.
        for kind in kinds:
            for design in range(cfg["n_designs"]):
                indices,_ = _locations(cfg,name,kind,design)
                KnownMeanConditioner(shape,spec,indices,mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
    if (not isinstance(cfg["risk_thresholds_C"],list) or not cfg["risk_thresholds_C"]
            or any(isinstance(v,bool) or not np.isfinite(v) or v<=0 for v in cfg["risk_thresholds_C"])
            or len(set(cfg["risk_thresholds_C"]))!=len(cfg["risk_thresholds_C"])):
        raise ValueError("distinct positive warming thresholds required")
    if cfg["model_config"] is not None:
        _relative_path(cfg["model_config"],source.parent)
    _relative_path(cfg["model_asset_root"],source.parent)
    return source,cfg


def _stream(cfg,name):
    return cfg.get("profile_pairing_group",name)


def _prior(cfg,spec):
    return GSToolsReferenceFFTSampler(replace(spec,sigma_R=1.),cfg["shape"],
        cell_size_m=cfg["cell_size_m"],workers=cfg["workers"],embedding_shape=cfg.get("paired_embedding_shape"))


def _locations(cfg,name,kind,design):
    return coverage_locations(cfg["shape"],cfg["measurement_counts"][-1],stride=cfg["candidate_stride_cells"],
        seed=seed_for(cfg["seed"],_stream(cfg,name),"layout",kind,design),kind=kind,cluster_scale=cfg["cluster_scale_fraction"])


def _designs(cfg,name):
    for kind in cfg["coverage_scenarios"]:
        for i in range(cfg["n_designs"]):
            yield f"{kind}-{i:03d}",kind,i


def _cases(cfg,name,trial,output):
    root = output/name/f"truth-{trial:04d}"
    yield "prior",0,root/"prior"
    for design,_,_ in _designs(cfg,name):
        for n in cfg["measurement_counts"][1:]:
            yield design,n,root/design/f"n-{n:04d}"


def _probes(cfg,name):
    rng = np.random.default_rng(seed_for(cfg["seed"],_stream(cfg,name),"calibration-probes"))
    all_data = np.concatenate([_locations(cfg,name,k,i)[0] for _,k,i in _designs(cfg,name)])
    forbidden = np.ravel_multi_index(all_data.T,cfg["shape"])
    # A fixed random geometry-only subset, excluding all candidate observations.
    eligible = np.setdiff1d(np.arange(np.prod(cfg["shape"])),forbidden)
    if len(eligible)<cfg["validation_probes"]:
        raise ValueError("not enough unobserved calibration probe cells")
    flat = rng.choice(eligible,cfg["validation_probes"],replace=False)
    return np.column_stack(np.unravel_index(flat,cfg["shape"]))


def _sources(source,cfg):
    files = [source,*sorted(Path(__file__).parents[1].rglob("*.py"))]
    if cfg["model_config"] is not None:
        files.append(_relative_path(cfg["model_config"],source.parent))
    result={str(p):sha256_file(p) for p in files}
    if "structural_profile_file" in cfg:
        from .structural_profiles import verify_artifact
        _,extra=verify_artifact(_relative_path(cfg["structural_profile_file"],source.parent))
        result.update(extra)
    return result


def plan(source,cfg,output):
    if output.exists():
        raise ValueError("output already exists; resume with a specific stage")
    for name in cfg["scenarios"]:
        _probes(cfg,name)
    output.mkdir(parents=True)
    ncases = 1 + len(cfg["coverage_scenarios"])*cfg["n_designs"]*(len(cfg["measurement_counts"])-1)
    calls = len(cfg["scenarios"])*cfg["n_truths"]*(1+cfg["n_samples"]*ncases)
    manifest = {"schema_version":1,"study_id":_identity({"cfg":cfg,"sources":_sources(source,cfg)},"controlled-"),
        "status":"planned", "configuration":cfg,"source_sha256":_sources(source,cfg),"code_commit":git_head(source.parent),
        "runtime":runtime_versions(),"planned_model_evaluations":calls,
        "planned_cases":len(cfg["scenarios"])*cfg["n_truths"]*ncases,
        "interpretation":"controlled model-consistency benchmark with a known mean and covariance; not a calibrated site posterior",
        "uncertainty_included":["permeability field conditional on exact synthetic readings"],
        "uncertainty_not_quantified":["geological parameter/model uncertainty","LGCNN weight uncertainty","LGCNN versus simulator discrepancy"],
        "scenario_pooling":None,"layout_pooling":None,"truth_reference":"independent prior field evaluated through the SAME frozen LGCNN",
        "sample_filtering":None,"measurement_error_variance":0.,
        "seed_contract":"SHA256 semantic streams; independent truth/posterior streams per scenario and outer trial; shared MC within trial across counts/layouts",
        "realism_test":"configs/measurement_density.validation.yaml; RUN withheld results remain separate and unchanged"}
    if "structural_profile_file" in cfg:
        manifest.update(structural_profile_file=cfg["structural_profile_file"],
            profile_probability_weights=None,profile_pairing_group=cfg["profile_pairing_group"],
            paired_embedding_shape=cfg["paired_embedding_shape"],
            seed_contract="common FFT Gaussian driving modes across profiles; shared truth/posterior seeds and geometry by trial; truth and posterior independent; outer trials independent",
            sensitivity_interpretation="paired joint-profile contrasts; different profiles have their own synthetic data; no parameter-wise attribution or geological probability mixture")
    _atomic(output/"manifest.json",manifest)
    return manifest


def validate(cfg,output,manifest):
    """Independent dense joint-Gaussian marginals at native physical coordinates.

    This checks conditioning and finite-ensemble calibration; it is not a native
    LGCNN run or an independent geological validation of the assumed law.
    """
    target = output/"dense_validation"
    if _completed(target):
        return
    target.mkdir(exist_ok=True)
    rows=[]; maximum_error=0.
    for name,law in cfg["scenarios"].items():
        spec=GSToolsReferenceSpec(**law["covariance"]);probes=_probes(cfg,name)
        for design,kind,index in _designs(cfg,name):
            data,_=_locations(cfg,name,kind,index)
            system=KnownMeanConditioner(cfg["shape"],spec,data,mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
            joint=np.concatenate((data,probes));offsets=(joint[:,None]-joint[None,:])*cfg["cell_size_m"]
            c=spec.correlation(offsets[...,0],offsets[...,1]);chol=np.linalg.cholesky(c)
            rng=np.random.default_rng(seed_for(cfg["seed"],name,design,"dense-validation"))
            t=cfg["validation_replicates"];s=cfg["n_samples"];d=len(data)
            truths=law["mean_log10_k"]+spec.sigma_R*(rng.standard_normal((t,len(joint)))@chol.T)
            priors=rng.standard_normal((t,s,len(joint)))@chol.T
            f=system.features(probes)
            for n in cfg["measurement_counts"]:
                scores=[[] for _ in probes];covered=np.zeros(len(probes),dtype=int)
                for trial in range(t):
                    readings=truths[trial,:n]
                    mean,var=system.moments(f,readings,count=n)
                    weights=(cho_solve((chol[:n,:n],True),c[:n,d:]).T if n else np.empty((len(probes),0)))
                    direct_mean=law["mean_log10_k"]+weights@(readings-law["mean_log10_k"])
                    direct_var=spec.sigma_R**2*(1.-np.sum(weights*c[d:,:n],axis=1))
                    maximum_error=max(maximum_error,float(np.max(np.abs(mean-direct_mean))),float(np.max(np.abs(var-direct_var))))
                    # Independently generated dense joint prior; production Matheron coefficients.
                    draws=np.empty((s,len(probes)))
                    for j in range(s):
                        coefficient=system.update_coefficients(readings,priors[trial,j,:n],count=n)
                        draws[j]=law["mean_log10_k"]+spec.sigma_R*priors[trial,j,d:]+f[:,:n]@coefficient
                    covered += np.abs(truths[trial,d:]-mean) <= 1.6448536269514722*np.sqrt(np.maximum(var,0.))
                    for j in range(len(probes)):
                        scores[j].append(ensemble_truth_scores(truths[trial,d+j],draws[:,j],rng=rng))
                for j in range(len(probes)):
                    ci=binomtest(int(covered[j]),t).proportion_ci(method="exact")
                    rows.append({"scenario":name,"design":design,"n_measurements":n,"probe":j,
                        "cell":probes[j].tolist(),"analytic_coverage90":float(covered[j]/t),
                        "analytic_coverage90_ci95":[float(ci.low),float(ci.high)],**calibration_summary(scores[j])})
    if maximum_error>1.e-8:
        raise ArithmeticError("known-law conditioner disagrees with independent dense Gaussian equations")
    _atomic(target/"summary.json",{"rows":rows,"maximum_dense_equation_error":maximum_error,
        "independent_truths_per_row":cfg["validation_replicates"],"sampler":"independent dense joint Gaussian at native coordinates",
        "coverage_policy":"descriptive diagnostics at fixed probes; no tuning to the generated truths or pass-by-selection",
        "limits":"validates Gaussian conditioning algebra and model-consistent marginal ranks, not full-grid FFT or LGCNN physical accuracy"})
    _commit(target,["summary.json"])
    manifest["status"]="dense_validated";_atomic(output/"manifest.json",manifest)


def _geometry(cfg,name,law,output):
    systems={}
    for design,kind,index in _designs(cfg,name):
        directory=output/name/"geometry"/design;directory.mkdir(parents=True,exist_ok=True)
        indices,meta=_locations(cfg,name,kind,index)
        system=KnownMeanConditioner(cfg["shape"],GSToolsReferenceSpec(**law["covariance"]),indices,
            mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
        if not _completed(directory):
            features=open_memmap(directory/"features.npy",mode="w+",dtype=np.float64,
                                 shape=(int(np.prod(cfg["shape"])),len(indices)))
            for start in range(0,len(features),cfg["chunk_size"]):
                stop=min(start+cfg["chunk_size"],len(features))
                queries=np.column_stack(np.unravel_index(np.arange(start,stop),cfg["shape"]))
                features[start:stop]=system.features(queries)
            features.flush();del features
            _atomic(directory/"design.json",{"indices":indices.tolist(),"metadata":meta,
                "condition_number":system.condition_number,"known_mean_log10":law["mean_log10_k"]})
            _commit(directory,["features.npy","design.json"])
        systems[design]=(system,np.load(directory/"features.npy",mmap_mode="r"))
    return systems


def _prepare_case(cfg,name,trial,design,n,directory,truth,system,features):
    directory.mkdir(parents=True,exist_ok=True)
    if _completed(directory,"input_files.json"):
        return
    readings=truth[tuple(system.indices[:n].T)]
    means=open_memmap(directory/"posterior_log10_mean.npy",mode="w+",dtype=np.float32,shape=tuple(cfg["shape"]))
    stds=open_memmap(directory/"posterior_log10_std.npy",mode="w+",dtype=np.float32,shape=tuple(cfg["shape"]))
    variance_sum=0.
    for start in range(0,truth.size,cfg["chunk_size"]):
        stop=min(start+cfg["chunk_size"],truth.size)
        f=(features[start:stop] if features is not None else np.empty((stop-start,0)))
        mean,var=system.moments(f,readings,count=n)
        means.ravel()[start:stop]=mean;stds.ravel()[start:stop]=np.sqrt(var)
        variance_sum+=float(var.sum())
    means.flush();stds.flush()
    stride=cfg["diagnostic_stride_cells"]
    rr,cc=np.meshgrid(np.arange(0,cfg["shape"][0],stride),np.arange(0,cfg["shape"][1],stride),indexing="ij")
    queries=np.column_stack((rr.ravel(),cc.ravel()))
    data_cells=set(map(tuple,system.indices[:n]));queries=np.array([q for q in queries if tuple(q) not in data_cells],dtype=int)
    if not len(queries):
        raise ValueError("diagnostic grid contains no withheld cells")
    m,v=system.moments(system.features(queries),readings,count=n)
    scores=predictive_scores(truth[tuple(queries.T)],m,np.sqrt(v))
    _atomic(directory/"input_summary.json",{"scenario":name,"trial":trial,"design":design,"n_measurements":n,
        "indices":system.indices[:n].tolist(),"observed_log10_k":readings.tolist(),
        "known_mean_log10":system.mean_log10,"covariance":asdict(system.spec),
        "analytic_average_variance_log10":variance_sum/truth.size,
        "withheld_descriptive_scores":scores,"diagnostic_grid_stride_cells":stride,
        "limits":"withheld pixel scores are descriptive within one correlated field; independent outer truths are the calibration units",
        "hidden_unobserved_truth_used_for_conditioning":False,"sample_filtering":None})
    del means,stds
    _commit(directory,["posterior_log10_mean.npy","posterior_log10_std.npy","input_summary.json"],"input_files.json")


def prepare(cfg,output,manifest):
    if not _completed(output/"dense_validation"):
        raise ValueError("complete dense validation before preparing native inputs")
    manifest["status"]="preparing_inputs";_atomic(output/"manifest.json",manifest)
    for name,law in cfg["scenarios"].items():
        systems=_geometry(cfg,name,law,output)
        spec=GSToolsReferenceSpec(**law["covariance"])
        prior=None
        empty=KnownMeanConditioner(cfg["shape"],spec,np.empty((0,2),int),mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
        for trial in range(cfg["n_truths"]):
            root=output/name/f"truth-{trial:04d}";root.mkdir(exist_ok=True)
            if not _completed(root,"truth_files.json"):
                if prior is None:
                    prior=_prior(cfg,spec)
                    _atomic(output/name/"prior.json",{**prior.metadata,
                        "spec":{**prior.metadata["spec"],"marginal":"zero-mean unit Gaussian residual",
                            "assumption_status":"known-law controlled experiment; not a fitted geological uncertainty model"},"law":law,
                        "interpretation":"known stationary Gaussian law; no heterogeneous reference is used"})
                seed=seed_for(cfg["seed"],_stream(cfg,name),trial,"hidden-truth")
                logs=law["mean_log10_k"]+spec.sigma_R*prior.sample_residual(seed)
                np.save(root/"truth_log10.npy",logs);np.save(root/"truth_permeability.npy",physical_permeability(logs))
                _atomic(root/"truth.json",{"seed":seed,"trial":trial,"scenario":name,"role":"independent hidden truth; only selected cells are revealed"})
                _commit(root,["truth_log10.npy","truth_permeability.npy","truth.json"],"truth_files.json")
            truth=np.load(root/"truth_log10.npy",mmap_mode="r")
            for design,n,directory in _cases(cfg,name,trial,output):
                system,features=(empty,None) if n==0 else systems[design]
                _prepare_case(cfg,name,trial,design,n,directory,truth,system,features)
            del truth
            print(f"controlled inputs {name}: hidden truth {trial+1}/{cfg['n_truths']}",flush=True)
        del systems,prior
    manifest["status"]="inputs_ready";_atomic(output/"manifest.json",manifest)


def _qois(temperature,model,cfg):
    q=scalar_temperature_qois(temperature,receptors=model.receptors,roi=model.mean_anomaly_roi,background=model.background_temperature)
    if model.mean_anomaly_roi is not None:
        r0,r1,c0,c1=model.mean_anomaly_roi
        q["roi_maximum_anomaly_C"]=float(np.max(temperature[r0:r1,c0:c1])-model.background_temperature)
    delta=temperature-np.asarray(model.background_temperature,dtype=temperature.dtype)
    for threshold in cfg["risk_thresholds_C"]:
        q[f"warming_area_gt_{threshold:g}C_m2"]=float(np.count_nonzero(delta>np.asarray(threshold,dtype=delta.dtype))*model.cell_size_m**2)
    return q


def _truth_temperature(root,model):
    field=np.load(root/"truth_permeability.npy",mmap_mode="r")
    if not _completed(root,"truth_temperature_files.json"):
        # PyTorch must not share a read-only memory map, even though the fixed
        # adapter currently stacks inputs before normalizing them.
        start=time.monotonic();temperature=np.asarray(model.predict_temperature(np.array(field,copy=True)),dtype=np.float32)
        validate_output_grid(temperature.shape,model.receptors,model.mean_anomaly_roi)
        if not np.all(np.isfinite(temperature)):
            raise ValueError("nonfinite hidden-truth prediction")
        np.save(root/"truth_temperature.npy",temperature)
        _atomic(root/"truth_temperature.json",{"input_sha256":array_hash(field),"seconds":time.monotonic()-start,
            "normalization":model.normalization_diagnostics(field),"output_shape":list(temperature.shape),
            "output_offset_in_input_cells":[(m-n)//2 for m,n in zip(field.shape,temperature.shape)],
            "interpretation":"hidden truth through the same frozen LGCNN, not PFLOTRAN truth"})
        _commit(root,["truth_temperature.npy","truth_temperature.json"],"truth_temperature_files.json")
    return np.load(root/"truth_temperature.npy")


def _prediction_state(directory,cfg,shape):
    path=directory/"prediction_records.json"
    records=json.loads(path.read_text()) if path.exists() else []
    store=directory/"temperature_ensemble.npy"
    if records and not store.exists():
        raise ValueError("prediction records exist without temperature cache")
    mode="r+" if store.exists() else "w+"
    values=open_memmap(store,mode=mode,dtype=np.float32,shape=(cfg["n_samples"],*shape))
    if values.shape!=(cfg["n_samples"],*shape) or values.dtype!=np.float32:
        raise ValueError("prediction cache shape/dtype changed")
    if len(records)>cfg["n_samples"] or any(r["sample_index"]!=i for i,r in enumerate(records)):
        raise ValueError("invalid prediction recovery sequence")
    for i,record in enumerate(records):
        if array_hash(values[i])!=record["temperature_sha256"]:
            raise ValueError("completed prediction cache changed")
    return records,values


def _summarize(cfg,name,trial,design,n,directory,truth_temperature,truth_logs,model,probes):
    if _completed(directory,"result_files.json"):
        return
    store=directory/"temperature_ensemble.npy";ensemble=np.load(store,mmap_mode="r")
    records=json.loads((directory/"prediction_records.json").read_text())
    if len(records)!=cfg["n_samples"]:
        raise ValueError("incomplete temperature case")
    stats=OnlineFieldStatistics()
    for temperature in ensemble:
        stats.update(temperature)
    final=stats.finalize()
    products=write_temperature_field_products(temperature_store=store,final_mean=final.mean,final_std=final.std,
        output_directory=directory,chunk_rows=cfg["quantile_chunk_rows"])
    risk=summarize_risk_store(store,directory/"risk",thresholds=cfg["risk_thresholds_C"],background=model.background_temperature,
        receptors=model.receptors,cell_size_m=model.cell_size_m,budgets=cfg["sample_budgets"],chunk_rows=cfg["quantile_chunk_rows"])
    _atomic(directory/"risk.json",risk)
    truth_qois=_qois(truth_temperature,model,cfg)
    names=list(truth_qois);qois=np.array([[q[name] for name in names] for q in [_qois(t,model,cfg) for t in ensemble]])
    np.save(directory/"qoi_samples.npy",qois)
    scores={}
    for budget in cfg["sample_budgets"]:
        scores[str(budget)]={q:ensemble_truth_scores(truth_qois[q],qois[:budget,j],
            rng=np.random.default_rng(seed_for(cfg["seed"],name,trial,design,n,budget,q,"rank"))) for j,q in enumerate(names)}
    input_scores={str(j):ensemble_truth_scores(float(truth_logs[tuple(cell)]),[r["input_probe_log10"][j] for r in records],
        rng=np.random.default_rng(seed_for(cfg["seed"],name,trial,design,n,j,"input-rank"))) for j,cell in enumerate(probes)}
    lo=np.load(products["temperature_q05"],mmap_mode="r");hi=np.load(products["temperature_q95"],mmap_mode="r")
    field_coverage=float(np.mean((truth_temperature>=lo)&(truth_temperature<=hi)))
    del lo,hi
    halves=[]
    for values in np.array_split(qois,2):
        halves.append({q:{"mean":float(values[:,j].mean()),"sd":float(values[:,j].std(ddof=1)) if len(values)>1 else None} for j,q in enumerate(names)})
    _atomic(directory/"summary.json",{"scenario":name,"trial":trial,"design":design,"n_measurements":n,
        "n_samples":cfg["n_samples"],"qoi_names":names,"qoi_truth":truth_qois,"qoi_calibration":scores,
        "input_probe_cells":probes.tolist(),"input_probe_calibration":input_scores,
        "field_statistics":global_uncertainty_metrics(final.std),
        "field_convergence":prefix_field_convergence(temperature_store=store,checkpoints=cfg["sample_budgets"],
            reference_mean=final.mean,reference_std=final.std,chunk_rows=cfg["quantile_chunk_rows"]),
        "independent_half_batches":halves,"mean_field_truth_rmse_C":float(np.sqrt(np.mean((final.mean-truth_temperature)**2))),
        "descriptive_fraction_truth_pixels_between_empirical_q05_q95":field_coverage,
        "limits":"correlated pixels are descriptive; ranks aggregate across independent truths at fixed scalar/layout/count; nested-prefix agreement is not proof of MC convergence"})
    sample=np.load(directory/"permeability_sample_0000.npy",mmap_mode="r")
    mean_log=np.load(directory/"posterior_log10_mean.npy",mmap_mode="r")
    _figure(directory/"preview.png",(truth_logs,mean_log,np.log10(sample.astype(float)),truth_temperature,final.mean,final.std),
        ("Hidden truth log10 k","Conditional mean log10 k","First conditional log10 k",
         "Hidden truth LGCNN T [C]","Conditional MC mean T [C]","Conditional temperature SD [C]"),
        shared_pairs=((0,1,2),(3,4)),cell_size_m=model.cell_size_m)
    del sample,mean_log
    del ensemble
    files=[str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file() and p.name not in {"result_files.json","input_files.json"}]
    _commit(directory,files,"result_files.json")


def temperatures(source,cfg,output,manifest,model_factory):
    if cfg["model_config"] is None:
        raise ValueError("a model configuration is required for LGCNN propagation")
    if manifest["status"] not in {"inputs_ready","propagating","propagated","complete"}:
        raise ValueError("native inputs must be prepared before model propagation")
    model=model_factory(_relative_path(cfg["model_config"],source.parent),
        asset_root=_relative_path(cfg["model_asset_root"],source.parent),torch_threads=cfg["torch_threads"])
    if tuple(model.input_shape)!=tuple(cfg["shape"]) or model.cell_size_m!=cfg["cell_size_m"]:
        raise ValueError("benchmark grid must match the frozen model; no resampling")
    if manifest.get("model") is not None and manifest["model"]!=model.metadata:
        raise ValueError("frozen model context changed")
    manifest.update(status="propagating",model=model.metadata);_atomic(output/"manifest.json",manifest)
    for name,law in cfg["scenarios"].items():
        systems=_geometry(cfg,name,law,output);probes=_probes(cfg,name);spec=GSToolsReferenceSpec(**law["covariance"])
        empty=KnownMeanConditioner(cfg["shape"],spec,np.empty((0,2),int),mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
        prior=None
        for trial in range(cfg["n_truths"]):
            root=output/name/f"truth-{trial:04d}"
            _completed(root,"truth_files.json")
            truth=np.load(root/"truth_log10.npy",mmap_mode="r");truth_temperature=_truth_temperature(root,model)
            states=[]
            for design,n,directory in _cases(cfg,name,trial,output):
                if not _completed(directory,"input_files.json"):
                    raise ValueError("native input context missing")
                if _completed(directory,"result_files.json"):
                    continue
                records,store=_prediction_state(directory,cfg,truth_temperature.shape)
                system,features=(empty,None) if n==0 else systems[design]
                # Revealed values are read from the persisted input context only.
                readings=np.asarray(json.loads((directory/"input_summary.json").read_text())["observed_log10_k"])
                states.append((design,n,directory,records,store,system,features,readings))
            for i in range(cfg["n_samples"]):
                pending=[v for v in states if len(v[3])<=i]
                if not pending:
                    continue
                if prior is None:
                    prior=_prior(cfg,spec)
                seed=seed_for(cfg["seed"],_stream(cfg,name),trial,"posterior",i)
                unit=prior.sample_residual(seed)
                for design,n,directory,records,store,system,features,readings in pending:
                    logs=system.conditional_log_field(unit,readings,count=n,features=features,chunk_size=cfg["chunk_size"])
                    field=physical_permeability(logs)
                    exact=float(np.max(np.abs(np.log10(field[tuple(system.indices[:n].T)].astype(float))-readings),initial=0.))
                    if exact>1.e-5:
                        raise ArithmeticError("exact readings not preserved; draw not replaced")
                    start=time.monotonic();temp=np.asarray(model.predict_temperature(field),dtype=np.float32)
                    if temp.shape!=truth_temperature.shape or not np.all(np.isfinite(temp)):
                        raise ValueError("invalid LGCNN prediction; draw not replaced")
                    store[i]=temp;store.flush()
                    if i==0:
                        np.save(directory/"permeability_sample_0000.npy",field)
                    record={"sample_index":i,"seed":seed,"input_sha256":array_hash(field),"temperature_sha256":array_hash(temp),
                        "input_probe_log10":logs[tuple(probes.T)].tolist(),"maximum_exact_log10_error":exact,
                        "seconds":time.monotonic()-start,"normalization":model.normalization_diagnostics(field)}
                    records.append(record);_atomic(directory/"prediction_records.json",records)
                    print(f"LGCNN controlled {name}/truth-{trial:04d}/{design}/n={n}: {i+1}/{cfg['n_samples']} ({record['seconds']:.2f} s)",flush=True)
                    del logs,field,temp
                del unit
            # Release all memory maps before result ledgers and resumed runs (Windows).
            for state in states:
                state[4].flush()
            del states,pending
            if 'store' in locals():
                del store
            for design,n,directory in _cases(cfg,name,trial,output):
                _summarize(cfg,name,trial,design,n,directory,truth_temperature,truth,model,probes)
            del truth
        del systems,prior
    _verify(Path(),model.metadata.get("files_sha256",{}))
    manifest.update(status="propagated",model_evaluations=manifest["planned_model_evaluations"])
    _atomic(output/"manifest.json",manifest)


def report(cfg,output,manifest):
    groups={};density=[]
    for name in cfg["scenarios"]:
        for trial in range(cfg["n_truths"]):
            for design,n,directory in _cases(cfg,name,trial,output):
                if not _completed(directory,"result_files.json"):
                    raise ValueError("complete propagation before reporting")
                summary=json.loads((directory/"summary.json").read_text())
                groups.setdefault((name,design,n),[]).append(summary)
    calibration=[];input_calibration=[]
    for (name,design,n),cases in sorted(groups.items()):
        for budget in cfg["sample_budgets"]:
            for qoi in cases[0]["qoi_names"]:
                calibration.append({"scenario":name,"design":design,"n_measurements":n,"qoi":qoi,
                    **calibration_summary([c["qoi_calibration"][str(budget)][qoi] for c in cases])})
        for probe in cases[0]["input_probe_calibration"]:
            input_calibration.append({"scenario":name,"design":design,"n_measurements":n,"probe":int(probe),
                **calibration_summary([c["input_probe_calibration"][probe] for c in cases])})
        baseline=groups[(name,"prior",0)]
        variance=np.array([c["field_statistics"]["U_RMS"]**2 for c in cases])
        change=np.array([b["field_statistics"]["U_RMS"]**2 for b in baseline])-variance
        density.append({"scenario":name,"design":design,"n_measurements":n,"n_independent_truths":len(cases),
            "mean_spatial_temperature_variance_C2":float(variance.mean()),
            "between_trial_variance_estimate_sem_C2":float(variance.std(ddof=1)/np.sqrt(len(cases))) if len(cases)>1 else None,
            "paired_variance_reduction_from_prior_C2":float(change.mean()),
            "paired_reduction_sem_C2":float(change.std(ddof=1)/np.sqrt(len(change))) if len(change)>1 else None})
    _atomic(output/"summary.json",{"study_id":manifest["study_id"],"temperature_calibration":calibration,
        "input_probe_calibration":input_calibration,"density":density,
        "interpretation":"controlled known-law conditional input UQ; exact data; same frozen LGCNN for truth and draws",
        "limits":"separate scenarios/layouts; outer trial intervals include finite MC error; no geological or PFLOTRAN validation; no claim of tail convergence"})
    _summary_figure(cfg,output,density,calibration)
    text=["# Controlled conditional-field benchmark", "",f"Study: `{manifest['study_id']}`.","",
        f"Completed {manifest['planned_model_evaluations']} model evaluations: independent hidden truths and conditional draws through the same frozen model.","",
        "Known geological parameters are experimental assumptions. These results test model-consistent propagation, not site realism or LGCNN versus simulator accuracy.","",
        "Coverage intervals use independent truths at a fixed QoI/layout/count. Finite ensemble ranks have an explicitly recorded attainable coverage, which may differ from 90%. Randomized ties retain constant/discrete outputs.","",
        "| Scenario | Layout | Readings | Mean spatial temperature variance [C2] | Paired reduction from prior [C2] |",
        "|---|---|---:|---:|---:|"]
    for r in density:
        text.append(f"| {r['scenario']} | {r['design']} | {r['n_measurements']} | {r['mean_spatial_temperature_variance_C2']:.6g} | {r['paired_variance_reduction_from_prior_C2']:.6g} |")
    text += ["","![Controlled benchmark](summary.png)","",
        "See summary.json for per-QoI ranks, CRPS, coverage and exact binomial intervals; each case retains quantiles, risk maps, half-batch diagnostics and nested MC checkpoints.","",
        "The RUN-based realism challenge remains separate: ../measurement_density_validation_20261025/reconstruction_validation.md. Its failed adequacy assessment is not changed by success under a synthetic law."]
    (output/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    report_files=["summary.json","summary.png","report.md"]
    if "structural_profile_file" in cfg:
        from .profile_sensitivity import report as sensitivity_report
        sensitivity_report(cfg,output,manifest)
        report_files += ["profile_sensitivity.json","profile_sensitivity.png"]
    manifest.update(status="complete",report_sha256={p:sha256_file(output/p) for p in report_files})
    _atomic(output/"manifest.json",manifest)


def _summary_figure(cfg,output,density,calibration):
    fig,axes=plt.subplots(len(cfg["scenarios"]),3,figsize=(14,4.4*len(cfg["scenarios"])),squeeze=False,layout="constrained")
    for row,name in enumerate(cfg["scenarios"]):
        baseline=next(r for r in density if r["scenario"]==name and r["n_measurements"]==0)
        for design,_,_ in _designs(cfg,name):
            rows=[baseline,*sorted([r for r in density if r["scenario"]==name and r["design"]==design],key=lambda r:r["n_measurements"])]
            axes[row,0].plot([r["n_measurements"] for r in rows],[np.sqrt(r["mean_spatial_temperature_variance_C2"]) for r in rows],"o-",label=design)
        selected=[r for r in calibration if r["scenario"]==name and r["n_samples"]==cfg["n_samples"]]
        qoi=next((r["qoi"] for r in selected if r["qoi"]=="mean_temperature_anomaly"),selected[0]["qoi"])
        for design,_,_ in _designs(cfg,name):
            rows=sorted([r for r in selected if r["qoi"]==qoi and (r["design"]==design or r["design"]=="prior")],key=lambda r:r["n_measurements"])
            y=np.array([r["rank_coverage"] for r in rows]);intervals=np.array([r["rank_coverage_ci95"] for r in rows])
            axes[row,1].errorbar([r["n_measurements"] for r in rows],y,yerr=np.array([y-intervals[:,0],intervals[:,1]-y]),fmt="o-",label=design)
        target=next(r["attainable_rank_coverage"] for r in selected)
        axes[row,1].axhline(target,color="black",ls=":",label=f"Finite-rank target {target:.3f}")
        rank=next(r for r in selected if r["qoi"]==qoi and r["design"]=="prior")
        axes[row,2].bar(np.arange(len(rank["rank_histogram"])),rank["rank_histogram"])
        axes[row,2].axhline(cfg["n_truths"]/(cfg["n_samples"]+1),color="black",ls=":")
        for j,ax in enumerate(axes[row]):
            ax.grid(alpha=.2)
            if j<2:
                ax.set_xscale("symlog",linthresh=1);ax.set_xlim(left=0)
                ax.set_xticks(cfg["measurement_counts"]);ax.set_xticklabels([str(n) for n in cfg["measurement_counts"]])
                ax.set_xlabel("Exact readings");ax.legend(fontsize=8)
        axes[row,0].set(title=f"{name}: conditional temperature spread",ylabel="RMS temperature SD [C]")
        axes[row,1].set(title=f"{qoi}: rank coverage",ylabel="Coverage; 95% interval across truths",ylim=(0,1.03))
        axes[row,2].set(title=f"{qoi}: prior ranks",xlabel="Randomized finite-ensemble rank",ylabel="Independent hidden truths")
    fig.suptitle(f"Controlled known-law LGCNN benchmark: {cfg['n_truths']} truths, {cfg['n_samples']} draws/case\nInput uncertainty only; model consistency, not geological or simulator validation")
    fig.savefig(output/"summary.png",dpi=150);plt.close(fig)


def run(config_path,output_dir,stage="all",*,model_factory=load_frozen_temperature_model):
    source,cfg=load_config(config_path);output=Path(output_dir).resolve()
    if stage in {"plan","all"}:
        manifest=plan(source,cfg,output)
    else:
        manifest=json.loads((output/"manifest.json").read_text())
        if manifest["configuration"]!=cfg:
            raise ValueError("configuration changed since planning")
        if _sources(source,cfg)!=manifest["source_sha256"]:
            raise ValueError("execution source or package installation changed since planning; use the original context or a fresh run")
        _verify(Path(),manifest["source_sha256"])
        if manifest.get("report_sha256"):
            _verify(output,manifest["report_sha256"])
    if stage in {"validate","all"}:validate(cfg,output,manifest)
    if stage in {"inputs","all"}:prepare(cfg,output,manifest)
    if stage in {"temperatures","all"}:temperatures(source,cfg,output,manifest,model_factory)
    if stage in {"report","all"}:report(cfg,output,manifest)
    _verify(Path(),manifest["source_sha256"])
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True);parser.add_argument("--output-dir",required=True)
    parser.add_argument("--stage",choices=("plan","validate","inputs","temperatures","report","all"),default="all")
    args=parser.parse_args(argv);run(args.config,args.output_dir,args.stage)


if __name__=="__main__":
    main()
