"""Retained controlled-field ensembles with one spatial conditioning pass.

The scratch prior is restartable per draw. Conditioning restarts per outer
trial; predictions restart per individual model call. No grid-by-data cache.
"""
from __future__ import annotations

import json
import numpy as np
from numpy.lib.format import open_memmap

from ..sampling.conditional_density import condition_prefixes
from ..sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from ..sampling.gstools_reference import GSToolsReferenceSpec
from . import controlled_field_benchmark as b


def preflight(cfg):
    """Test the actual native FFT embedding and every observation prefix."""
    checks={}
    for name,law in cfg["scenarios"].items():
        spec=GSToolsReferenceSpec(**law["covariance"]);sampler=b._prior(cfg,spec)
        unit=sampler.sample_residual(b.seed_for(cfg["seed"],"fft-preflight",name))
        rows=[]
        for design,kind,index in b._designs(cfg,name):
            indices,_=b._locations(cfg,name,kind,index)
            system=KnownMeanConditioner(cfg["shape"],spec,indices,
                mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
            f=system.features(indices)
            # These are arbitrary deterministic revealed values, independent of
            # production truths; assess interpolation algebra at ALL data cells.
            data=law["mean_log10_k"]+.1*np.sin(np.arange(len(indices)))
            for n in cfg["measurement_counts"][1:]:
                coeff=system.update_coefficients(data[:n],unit[tuple(indices[:n].T)],count=n)
                predicted=law["mean_log10_k"]+spec.sigma_R*unit[tuple(indices[:n].T)]+f[:n,:n]@coeff
                error=float(np.max(np.abs(predicted-data[:n])))
                _,var=system.moments(f,data[:n],count=n)
                if error>1.e-5:
                    raise ArithmeticError("native exact-conditioning preflight failed")
                offsets=(indices[:n,None]-indices[None,:n])*cfg["cell_size_m"]
                eig=np.linalg.eigvalsh(spec.correlation(offsets[...,0],offsets[...,1]))
                rows.append({"design":design,"n_measurements":n,"condition_number":float(eig[-1]/eig[0]),
                    "maximum_exact_log10_error":error,"minimum_probe_variance_log10":float(var.min())})
        checks[name]={"sampler":sampler.metadata,"conditioning":rows}
        print(f"Native FFT/conditioning preflight passed: {name}",flush=True)
        del sampler,unit
    return checks


def _priors(cfg,name,trial,root,sampler):
    path=root/"prior_scratch.npy";marker=root/"prior_records.json"
    records=json.loads(marker.read_text()) if marker.exists() else []
    if records and not path.exists():
        raise ValueError("incomplete inputs lost their recorded prior scratch")
    values=open_memmap(path,mode="r+" if path.exists() else "w+",dtype=np.float64,
                       shape=(cfg["n_samples"],*cfg["shape"]))
    if values.shape!=(cfg["n_samples"],*cfg["shape"]) or values.dtype!=np.float64 or len(records)>len(values):
        raise ValueError("prior scratch shape/dtype/records changed")
    for i,record in enumerate(records):
        seed=b.seed_for(cfg["seed"],b._stream(cfg,name),trial,"posterior",i)
        if record!={"sample_index":i,"seed":seed,"sha256":b.array_hash(values[i])}:
            raise ValueError("recorded prior draw changed")
    for i in range(len(records),len(values)):
        seed=b.seed_for(cfg["seed"],b._stream(cfg,name),trial,"posterior",i)
        values[i]=sampler.sample_residual(seed);values.flush()
        records.append({"sample_index":i,"seed":seed,"sha256":b.array_hash(values[i])})
        b._atomic(marker,records)
        print(f"FFT priors {name}/truth-{trial:04d}: {i+1}/{len(values)}",flush=True)
    return values,records


def prepare(cfg,output,manifest):
    manifest["status"]="preparing_inputs";b._atomic(output/"manifest.json",manifest)
    for name,law in cfg["scenarios"].items():
        spec=GSToolsReferenceSpec(**law["covariance"]);sampler=None
        for trial in range(cfg["n_truths"]):
            root=output/name/f"truth-{trial:04d}";root.mkdir(parents=True,exist_ok=True)
            if b._completed(root,"ensemble_files.json"):
                # A crash immediately after committing may leave disposable scratch.
                (root/"prior_scratch.npy").unlink(missing_ok=True)
                continue
            if sampler is None:
                sampler=b._prior(cfg,spec)
                b._atomic(output/name/"prior.json",{**sampler.metadata,"law":law,
                    "field_role":"unit Gaussian driving field for controlled lognormal prior"})
            if not b._completed(root,"truth_files.json"):
                seed=b.seed_for(cfg["seed"],b._stream(cfg,name),trial,"hidden-truth")
                logs=law["mean_log10_k"]+spec.sigma_R*sampler.sample_residual(seed)
                np.save(root/"truth_log10.npy",logs)
                np.save(root/"truth_permeability.npy",physical_permeability(logs));del logs
                b._atomic(root/"truth.json",{"seed":seed,"trial":trial,"scenario":name,
                    "role":"independent hidden truth; only selected cells are revealed"})
                b._commit(root,["truth_log10.npy","truth_permeability.npy","truth.json"],"truth_files.json")
            truth=np.load(root/"truth_log10.npy",mmap_mode="r")
            prior,prior_records=_priors(cfg,name,trial,root,sampler)
            files=["prior_records.json"];first=True
            for design,kind,index in b._designs(cfg,name):
                indices,meta=b._locations(cfg,name,kind,index)
                system=KnownMeanConditioner(cfg["shape"],spec,indices,
                    mean_log10=law["mean_log10_k"],cell_size_m=cfg["cell_size_m"])
                geometry=output/name/"geometry"/design;geometry.mkdir(parents=True,exist_ok=True)
                b._atomic(geometry/"design.json",{"indices":indices.tolist(),"metadata":meta,
                    "condition_number":system.condition_number,"known_mean_log10":law["mean_log10_k"]})
                counts=cfg["measurement_counts"] if first else cfg["measurement_counts"][1:]
                directories={n:root/"prior" if n==0 else root/design/f"n-{n:04d}" for n in counts}
                readings=truth[tuple(indices.T)]
                last_percent=-1
                def progress(stop,size):
                    nonlocal last_percent
                    percent=int(100*stop/size)
                    if percent//10!=last_percent//10:
                        print(f"Conditioning {name}/truth-{trial:04d}/{design}: {percent}%",flush=True)
                        last_percent=percent
                averages=condition_prefixes(system,readings,prior,counts,directories,
                    chunk_size=cfg["chunk_size"],progress=progress)
                for n,directory in directories.items():
                    # Input ledgers are committed only after the whole trial is
                    # finished. An interrupted spatial pass is overwritten.
                    old=directory/"input_files.json"
                    if old.exists():
                        old.unlink()
                    b._prepare_case(cfg,name,trial,"prior" if n==0 else design,n,
                        directory,truth,system,None,average_variance=averages[n])
                    fields=np.load(directory/"permeability_samples.npy",mmap_mode="r")
                    rows=[]
                    for i,field in enumerate(fields):
                        error=float(np.max(np.abs(np.log10(field[tuple(indices[:n].T)].astype(float))-readings[:n]),initial=0.))
                        if error>1.e-5:
                            raise ArithmeticError("exact readings not preserved; draw not replaced")
                        rows.append({"sample_index":i,"seed":prior_records[i]["seed"],
                            "input_sha256":b.array_hash(field),"maximum_exact_log10_error":error})
                    del field,fields
                    b._atomic(directory/"input_records.json",rows)
                    files.extend(str((directory/f).relative_to(root)) for f in
                        ["permeability_samples.npy","input_records.json","input_files.json"])
                first=False
            del prior,truth
            b._commit(root,files,"ensemble_files.json")
            (root/"prior_scratch.npy").unlink()
        del sampler
    manifest["status"]="inputs_ready";b._atomic(output/"manifest.json",manifest)
