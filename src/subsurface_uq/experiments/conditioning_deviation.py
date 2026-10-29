"""Native-grid reference error and within-setting spread versus measurement count.

Reuse full FFT prior draws across nested budgets. Blocked Matheron conditioning
is the same fixed-design law as ConditionalReferenceFFTSampler; no KL reduction,
grid coarsening, empirical field normalization or sample selection is applied.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler
from ..sampling.reference_conditioning import ConditionalReferenceFFTSampler, _covariance, draw_reference_measurements
from .reference_conditioning import _identity, _relative_path, _seed, _write, _write_observations, load_config as load_base


def cached_prior(prior, seeds, destination):
    """Float64 disk cache; use exactly the scalar sampler's prior seed stream."""
    store = open_memmap(destination,mode="w+",dtype=np.float64,shape=(len(seeds),*prior.shape))
    for index,seed in enumerate(seeds):
        stream = np.random.SeedSequence(seed).spawn(2)[0]
        store[index] = prior.sample_residual(int(stream.generate_state(1,dtype=np.uint64)[0]))
    store.flush()
    return store


def conditional_blocks(sampler, prior_draws, seeds):
    """Yield (flat start, log draws [cells,draws], exact mean, exact variance).

    Whiten each cross-covariance block once for every draw. This reorders linear
    algebra only. Independent simulated likelihood noise is retained for noisy
    data, with the scalar sampler's original stream and prefix coupling.
    """
    if prior_draws.shape != (len(seeds),*sampler.field_shape):
        raise ValueError("prior draws, seeds and sampler grid must agree")
    transform = None
    if sampler.observations is not None:
        sampled_at_data = prior_draws[:,sampler.indices[:,0],sampler.indices[:,1]].T
        noise = np.column_stack([np.random.default_rng(np.random.SeedSequence(s).spawn(2)[1]).standard_normal(len(sampler.indices))
                                 for s in seeds])*sampler.observations.observation_std_log10_k[:,None]
        transform = sampler._solver.transform(sampler._innovation[:,None]-sampled_at_data-noise)
    flat = prior_draws.reshape(len(seeds),-1)
    for start,queries in sampler._blocks():
        stop = start+len(queries)
        center = sampler._mean.ravel()[start:stop]
        draws = center[:,None]+np.asarray(flat[:,start:stop]).T
        mean = center.copy()
        variance = np.full(len(queries),sampler.covariance.sigma_R**2)
        if transform is not None:
            cross = _covariance(sampler.covariance,queries,sampler.indices,sampler.cell_size_m)
            whitened = sampler._solver.whiten(cross)
            draws += whitened@transform
            mean += whitened@sampler._whitened_innovation
            variance -= np.sum(whitened**2,axis=1)
        if np.min(variance,initial=0)<-sampler.covariance.sigma_R**2*1e-8:
            raise ArithmeticError("negative posterior variance beyond solver tolerance")
        yield start,draws,mean,np.maximum(variance,0.)


def pairwise_rms(gram, pixel_count):
    diagonal = np.diag(gram)
    distances = (diagonal[:,None]+diagonal[None,:]-2*gram)/pixel_count
    if distances.min() < -1e-10*max(1.,float(diagonal.max()/pixel_count)):
        raise ArithmeticError("pair-distance cancellation exceeds numerical tolerance")
    np.fill_diagonal(distances,0.)
    return np.sqrt(np.maximum(distances,0.))


def measure_case(sampler, prior_draws, seeds, directory, retained, checkpoints):
    """All metrics use every native cell and every draw; storage uses fixed indices."""
    directory = Path(directory)
    directory.mkdir(parents=True)
    shape,L = sampler.field_shape,len(seeds)
    M = int(np.prod(shape))
    gram = np.zeros((L,L))
    relative_squares,absolute_squares = np.zeros(L),np.zeros(L)
    analytic_reference,analytic_relative,analytic_pair = 0.,0.,0.
    maps = {name:open_memmap(directory/f"{name}.npy",mode="w+",dtype=np.float32,shape=shape)
            for name in ("analytic_log10_std","ensemble_log10_std","ensemble_log10_bias")}
    fields = {i:open_memmap(directory/f"field-{i:04d}.npy",mode="w+",dtype=np.float32,shape=shape) for i in retained}
    differences = {i:open_memmap(directory/f"log10_difference-{i:04d}.npy",mode="w+",dtype=np.float32,shape=shape) for i in retained}
    ref = sampler.reference.ravel()
    logs_ref = np.log10(ref)
    exact_error = 0.
    observation_flat = np.ravel_multi_index(sampler.indices.T,shape) if len(sampler.indices) else np.empty(0,dtype=int)
    for start,draws,mean,variance in conditional_blocks(sampler,prior_draws,seeds):
        stop = start+len(mean)
        residuals = draws-logs_ref[start:stop,None]
        if not np.all(np.isfinite(residuals)):
            raise ArithmeticError("nonfinite log-field; no replacement")
        gram += residuals.T@residuals
        with np.errstate(over="raise",invalid="raise",under="ignore"):
            relative = np.expm1(np.log(10.)*residuals)
            physical = np.power(10.,draws).astype(np.float32)
        if not np.all(np.isfinite(physical)) or np.any(physical<=0):
            raise ArithmeticError("draw exceeds positive finite float32 range; no replacement")
        relative_squares += np.sum(relative**2,axis=0)
        absolute_squares += np.sum((relative*ref[start:stop,None])**2,axis=0)
        delta = mean-logs_ref[start:stop]
        analytic_reference += np.sum(variance+delta**2)
        analytic_pair += 2*np.sum(variance)
        a,b = np.log(10.)*delta,np.log(10.)**2*variance
        analytic_relative += np.sum(np.exp(2*a+b)*np.expm1(b)+np.expm1(a+b/2)**2)
        maps["analytic_log10_std"].ravel()[start:stop] = np.sqrt(variance)
        maps["ensemble_log10_std"].ravel()[start:stop] = np.std(residuals,axis=1,ddof=1)
        maps["ensemble_log10_bias"].ravel()[start:stop] = residuals.mean(axis=1)
        for i in retained:
            fields[i].ravel()[start:stop] = physical[:,i]
            differences[i].ravel()[start:stop] = residuals[:,i]
        mask = (observation_flat>=start)&(observation_flat<stop)
        if np.any(mask):
            exact = mask & (sampler.observations.observation_std_log10_k==0)
            if np.any(exact):
                discrepancy = draws[observation_flat[exact]-start]-sampler.observations.observed_log10_k[exact,None]
                exact_error = max(exact_error,float(np.abs(discrepancy).max()))
    if exact_error > max(sampler.covariance.sigma_R*1e-7,1e-10):
        raise ArithmeticError("exact-data residual exceeds scalar sampler tolerance; no replacement")
    for values in (*maps.values(),*fields.values(),*differences.values()):
        values.flush()
    del maps,fields,differences,values
    pair = pairwise_rms(gram,M)
    np.save(directory/"pairwise_log10_rms.npy",pair)
    np.savez(directory/"sample_metrics.npz",log10_rmse=np.sqrt(np.diag(gram)/M),
             relative_rmse_percent=100*np.sqrt(relative_squares/M),absolute_rmse_m2=np.sqrt(absolute_squares/M),seeds=seeds)
    summaries = []
    for n in checkpoints:
        pairs = pair[:n,:n][np.triu_indices(n,1)]
        summaries.append({"n_samples":n,"reference_log10_rms":float(np.sqrt(np.diag(gram)[:n].mean()/M)),
            "between_sample_log10_rms":float(np.sqrt(np.mean(pairs**2))),
            "relative_reference_rms_percent":float(100*np.sqrt(relative_squares[:n].mean()/M)),
            "single_field_log10_rmse_q05_q50_q95":np.quantile(np.sqrt(np.diag(gram)[:n]/M),[.05,.5,.95]).tolist(),
            "pair_log10_rms_q05_q50_q95":np.quantile(pairs,[.05,.5,.95]).tolist()})
    result = {"checkpoints":summaries,"analytic_reference_log10_rms":float(np.sqrt(analytic_reference/M)),
        "analytic_between_sample_log10_rms":float(np.sqrt(analytic_pair/M)),
        "analytic_relative_reference_rms_percent":float(100*np.sqrt(analytic_relative/M)),
        "maximum_exact_log10_error":exact_error,"pixel_count":M,"retained_sample_indices":retained,
        "pair_count":L*(L-1)//2,"pair_interpretation":"all unordered pairs within one fixed design; pairs are dependent, not independent replicates",
        "sample_filtering":None,"sampler":sampler.metadata}
    _write(directory/"summary.json",result)
    return result


def load_config(path):
    source = Path(path).resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or set(cfg)!={"conditioning_config","measurement_counts","sample_budgets"}:
        raise ValueError("declare exactly conditioning_config, measurement_counts and sample_budgets")
    for key,minimum in (("measurement_counts",0),("sample_budgets",2)):
        values = cfg[key]
        if (not isinstance(values,list) or not values or values!=sorted(set(values)) or
                any(isinstance(v,bool) or not isinstance(v,int) or v<minimum for v in values)):
            raise ValueError(f"{key} must be strictly increasing integers >= {minimum}")
    base_source,base,policies,references = load_base(_relative_path(cfg["conditioning_config"],source.parent))
    if cfg["measurement_counts"][0]!=0 or len(cfg["measurement_counts"])<2 or cfg["measurement_counts"][-1]!=base["n_measurements"]:
        raise ValueError("measurement counts must start at zero and end at base n_measurements")
    if cfg["sample_budgets"][-1]!=base["n_samples"]:
        raise ValueError("largest sample budget must equal base n_samples")
    if not {0,1}<=set(base["retained_sample_indices"]):
        raise ValueError("retain samples 0 and 1 to visualize within-setting differences")
    return source,cfg,base_source,base,policies,references


def _plots(output,name,reference,policy,counts,designs,cell_size,sigma):
    # First design's paired examples use common, fixed color scales across budgets.
    ordered = list(reversed(counts))
    examples = [output/name/policy/"design-000"/f"n-{n:04d}" for n in ordered]
    fig,axes = plt.subplots(4,len(ordered),figsize=(4*len(ordered),14),layout="constrained",squeeze=False)
    data = []
    for case in examples:
        a,b = (np.load(case/f"log10_difference-{i:04d}.npy",mmap_mode="r") for i in (0,1))
        data.append((a,b,a-b,np.load(case/"analytic_log10_std.npy",mmap_mode="r")))
    limit = max(float(np.max(np.abs(v))) for row in data for v in row[:3])
    extent = (0,reference.shape[1]*cell_size/1000,0,reference.shape[0]*cell_size/1000)
    stride = max(1,int(np.ceil(max(reference.shape)/600)))
    for c,(n,values) in enumerate(zip(ordered,data)):
        for r,value in enumerate(values):
            artist = axes[r,c].imshow(value[::stride,::stride],origin="lower",extent=extent,
                cmap="magma" if r==3 else "RdBu_r",vmin=0 if r==3 else -limit,vmax=sigma if r==3 else limit)
            axes[r,c].set_title(f"n={n}: "+("draw A − reference","draw B − reference","draw A − draw B","analytic log10 SD")[r],fontsize=10)
            axes[r,c].set(xlabel="Column distance [km]",ylabel="Row distance [km]")
            fig.colorbar(artist,ax=axes[r,c],shrink=.75)
    fig.suptitle(f"{name} / {policy}: fewer measurements →; common draws and color scales; all differences in log10 K")
    fig.savefig(output/f"{name}.{policy}.deviations.png",dpi=140)
    plt.close(fig)
    fig,axes = plt.subplots(2,len(ordered)+1,figsize=(4*(len(ordered)+1),7),layout="constrained",squeeze=False)
    fields = [[np.log10(reference),*[np.log10(np.load(case/f"field-{i:04d}.npy",mmap_mode="r")) for case in examples]] for i in (0,1)]
    lower,upper = min(float(v.min()) for row in fields for v in row),max(float(v.max()) for row in fields for v in row)
    for i,row in enumerate(fields):
        for j,value in enumerate(row):
            artist = axes[i,j].imshow(value[::stride,::stride],origin="lower",extent=extent,cmap="viridis",vmin=lower,vmax=upper)
            axes[i,j].set_title("Reference" if j==0 else f"n={ordered[j-1]}, draw {'AB'[i]}")
            fig.colorbar(artist,ax=axes[i,j],shrink=.7,label="log10(K / m²)")
    fig.savefig(output/f"{name}.{policy}.fields.png",dpi=140)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for ax,empirical,analytic,title in zip(axes,
            ("reference_log10_rms","between_sample_log10_rms","relative_reference_rms_percent"),
            ("analytic_reference_log10_rms","analytic_between_sample_log10_rms","analytic_relative_reference_rms_percent"),
            ("Deviation from reference [log10]","Spread between samples [log10]","Relative reference error [%]")):
        for budget in sorted({r["n_samples"] for d in designs for r in d["checkpoints"]}):
            values = [np.mean([next(r[empirical]**2 for r in d["checkpoints"] if r["n_samples"]==budget)
                              for d in designs if d["n_measurements"]==n])**.5 for n in ordered]
            ax.plot(range(len(ordered)),values,"o-",label=f"MC, L={budget}")
        values = [np.mean([d[analytic]**2 for d in designs if d["n_measurements"]==n])**.5 for n in ordered]
        ax.plot(range(len(ordered)),values,"k--",label="Exact GP expectation")
        ax.set(xticks=range(len(ordered)),xticklabels=ordered,xlabel="Measurements retained (fewer →)",title=title)
        ax.legend(fontsize=8)
    fig.suptitle(f"{name} / {policy}; RMS over all native cells, draws and separately conditioned designs")
    fig.savefig(output/f"{name}.{policy}.curves.png",dpi=150)
    plt.close(fig)


def run_study(config_path,output_dir):
    source,cfg,base_source,base,policies,references = load_config(config_path)
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("use a fresh output directory")
    sources = [source,base_source,*Path(__file__).parents[1].rglob("*.py"),*[v["path"] for v in references.values()],
               *[v["weights_path"] for v in references.values() if v["weights_path"] is not None]]
    hashes = {str(p):sha256_file(p) for p in sources}
    output.mkdir(parents=True)
    manifest = {"status":"running","config":cfg,"conditioning_config":base,"source_sha256":hashes,
        "code_commit":git_head(Path(__file__).parents[3]),"runtime":runtime_versions(("numpy","scipy","gstools","matplotlib")),
        "grid":"original native reference grid; no coarsening or KL truncation","sample_filtering":None,
        "reference_pooling":None,"policy_pooling":None,"temperature_model_evaluated":False,
        "method":"blocked Matheron conditioning of cached float64 full FFT prior draws; same scalar sampler seed/noise map",
        "information_state":"reference_median already knows complete reference structure; not sparse reconstruction"}
    _write(output/"manifest.json",manifest)
    cases = []
    for name,item in sorted(references.items()):
        reference,spec = item["reference"],item["covariance"]
        law_id = _identity({"reference_sha256":item["sha256"],"covariance":asdict(spec),"mean_mode":base["mean_mode"],"cell_size_m":base["cell_size_m"]},"reference-")
        prior = GSToolsReferenceFFTSampler(spec,reference.shape,cell_size_m=base["cell_size_m"],workers=base["workers"])
        for policy_name,policy in sorted(policies.items()):
            group = []
            for design in range(base["n_designs"]):
                directory = output/name/policy_name/f"design-{design:03d}"
                directory.mkdir(parents=True)
                maximum = draw_reference_measurements(reference,covariance=spec,n_measurements=base["n_measurements"],
                    location_seed=_seed(base["seed"],law_id,policy_name,design,"location"),
                    observation_seed=_seed(base["seed"],law_id,policy_name,design,"reading"),cell_size_m=base["cell_size_m"],
                    location_prior=policy,location_weights=item["weights"],measurement_mode=base["measurement_mode"],observation_std_log10_k=base["observation_std_log10_k"])
                _write_observations(directory/"maximum_measurements.csv",maximum,base["cell_size_m"])
                preflight = ConditionalReferenceFFTSampler(reference,spec,maximum,cell_size_m=base["cell_size_m"],
                    mean_mode=base["mean_mode"],chunk_size=base["chunk_size"],workers=base["workers"],prior_sampler=prior)
                del preflight
                seeds = [_seed(base["seed"],law_id,policy_name,design,"field",i) for i in range(base["n_samples"])]
                cache_path = directory/"temporary_prior.npy"
                print(f"{name}/{policy_name}/design-{design}: generating {len(seeds)} native FFT priors",flush=True)
                cache = cached_prior(prior,seeds,cache_path)
                for count in cfg["measurement_counts"]:
                    case_dir = directory/f"n-{count:04d}"
                    observations = maximum.prefix(count) if count else None
                    sampler = ConditionalReferenceFFTSampler(reference,spec,observations,cell_size_m=base["cell_size_m"],mean_mode=base["mean_mode"],
                        chunk_size=base["chunk_size"],workers=base["workers"],prior_sampler=prior)
                    result = measure_case(sampler,cache,seeds,case_dir,base["retained_sample_indices"],cfg["sample_budgets"])
                    if observations is not None:
                        _write_observations(case_dir/"measurements.csv",observations,base["cell_size_m"])
                    result.update(reference=name,policy=policy_name,design=design,n_measurements=count,directory=str(case_dir.relative_to(output)))
                    _write(case_dir/"summary.json",result)
                    cases.append(result)
                    group.append(result)
                    print(f"{name}/{policy_name}/design-{design}/n={count}: {len(seeds)} fields measured",flush=True)
                    del sampler
                del cache
                cache_path.unlink()
            _plots(output,name,reference,policy_name,cfg["measurement_counts"],group,base["cell_size_m"],spec.sigma_R)
        del prior
    _write(output/"summary.json",{"cases":cases,"reference_pooling":None,"policy_pooling":None,
        "interpretation":"native-grid conditional deviation metrics; no pass/fail cutoff or inferred scenario probabilities"})
    for path,digest in hashes.items():
        if sha256_file(path)!=digest:
            raise ValueError("source context changed during execution")
    manifest.update(status="complete",case_count=len(cases),generated_fields=len(cases)*base["n_samples"],
                    independent_prior_draws=len(references)*len(policies)*base["n_designs"]*base["n_samples"])
    manifest["artifacts_sha256"] = {str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p!=output/"manifest.json"}
    _write(output/"manifest.json",manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output-dir",required=True)
    args = parser.parse_args(argv)
    run_study(args.config,args.output_dir)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
