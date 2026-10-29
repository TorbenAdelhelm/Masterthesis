"""Value-free coverage designs and native ordinary-kriging ensembles."""
from __future__ import annotations

import numpy as np
from numpy.lib.format import open_memmap
from scipy.linalg import solve_triangular
from scipy.special import ndtr

from .covariance_reconstruction import OrdinaryKrigingSystem
from .reference_conditioning import _positive_integer


def coverage_locations(shape, count, *, stride, seed, kind, cluster_scale=.1, excluded_mask=None):
    """Freeze a nested geometry-only maximin or clustered observation sequence.

    These are declared coverage scenarios, not inferred historical boreholes.
    Maximin starts uniformly and breaks farthest-distance ties randomly. The
    clustered design uses four uniformly drawn anchors and fixed Gaussian
    location weights, without replacement. Neither policy accepts field values.
    """
    count = _positive_integer(count, "count")
    stride = _positive_integer(stride, "stride")
    seed = _positive_integer(seed, "seed", allow_zero=True)
    if len(shape)!=2 or min(shape)<3 or kind not in {"balanced", "clustered"}:
        raise ValueError("two-dimensional grid and balanced/clustered coverage required")
    if not np.isfinite(cluster_scale) or not 0 < cluster_scale < 1:
        raise ValueError("cluster_scale must lie in (0,1)")
    rr,cc=np.meshgrid(np.arange(0,shape[0],stride),np.arange(0,shape[1],stride),indexing="ij")
    candidates=np.column_stack((rr.ravel(),cc.ravel()))
    if excluded_mask is not None:
        excluded_mask = np.asarray(excluded_mask)
        if excluded_mask.shape != tuple(shape) or excluded_mask.dtype != np.bool_:
            raise ValueError("excluded_mask must be a boolean mask of the native grid")
        candidates = candidates[~excluded_mask[tuple(candidates.T)]]
    if count>len(candidates):
        raise ValueError("not enough candidate cells")
    rng=np.random.default_rng(seed)
    if kind=="clustered":
        anchors=candidates[rng.choice(len(candidates),min(4,len(candidates)),replace=False)]
        distance=(candidates[:,None]-anchors[None])/np.asarray(shape)
        weights=np.exp(-.5*np.sum(distance**2,axis=2)/cluster_scale**2).sum(axis=1)
        weights=np.maximum(weights,np.finfo(float).tiny)
        order=[]
        for _ in range(count):
            chosen=int(rng.choice(len(candidates),p=weights/weights.sum()))
            order.append(chosen);weights[chosen]=0.
    else:
        anchors=np.empty((0,2),dtype=int)
        distance=np.full(len(candidates),np.inf)
        order=[]
        for n in range(count):
            eligible=np.flatnonzero(distance==distance.max())
            chosen=int(rng.choice(eligible))
            order.append(chosen)
            distance=np.minimum(distance,np.sum((candidates-candidates[chosen])**2,axis=1))
            distance[np.asarray(order)]=-1
    return candidates[order],{"kind":kind,"seed":seed,"candidate_stride_cells":stride,
        "candidate_count":len(candidates),"cluster_scale_fraction":cluster_scale if kind=="clustered" else None,
        "anchors":anchors.tolist(),"reference_values_used":False,"reference_mean_used":False,
        "test_regions_excluded": excluded_mask is not None,
        "interpretation":"declared synthetic coverage scenario; not historical location probabilities"}


def gaussian_crps(mean, std, truth):
    """Normal predictive CRPS; a degenerate normal has absolute-error score."""
    mean,std,truth=np.broadcast_arrays(np.asarray(mean,dtype=float),np.asarray(std,dtype=float),np.asarray(truth,dtype=float))
    if not np.all(np.isfinite([mean,std,truth])) or np.any(std<0):
        raise ValueError("finite means/truth and nonnegative standard deviations required")
    delta=truth-mean
    z=np.divide(delta,std,out=np.zeros_like(delta),where=std>0)
    score=std*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    return np.where(std>0,score,np.abs(delta))


def observation_condition_number(spec, indices, cell_size_m):
    offsets=(indices[:,None]-indices[None,:])*cell_size_m
    matrix=spec.correlation(offsets[...,0],offsets[...,1])
    eigenvalues=np.linalg.eigvalsh(matrix)
    if eigenvalues[0]<=eigenvalues[-1]*1.e-12:
        raise ValueError("exact observation covariance loses numerical rank at relative tolerance 1e-12")
    return float(eigenvalues[-1]/eigenvalues[0])


def native_ensembles(reference, spec, indices, counts, prior_draws, directory, *, cell_size_m, chunk_size=2048):
    """Condition all draws/counts in blocks; cache only geometry on disk.

    The reference provides selected readings and separate error scoring. The
    kriging system has no access to unobserved reference values. All native
    fields are retained; no field filtering or covariance reduction is used.
    """
    from pathlib import Path
    from dataclasses import asdict
    from hashlib import sha256
    import json
    from ..io import sha256_file
    from ..experiments.reference_conditioning import _write
    directory=Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reference=np.asarray(reference)
    S=len(prior_draws); M=reference.size
    if tuple(prior_draws.shape[1:])!=reference.shape or counts[-1]!=len(indices):
        raise ValueError("prior grid, counts and maximum observations must agree")
    readings=np.log10(reference[tuple(indices.T)].astype(float))
    # A completed count is the recovery unit. Partial counts are recomputed with
    # the same prior, and completed arrays are verified before they are reused.
    context = sha256(json.dumps({"spec": asdict(spec), "counts": counts,
        "cell_size_m": cell_size_m, "shape": list(reference.shape),
        "reference": sha256(np.ascontiguousarray(reference).view(np.uint8)).hexdigest(),
        "indices": indices.tolist(), "readings": readings.tolist(),
        "prior": sha256(np.ascontiguousarray(prior_draws).view(np.uint8)).hexdigest()},
        sort_keys=True).encode()).hexdigest()
    records = []
    remaining = []
    for n in counts:
        target = directory/f"n-{n:04d}"
        ledger = target/"input_files.json"
        marker = target/"input_context.json"
        if ledger.exists():
            if not marker.exists() or json.loads(marker.read_text())["context"] != context:
                raise ValueError("native input recovery context changed")
            for filename, digest in json.loads(ledger.read_text()).items():
                if sha256_file(target/filename) != digest:
                    raise ValueError("completed native input cache changed")
            records.append(json.loads((target/"input_summary.json").read_text()))
        else:
            owned_validation = {"withheld_predictions.npz", "withheld_diagnostics.json"}
            has_inputs = target.exists() and any(p.name not in owned_validation for p in target.iterdir())
            if has_inputs and (
                not marker.exists() or json.loads(marker.read_text())["context"] != context
            ):
                raise ValueError("partial native input recovery context changed")
            remaining.append(n)
    if not remaining:
        return records
    full=OrdinaryKrigingSystem(reference.shape,spec,indices,readings,cell_size_m=cell_size_m)
    feature_path=directory/"temporary_geometry.npy"
    features=open_memmap(feature_path,mode="w+",dtype=np.float64,shape=(M,len(indices)))
    for start in range(0,M,chunk_size):
        flat=np.arange(start,min(start+chunk_size,M))
        queries=np.column_stack(np.unravel_index(flat,reference.shape))
        features[start:start+len(flat)]=full.features(queries)
    features.flush()
    flat_prior=prior_draws.reshape(S,M)
    for n in remaining:
        print(f"native conditioning: {n} readings, {S} draws",flush=True)
        target=directory/f"n-{n:04d}";target.mkdir(exist_ok=True)
        _write(target/"input_context.json", {"context": context})
        system=OrdinaryKrigingSystem(reference.shape,spec,indices[:n],readings[:n],cell_size_m=cell_size_m,cholesky=full.cholesky[:n,:n])
        at_data=prior_draws[:,indices[:n,0],indices[:n,1]].T
        white_prior=solve_triangular(system.cholesky,at_data,lower=True)
        fields=open_memmap(target/"permeability_ensemble.npy",mode="w+",dtype=np.float32,shape=(S,*reference.shape))
        means=open_memmap(target/"posterior_log10_mean.npy",mode="w+",dtype=np.float32,shape=reference.shape)
        stds=open_memmap(target/"posterior_log10_std.npy",mode="w+",dtype=np.float32,shape=reference.shape)
        error_squares=np.zeros(S);spread_squares=np.zeros(S)
        maximum_exact=0.; bias_squares=variance_sum=0.
        for start in range(0,M,chunk_size):
            stop=min(start+chunk_size,M)
            f=np.asarray(features[start:stop,:n])
            mean,unit_variance=system.moments(f)
            r=1.-f@system.h
            residual=np.asarray(flat_prior[:,start:stop]).T-f@white_prior-r[:,None]*(system.h@white_prior/system.precision)
            logs=mean[:,None]+spec.sigma_R*residual
            with np.errstate(over="raise",invalid="raise"):
                physical=np.power(10.,logs).astype(np.float32)
            if not np.all(np.isfinite(physical)) or np.any(physical<=0):
                raise ArithmeticError("native field outside finite positive float32 support; no replacement")
            fields.reshape(S,M)[:,start:stop]=physical.T
            means.ravel()[start:stop]=mean
            stds.ravel()[start:stop]=spec.sigma_R*np.sqrt(unit_variance)
            truth=np.log10(reference.ravel()[start:stop].astype(float))
            delta=mean-truth
            error_squares+=np.sum((logs-truth[:,None])**2,axis=0)
            spread_squares+=np.sum((spec.sigma_R*residual)**2,axis=0)
            bias_squares+=float(delta@delta)
            variance_sum+=float(unit_variance.sum())*spec.sigma_R**2
        for arr in (fields,means,stds):arr.flush()
        values=fields[:,indices[:n,0],indices[:n,1]]
        discrepancy=np.log10(values.astype(float))-readings[:n]
        maximum_exact=float(np.max(np.abs(discrepancy)))
        if maximum_exact>1.e-5:
            raise ArithmeticError("exact observations not preserved; no nugget or replacement")
        record={"n_measurements":n,"n_samples":S,"native_shape":list(reference.shape),
            "maximum_exact_log10_error":maximum_exact,"estimated_mean_selected_readings_only":system.estimated_mean,
            "reference_mean_used":False,"analytic_average_input_variance_log10":variance_sum/M,
            "analytic_expected_reference_squared_rmse_log10":(bias_squares+variance_sum)/M,
            "kriging_mean_reference_rmse_log10":float(np.sqrt(bias_squares/M)),
            "reference_rmse_log10":np.sqrt(error_squares/M).tolist(),
            "centered_rms_log10":np.sqrt(spread_squares/M).tolist(),
            "conditional_covariance":"ordinary kriging including unknown-mean uncertainty; full matched covariance",
            "measurement_error_variance":0.,"sample_filtering":None}
        _write(target/"input_summary.json",record);records.append(record)
        del fields,means,stds,system
        filenames = ["permeability_ensemble.npy", "posterior_log10_mean.npy",
                     "posterior_log10_std.npy", "input_summary.json", "input_context.json"]
        ledger = {p: sha256_file(target/p) for p in filenames}
        temporary = target/"input_files.json.partial"
        _write(temporary, ledger)
        temporary.replace(target/"input_files.json")
    del f,features,full
    feature_path.unlink()
    return sorted(records, key=lambda r: r["n_measurements"])
