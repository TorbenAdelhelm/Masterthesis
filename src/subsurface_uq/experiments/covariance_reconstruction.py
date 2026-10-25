"""Covariance-only synthetic observation placement and stochastic error budgets."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.linalg import solve_triangular
from scipy.stats import binom, binomtest
import yaml

from ..io import git_head, runtime_versions, sha256_file
from ..sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from ..sampling.covariance_reconstruction import OrdinaryKrigingSystem, covariance_locations
from .reference_conditioning import _identity, _name, _positive_integer, _relative_path, _seed, _write


def quantile_interval(values, probability=.95, confidence=.95):
    """Distribution-free order-statistic interval; null means an infinite end."""
    ordered = np.sort(np.asarray(values, dtype=float))
    if ordered.ndim != 1 or not len(ordered) or not np.all(np.isfinite(ordered)):
        raise ValueError("finite scalar samples required")
    if not 0 < probability < 1 or not 0 < confidence < 1:
        raise ValueError("probability and confidence must lie in (0,1)")
    alpha = (1-confidence)/2
    lo = int(binom.ppf(alpha, len(ordered), probability))
    hi = int(binom.ppf(1-alpha, len(ordered), probability))+1
    return {"estimate": float(np.quantile(ordered, probability, method="inverted_cdf")),
            "lower": None if lo < 1 else float(ordered[lo-1]),
            "upper": None if hi > len(ordered) else float(ordered[hi-1]),
            "confidence": confidence, "method": "pointwise binomial order-statistic interval"}


def stopping_table(cases, error_limits, probability):
    """First tested crossing, without assuming reference errors are monotone."""
    result = []
    for sigma in sorted({case["sigma_log10"] for case in cases}):
        ordered = sorted((c for c in cases if c["sigma_log10"] == sigma), key=lambda c:c["n_measurements"])
        for limit in error_limits:
            row = {"sigma_log10": sigma, "error_limit_log10_rms": limit}
            for metric in ("reference_rmse", "conditional_spread"):
                hit = next((c for c in ordered if c[metric]["estimate"] <= limit), None)
                conservative = next((c for c in ordered if c[metric]["upper"] is not None and c[metric]["upper"] <= limit), None)
                row[metric+"_first_tested_count"] = None if hit is None else hit["n_measurements"]
                row[metric+"_upper_interval_first_tested_count"] = None if conservative is None else conservative["n_measurements"]
                # The analytic Markov bound is conservative but does not use MC quantiles.
                bound = next((c for c in ordered if c[metric+"_markov_bound"] <= limit), None)
                row[metric+"_markov_first_tested_count"] = None if bound is None else bound["n_measurements"]
            row["probability"] = probability
            row["interpretation"] = "first evaluated crossing, not a proven minimum; null = not reached within evaluated counts"
            result.append(row)
    return result


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"references", "covariance", "covariance_by_reference", "native_cell_size_m", "evaluation_stride_cells",
                "candidate_stride_cells", "measurement_counts", "n_designs", "n_samples", "sigma_log10",
                "error_limits_log10_rms", "probability", "seed", "workers", "chunk_size", "pivot_floor"}
    if not isinstance(cfg, dict) or set(cfg) != required:
        raise ValueError(f"configuration requires exactly: {sorted(required)}")
    for key in ("evaluation_stride_cells", "candidate_stride_cells", "n_designs", "n_samples", "workers", "chunk_size"):
        _positive_integer(cfg[key], key)
    _positive_integer(cfg["seed"], "seed", zero=True)
    if cfg["n_samples"] < 2:
        raise ValueError("at least two samples required")
    if not isinstance(cfg["references"], dict) or not cfg["references"]:
        raise ValueError("references must be a nonempty mapping")
    if not isinstance(cfg["covariance_by_reference"], dict) or not set(cfg["covariance_by_reference"]) <= set(cfg["references"]):
        raise ValueError("covariance overrides must name supplied references")
    counts = cfg["measurement_counts"]
    if not isinstance(counts, list) or not counts or counts != sorted(set(counts)):
        raise ValueError("measurement_counts must be strictly increasing")
    for n in counts:
        _positive_integer(n, "measurement count")
    for key in ("sigma_log10", "error_limits_log10_rms"):
        values = cfg[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values) or any(isinstance(v, bool) or not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"{key} must contain distinct positive finite values")
    if not 0 < cfg["probability"] < 1 or not 0 < cfg["pivot_floor"] < 1:
        raise ValueError("probability and pivot_floor must lie in (0,1)")
    if isinstance(cfg["native_cell_size_m"], bool) or not np.isfinite(cfg["native_cell_size_m"]) or cfg["native_cell_size_m"] <= 0:
        raise ValueError("native_cell_size_m must be positive")
    if "sigma_R" in cfg["covariance"] or any("sigma_R" in v for v in cfg["covariance_by_reference"].values()):
        raise ValueError("declare amplitude only in sigma_log10")
    refs = {}
    for name, raw in cfg["references"].items():
        _name(name, "reference")
        source = _relative_path(raw, path.parent)
        ref = np.load(source, mmap_mode="r", allow_pickle=False)
        if np.iscomplexobj(ref) or ref.ndim != 2 or min(ref.shape) < 3 or not np.all(np.isfinite(ref)) or np.any(ref <= 0):
            raise ValueError("references must be finite positive 2-D arrays")
        spec = GSToolsReferenceSpec(**{**cfg["covariance"], **cfg["covariance_by_reference"].get(name, {}), "sigma_R": 1.})
        if spec.center != "median":
            raise ValueError("log-space reconstruction uses median transformation")
        refs[name] = {"path": source, "sha256": sha256_file(source), "native_shape": list(ref.shape), "covariance": spec}
    return path, cfg, refs


def _case(system, features, prior, reference_log, sigmas, probability, error_limits, directory, chunk_size):
    directory.mkdir()
    S, M = len(prior), reference_log.size
    prior_flat = prior.reshape(S, M)
    at_data = prior[:, system.indices[:, 0], system.indices[:, 1]].T
    white_prior = solve_triangular(system.cholesky, at_data, lower=True)
    rmse_squares = {s:np.zeros(S) for s in sigmas}
    maximum = {s:np.zeros(S) for s in sigmas}
    spread_squares = np.zeros(S)
    bias_square = var_sum = 0.
    mean_map = open_memmap(directory/"kriging_log10_mean.npy", mode="w+", dtype=np.float64, shape=system.shape)
    std_map = open_memmap(directory/"unit_posterior_log10_std.npy", mode="w+", dtype=np.float64, shape=system.shape)
    kept = {s:open_memmap(directory/f"fields_sigma-{s:.3f}.npy", mode="w+", dtype=np.float64, shape=(2,*system.shape)) for s in sigmas}
    pin_error = 0.
    observation_flat = np.ravel_multi_index(system.indices.T, system.shape)
    for start in range(0, M, chunk_size):
        stop = min(start+chunk_size, M)
        f = np.asarray(features[start:stop, :len(system.indices)])
        mean, variance = system.moments(f)
        r = 1.-f@system.h
        residual = np.asarray(prior_flat[:, start:stop]).T-f@white_prior-r[:, None]*(system.h@white_prior/system.precision)
        delta = mean-reference_log.ravel()[start:stop]
        bias_square += float(delta@delta)
        var_sum += float(variance.sum())
        spread_squares += np.sum(residual**2, axis=0)
        mean_map.ravel()[start:stop], std_map.ravel()[start:stop] = mean, np.sqrt(variance)
        mask = (observation_flat >= start)&(observation_flat < stop)
        for sigma in sigmas:
            difference = delta[:, None]+sigma*residual
            rmse_squares[sigma] += np.sum(difference**2, axis=0)
            maximum[sigma] = np.maximum(maximum[sigma], np.max(np.abs(difference), axis=0))
            kept[sigma].reshape(2, M)[:, start:stop] = (mean[:, None]+sigma*residual[:, :2]).T
            if mask.any():
                at = mean[observation_flat[mask]-start, None]+sigma*residual[observation_flat[mask]-start]
                pin_error = max(pin_error, float(np.max(np.abs(at-system.readings[mask, None]))))
    if pin_error > 1.e-5:
        raise ArithmeticError("exact readings not reproduced; no replacement or nugget added")
    for arr in (mean_map, std_map, *kept.values()):
        arr.flush()
    for sigma, log_fields in kept.items():
        physical = np.power(10., log_fields)
        if not np.all(np.isfinite(physical)) or np.any(physical <= 0):
            raise ArithmeticError("retained permeability outside finite positive range")
        np.save(directory/f"permeability_sigma-{sigma:.3f}.npy", physical)
    results = []
    for sigma in sigmas:
        rmse = np.sqrt(rmse_squares[sigma]/M)
        spread = sigma*np.sqrt(spread_squares/M)
        np.savez(directory/f"errors_sigma-{sigma:.3f}.npz", reference_log10_rmse=rmse,
                 centered_log10_rms=spread, reference_maximum_log10_error=maximum[sigma])
        probabilities = []
        for limit in error_limits:
            successes = int(np.count_nonzero(rmse <= limit))
            ci = binomtest(successes, S).proportion_ci(confidence_level=.95, method="exact")
            probabilities.append({"error_limit": limit, "probability_estimate": successes/S,
                                  "pointwise_ci95": [float(ci.low), float(ci.high)]})
        results.append({"sigma_log10": sigma, "n_measurements": len(system.indices), "n_samples": S,
            "reference_rmse": quantile_interval(rmse, probability),
            "conditional_spread": quantile_interval(spread, probability),
            "reference_maximum_error": quantile_interval(maximum[sigma], probability),
            "reference_mean_log10_rmse": float(np.sqrt(bias_square/M)),
            "analytic_expected_reference_squared_rmse": (bias_square+sigma**2*var_sum)/M,
            "analytic_average_posterior_variance": sigma**2*var_sum/M,
            "analytic_pair_rms": float(sigma*np.sqrt(2*var_sum/M)),
            "reference_rmse_markov_bound": float(np.sqrt((bias_square+sigma**2*var_sum)/(M*(1-probability)))),
            "conditional_spread_markov_bound": float(sigma*np.sqrt(var_sum/(M*(1-probability)))),
            "prefix_quantiles": {str(n):float(np.quantile(rmse[:n], probability, method="inverted_cdf")) for n in sorted({max(2,S//2),S})},
            "probability_within_limits": probabilities, "maximum_exact_log10_error": pin_error,
            "estimated_constant_mean_from_selected_readings": system.estimated_mean})
    _write(directory/"summary.json", results)
    return results


def _figures(root, name, reference, design, cases, spacing, sigma):
    counts = sorted({c["n_measurements"] for c in cases})
    panels = sorted(set([counts[0], min(counts,key=lambda n:abs(n-16)), min(counts,key=lambda n:abs(n-64)), counts[-1]]))
    fig, axes = plt.subplots(1,len(panels),figsize=(4*len(panels),4), squeeze=False, constrained_layout=True)
    extent = [-spacing/2000,(reference.shape[1]-.5)*spacing/1000,
              (reference.shape[0]-.5)*spacing/1000,-spacing/2000]
    for ax,n in zip(axes[0],panels):
        xy = design.indices[:n]*spacing/1000
        ax.scatter(xy[:,1],xy[:,0],c=np.arange(n),cmap="viridis",s=10)
        ax.set(xlim=extent[:2],ylim=extent[2:],aspect="equal",title=f"{n} covariance-selected sites",xlabel="Column distance [km]",ylabel="Row distance [km]")
    fig.suptitle(f"{name}: locations use covariance only; color = selection order")
    fig.savefig(root/"locations.png",dpi=160); plt.close(fig)
    ref_log = np.log10(reference)
    fig,axes = plt.subplots(4,len(panels),figsize=(4*len(panels),13),squeeze=False,constrained_layout=True)
    fields = [np.load(root/f"n-{n:04d}"/f"fields_sigma-{sigma:.3f}.npy")[0] for n in panels]
    difference_limit = max(float(np.max(np.abs(f-ref_log))) for f in fields)
    vmin,vmax = min(ref_log.min(), *(f.min() for f in fields)),max(ref_log.max(), *(f.max() for f in fields))
    max_sd = max(float(np.load(root/f"n-{n:04d}"/"unit_posterior_log10_std.npy").max()*sigma) for n in panels)
    for col,(n,field) in enumerate(zip(panels,fields)):
        values = [ref_log,field,field-ref_log,np.load(root/f"n-{n:04d}"/"unit_posterior_log10_std.npy")*sigma]
        for row,value in enumerate(values):
            limits = (vmin,vmax) if row<2 else ((-difference_limit,difference_limit) if row==2 else (0,max_sd))
            im=axes[row,col].imshow(value,extent=extent,cmap="coolwarm" if row==2 else "viridis",vmin=limits[0],vmax=limits[1])
            if row<2:
                xy=design.indices[:n]*spacing/1000
                axes[row,col].scatter(xy[:,1],xy[:,0],s=5,facecolors="none",edgecolors="white",linewidths=.5)
            axes[row,col].set_title(f"{['Reference log10 K','Conditional draw 0','Draw − reference','Posterior SD'][row]}; n={n}")
            fig.colorbar(im,ax=axes[row,col],shrink=.7,label="log10 permeability")
    fig.suptitle(f"{name}: unknown mean; sigma={sigma:g}; {spacing:g} m point grid; exact synthetic readings")
    fig.savefig(root/"fields.png",dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    for sigma in sorted({c["sigma_log10"] for c in cases}):
        subset=sorted([c for c in cases if c["sigma_log10"]==sigma],key=lambda c:c["n_measurements"])
        x=[c["n_measurements"] for c in subset]
        for ax,key in zip(axes[:2],("reference_rmse","conditional_spread")):
            ax.plot(x,[c[key]["estimate"] for c in subset],"o-",label=f"sigma={sigma:g}")
            ax.fill_between(x,[c[key]["lower"] or 0 for c in subset],
                [c[key]["upper"] if c[key]["upper"] is not None else np.nan for c in subset],alpha=.1)
    base=sorted([c for c in cases if c["sigma_log10"]==cases[0]["sigma_log10"]],key=lambda c:c["n_measurements"])
    axes[2].plot([c["n_measurements"] for c in base],[c["reference_mean_log10_rmse"] for c in base],"o-",color="black")
    for ax,title in zip(axes,("95th-percentile reference RMSE","95th-percentile spread around kriging mean","Kriging-mean reference RMSE")):
        ax.set(xscale="log",xlabel="Number of exact readings",ylabel="log10 RMS error",title=title);ax.grid(alpha=.25)
    axes[0].legend();axes[1].legend()
    fig.suptitle(f"{name}: shaded bands = pointwise MC quantile intervals; no pooling across designs")
    fig.savefig(root/"error_curves.png",dpi=160);plt.close(fig)


def run(config_path, output):
    source,cfg,refs=load_config(config_path)
    output=Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; choose a fresh directory")
    output.mkdir(parents=True)
    impl_paths=[Path(__file__),Path(__file__).parents[1]/"sampling"/"covariance_reconstruction.py",Path(__file__).parents[1]/"sampling"/"gstools_reference.py"]
    impl={str(p.relative_to(Path(__file__).parents[1])).replace("\\","/"):sha256_file(p) for p in impl_paths}
    manifest={"status":"running","config":cfg,"config_sha256":sha256_file(source),"implementation_sha256":impl,
        "git_head":git_head(Path(__file__).resolve().parents[3]),"runtime":runtime_versions(),"reference_mean_used":False,
        "git_head_context":"base checkout revision; implementation_sha256 identifies executed source including working-tree edits",
        "measurement_error_variance":0.,"zero_measurements":"undefined with an unknown constant mean and flat prior",
        "scope":"full-domain point-sampled reconstruction benchmark; no LGCNN queries or historical borehole inference",
        "stopping":"reference-relative errors use withheld truth for retrospective scoring only; centered spread is covariance-only",
        "scenario_weights":None,"references":{n:{"sha256":r["sha256"],"native_shape":r["native_shape"]} for n,r in refs.items()}}
    manifest["study_id"]=_identity({"config":cfg,"references":manifest["references"],"implementation":impl},"covariance-")
    _write(output/"manifest.json",manifest)
    all_rows=[]
    for name,record in refs.items():
        native=np.load(record["path"],mmap_mode="r",allow_pickle=False)
        reference=np.asarray(native[::cfg["evaluation_stride_cells"],::cfg["evaluation_stride_cells"]],dtype=np.float64)
        spacing=cfg["native_cell_size_m"]*cfg["evaluation_stride_cells"]
        spec=record["covariance"]
        print(f"{name}: point grid {reference.shape}, spacing {spacing} m",flush=True)
        prior=GSToolsReferenceFFTSampler(spec,reference.shape,cell_size_m=spacing,workers=cfg["workers"])
        root=output/name;root.mkdir()
        prior_path=root/"temporary_unit_prior.npy"
        draws=open_memmap(prior_path,mode="w+",dtype=np.float64,shape=(cfg["n_samples"],*reference.shape))
        seeds=[_seed(cfg["seed"],name,"unit-prior",i) for i in range(cfg["n_samples"])]
        for i,seed in enumerate(seeds):
            draws[i]=prior.sample_residual(seed)
        draws.flush()
        prior_metadata={**prior.metadata,"spec":{**prior.metadata["spec"],
            "marginal":"unit zero-mean Gaussian residual; permeability transformation follows ordinary conditioning",
            "assumption_status":"reference-texture covariance shape; total amplitudes declared separately"}}
        _write(root/"prior.json",{"metadata":prior_metadata,"seeds":seeds,"interpretation":"zero-mean UNIT Gaussian residuals only; no reference mean or empirical normalization"})
        for design_index in range(cfg["n_designs"]):
            design=covariance_locations(reference.shape,spec,cell_size_m=spacing,n_measurements=max(cfg["measurement_counts"]),
                candidate_stride_cells=cfg["candidate_stride_cells"],seed=_seed(cfg["seed"],name,"locations",design_index),pivot_floor=cfg["pivot_floor"])
            dr=root/f"design-{design_index:03d}";dr.mkdir()
            _write(dr/"design.json",{**design.metadata,"covariance":asdict(spec),"evaluation_spacing_m":spacing,
                "native_cell_size_m":cfg["native_cell_size_m"],"evaluation_stride_cells":cfg["evaluation_stride_cells"]})
            with (dr/"measurements.csv").open("w",newline="",encoding="utf-8") as handle:
                writer=csv.writer(handle);writer.writerow(["order","row","column","native_row","native_column","log10_K","selection_probability","preselection_unit_variance"])
                for i,(r,c) in enumerate(design.indices):
                    writer.writerow([i+1,r,c,r*cfg["evaluation_stride_cells"],c*cfg["evaluation_stride_cells"],np.log10(reference[r,c]),design.probabilities[i],design.preselection_variance[i]])
            # Only after the complete location sequence is frozen are readings used.
            data=np.log10(reference[tuple(design.indices.T)])
            full=OrdinaryKrigingSystem(reference.shape,spec,design.indices,data,cell_size_m=spacing,cholesky=design.cholesky)
            feature_path=dr/"temporary_features.npy"
            features=open_memmap(feature_path,mode="w+",dtype=np.float64,shape=(reference.size,len(data)))
            for start in range(0,reference.size,cfg["chunk_size"]):
                flat=np.arange(start,min(start+cfg["chunk_size"],reference.size))
                queries=np.column_stack(np.unravel_index(flat,reference.shape))
                features[start:start+len(flat)]=full.features(queries)
            features.flush()
            counts=sorted(set([n for n in cfg["measurement_counts"] if n<=len(data)]+[len(data)]))
            cases=[]
            for n in counts:
                system=OrdinaryKrigingSystem(reference.shape,spec,design.indices[:n],data[:n],cell_size_m=spacing,cholesky=design.cholesky[:n,:n])
                print(f"{name} design {design_index}: {n} readings",flush=True)
                cases.extend(_case(system,features,draws,np.log10(reference),cfg["sigma_log10"],cfg["probability"],
                    cfg["error_limits_log10_rms"],dr/f"n-{n:04d}",cfg["chunk_size"]))
            table=stopping_table(cases,cfg["error_limits_log10_rms"],cfg["probability"])
            _write(dr/"stopping_counts.json",table)
            for row in table:
                all_rows.append({"reference":name,"design":design_index,**row})
            _figures(dr,name,reference,design,cases,spacing,cfg["sigma_log10"][len(cfg["sigma_log10"])//2])
            del features,full,system
            feature_path.unlink()
        del draws,prior
        prior_path.unlink()
    _write(output/"stopping_counts.json",all_rows)
    with (output/"stopping_counts.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    if (sha256_file(source)!=manifest["config_sha256"] or any(sha256_file(r["path"])!=r["sha256"] for r in refs.values())
            or any(sha256_file(p)!=impl[str(p.relative_to(Path(__file__).parents[1])).replace("\\","/")] for p in impl_paths)):
        raise RuntimeError("configuration/reference/implementation changed during execution")
    files={str(p.relative_to(output)):sha256_file(p) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"}
    manifest.update(status="completed",artifact_sha256=files)
    _write(output/"manifest.json",manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    run(args.config,args.output)


if __name__=="__main__":
    main()
