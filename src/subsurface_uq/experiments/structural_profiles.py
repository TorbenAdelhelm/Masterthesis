"""Derive and check synthetic-map probability laws from supplied native maps."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ..io import sha256_file,git_head,runtime_versions
from ..sampling.gstools_reference import GSToolsReferenceSpec,GSToolsReferenceFFTSampler
from ..sampling.controlled_fields import physical_permeability
from ..sampling.structural_profiles import (fit_profile,structural_features,variogram_curves,
    fitted_curves,DIRECTIONS)
from .reference_conditioning import _relative_path,_name,_write,_identity,_positive_integer
from .controlled_field_benchmark import seed_for,array_hash


_KEYS={"references","preparation_manifest","cell_size_m","permeability_units","lags_cells",
       "length_bounds_m","quadrature_order","structural_samples","diagnostic_stride_cells",
       "paired_embedding_shape","seed","workers"}


def load_config(path):
    source=Path(path).resolve();cfg=yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or set(cfg)!=_KEYS:
        raise ValueError("declare exactly the structural-profile settings")
    if cfg["cell_size_m"]!=5. or cfg["permeability_units"]!="m2":
        raise ValueError("native 5 m intrinsic-permeability maps in m2 required")
    for name in ["quadrature_order","structural_samples","diagnostic_stride_cells","workers"]:
        _positive_integer(cfg[name],name)
    _positive_integer(cfg["seed"],"seed",zero=True)
    if cfg["quadrature_order"]<8 or cfg["structural_samples"]<4:
        raise ValueError("adequate quadrature and >=4 validation draws required")
    if not isinstance(cfg["references"],dict) or not cfg["references"]:
        raise ValueError("named source maps required")
    lags=cfg["lags_cells"]
    if not isinstance(lags,list) or not lags or sorted(set(lags))!=lags or lags[0]!=1:
        raise ValueError("increasing lags beginning at the native increment required")
    for lag in lags:_positive_integer(lag,"lag")
    bounds=cfg["length_bounds_m"]
    if not isinstance(bounds,list) or len(bounds)!=2 or not np.all(np.isfinite(bounds)) or not 0<bounds[0]<bounds[1]:
        raise ValueError("positive increasing correlation-length bounds required")
    for name in cfg["references"]:_name(name,"source map")
    return source,cfg


def verify_artifact(path):
    path=Path(path).resolve();payload=json.loads(path.read_text())
    if payload.get("schema_version")!=1 or payload.get("family")!="matern32_radial":
        raise ValueError("supported structural-profile artifact required")
    files={str(path):sha256_file(path)}
    for row in [*payload["source_maps"].values(),*payload.get("supporting_files",[])]:
        target=_relative_path(row["path"].replace("\\","/"),path.parent)
        if sha256_file(target)!=row["sha256"]:
            raise ValueError("structural profile source map or provenance file changed")
        files[str(target)]=row["sha256"]
    return payload,files


def fit(config_path,output_path,report_dir):
    source,cfg=load_config(config_path);output=Path(output_path).resolve();report=Path(report_dir).resolve()
    if output.exists() or report.exists():
        raise ValueError("fit artifact and report directory must be fresh")
    output.parent.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True)
    payload={"schema_version":1,"family":"matern32_radial","cell_size_m":cfg["cell_size_m"],
        "settings":cfg,"source_maps":{},"profiles":{},"supporting_files":[],
        "configuration_sha256":sha256_file(source),"code_commit":git_head(source.parent),
        "runtime":runtime_versions(),"source_code_sha256":{
            Path(__file__).relative_to(Path(__file__).parents[3]).as_posix():sha256_file(__file__),
            **{f"src/subsurface_uq/sampling/{name}.py":sha256_file(Path(__file__).parents[1]/f"sampling/{name}.py")
               for name in ["structural_profiles","gstools_reference"]}},
        "interpretation":"separate stationary synthetic-map priors informed by processed-map structure; not geological reconstruction-error models",
        "source_independence":"RUN maps may share one parent raster; not independent geological realizations",
        "profile_probability_weights":None,"sample_filtering":None,
        "variance_estimator":"sigma² = observed population spatial variance/(1-q); q is the exact discrete correlation of a grid average",
        "fit_criterion":"equal-weight squared log-semivariogram residuals at all declared directions and lags; no iid-pixel likelihood",
        "mean_estimator":"full-map spatial log10 mean, treated as a fixed generative-prior parameter"}
    for name,raw in cfg["references"].items():
        path=_relative_path(raw,source.parent);field=np.load(path,mmap_mode="r",allow_pickle=False)
        if field.ndim!=2 or not np.all(np.isfinite(field)) or np.any(field<=0):
            raise ValueError("source maps must contain finite positive permeability")
        logs=np.log10(field.astype(float))
        if "native_shape" in payload and payload["native_shape"]!=list(logs.shape):
            raise ValueError("all structural maps must share a native grid")
        payload["native_shape"]=list(logs.shape)
        result=fit_profile(logs,cell_size_m=cfg["cell_size_m"],lags_cells=cfg["lags_cells"],
            length_bounds_m=cfg["length_bounds_m"],quadrature_order=cfg["quadrature_order"])
        law=result["law"];spec=GSToolsReferenceSpec(**law["covariance"])
        features=structural_features(logs,cell_size_m=cfg["cell_size_m"],mean_log10=law["mean_log10_k"],
            sigma_log10=spec.sigma_R,stride=cfg["diagnostic_stride_cells"])
        result.update(source_map=name,source_features=features,
            expected_native_gradient_rms={direction:float(np.sqrt(2*spec.sigma_R**2*(1-spec.correlation(dr*cfg["cell_size_m"],dc*cfg["cell_size_m"])))/cfg["cell_size_m"])
                for direction,(dr,dc) in list(DIRECTIONS.items())[:2]},
            local_coordinate_convention="angle from positive column toward positive row; no historical geographic registration inferred")
        payload["profiles"][name+"_structural"]=result
        payload["source_maps"][name]={"path":Path(os.path.relpath(path,output.parent)).as_posix(),"sha256":sha256_file(path),
            "shape":list(field.shape),"dtype":str(field.dtype),"units":"m2"}
        print(f"structural fit {name}: mean={law['mean_log10_k']:.4f}, point SD={spec.sigma_R:.4f}, lengths={spec.ell_row_m:.0f}/{spec.ell_col_m:.0f} m, angle={spec.angle_deg:.1f}",flush=True)
    if cfg["preparation_manifest"] is not None:
        path=_relative_path(cfg["preparation_manifest"],source.parent)
        payload["supporting_files"].append({"path":Path(os.path.relpath(path,output.parent)).as_posix(),"sha256":sha256_file(path),"role":"native-map preparation provenance"})
    payload["profile_set_id"]=_identity({"settings":cfg,"sources":payload["source_maps"],"profiles":payload["profiles"]},"structural-")
    _write(output,payload)
    # Archive the actual estimator/configuration for reproducibility.
    archive=report/"source_snapshot";archive.mkdir()
    for path in [source,Path(__file__),Path(__file__).parents[1]/"sampling/structural_profiles.py",Path(__file__).parents[1]/"sampling/gstools_reference.py"]:
        root=Path(__file__).parents[3]
        relative=path.relative_to(root) if path.is_relative_to(root) else Path("configuration")/path.name
        target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    _write(report/"fit_context.json",{"profile_file":str(output),"profile_file_sha256":sha256_file(output),
        "configuration":str(source),"configuration_sha256":sha256_file(source)})
    fit_figure(payload,report/"source_fits.png")
    return payload


def fit_figure(payload,path):
    fig,axes=plt.subplots(len(payload["profiles"]),2,figsize=(12,4*len(payload["profiles"])),squeeze=False,layout="constrained")
    for (name,p),row in zip(payload["profiles"].items(),axes):
        for direction,(dr,dc) in DIRECTIONS.items():
            distance=np.array(payload["settings"]["lags_cells"])*payload["cell_size_m"]*np.hypot(dr,dc)
            row[0].loglog(distance,p["empirical_variograms"][direction],"o",ms=3,label=direction)
            row[0].loglog(distance,p["fitted_variograms"][direction],"-")
        row[0].set(title=name,xlabel="Separation [m]",ylabel="Semivariance [log10²]");row[0].legend(fontsize=8)
        feature=p["source_features"];expected=p["expected_native_gradient_rms"]
        x=np.arange(2);row[1].bar(x-.16,[feature[f"gradient_rms_{d}_log10_per_m"] for d in ["row","column"]],width=.32,label="Source")
        row[1].bar(x+.16,[expected[d] for d in ["row","column"]],width=.32,label="Declared law")
        row[1].set_xticks(x,["Row","Column"]);row[1].set(title="Native 5 m roughness",ylabel="RMS log10 increment / m");row[1].legend(fontsize=8)
        for ax in row:ax.grid(alpha=.2)
    fig.suptitle("Structural profiles: map statistics inform a synthetic distribution\nCommon Matérn-3/2 family; total heterogeneity, finite-map correction; no reconstruction target")
    fig.savefig(path,dpi=140);plt.close(fig)


def validate(profile_path,report_dir):
    payload,files=verify_artifact(profile_path);profile_path=Path(profile_path).resolve();report=Path(report_dir).resolve()
    cfg=payload["settings"];shape=payload["native_shape"];records={}
    marker=report/"structural_validation.json"
    if marker.exists():
        raise ValueError("structural validation already exists; preserve frozen draws")
    report.mkdir(parents=True,exist_ok=True)
    stride=cfg["diagnostic_stride_cells"]
    lags=[int(k//stride) for k in cfg["lags_cells"] if k>=stride and k%stride==0]
    for name,p in payload["profiles"].items():
        target=report/name;target.mkdir()
        law=p["law"];spec=GSToolsReferenceSpec(**law["covariance"])
        sampler=GSToolsReferenceFFTSampler(replace(spec,sigma_R=1.),shape,cell_size_m=payload["cell_size_m"],
            workers=cfg["workers"],embedding_shape=cfg["paired_embedding_shape"])
        rows=[]
        for i in range(cfg["structural_samples"]):
            seed=seed_for(cfg["seed"],"structural-input-validation",i)
            logs=law["mean_log10_k"]+spec.sigma_R*sampler.sample_residual(seed)
            field=physical_permeability(logs)
            features=structural_features(logs,cell_size_m=payload["cell_size_m"],mean_log10=law["mean_log10_k"],sigma_log10=spec.sigma_R,stride=stride)
            curves=variogram_curves(logs[::stride,::stride],lags)
            rows.append({"sample_index":i,"seed":seed,"permeability_sha256":array_hash(field),"features":features,
                         "variograms":{k:v.tolist() for k,v in curves.items()}})
            if i==0:
                np.save(target/"sample-0000.npy",field)
                source=np.load(_relative_path(payload["source_maps"][p["source_map"]]["path"],profile_path.parent),mmap_mode="r")
                fig,axes=plt.subplots(1,2,figsize=(9,4),layout="constrained")
                arrays=[np.log10(source[::stride,::stride].astype(float)),logs[::stride,::stride]]
                lower=min(a.min() for a in arrays);upper=max(a.max() for a in arrays)
                for ax,a,title in zip(axes,arrays,["Source map","Independent synthetic draw"]):
                    artist=ax.imshow(a,origin="lower",vmin=lower,vmax=upper,cmap="viridis");ax.set_title(title);ax.set_xlabel("Column cells / diagnostic stride");ax.set_ylabel("Row cells / diagnostic stride")
                    fig.colorbar(artist,ax=ax,label="log10 k [m2]",shrink=.8)
                fig.suptitle(name+": structural comparison; no pixel reconstruction target")
                fig.savefig(target/"example.png",dpi=140);plt.close(fig)
            print(f"structural validation {name}: {i+1}/{cfg['structural_samples']}",flush=True)
        keys=list(rows[0]["features"])
        envelopes={k:{"source":p["source_features"][k],"q05_q50_q95":np.quantile(np.array([r["features"][k] for r in rows],dtype=float),[.05,.5,.95]).tolist()} for k in keys}
        records[name]={"draws":rows,"feature_envelopes":envelopes,"sampler_metadata":sampler.metadata,
            "diagnostic_variogram_lags_cells":lags,"diagnostic_cell_size_m":stride*payload["cell_size_m"],
            "expected_grid_spatial_variance_log10":p["expected_grid_spatial_variance_log10"],
            "mean_generated_grid_spatial_variance_log10":float(np.mean([r["features"]["spatial_variance_log10"] for r in rows])),
            "sample_filtering":None,"interpretation":"internal structural check and generator diagnostics; no geological probability calibration or iid-pixel inference"}
        records[name]["sampler_metadata"].update(
            field_role="unit Gaussian driving field, scaled by the declared total-heterogeneity SD",
            field_distribution="Y=mu+sigma*Z; k=10**Y; no reference multiplier or moment normalization")
        records[name]["sampler_metadata"]["spec"].update(
            marginal="zero-mean unit Gaussian driving field",
            assumption_status="map-informed synthetic structural prior; no calibrated site posterior")
        del sampler
    for path,h in files.items():
        if sha256_file(path)!=h:raise ValueError("profile/source changed during validation")
    result={"profile_set_id":payload["profile_set_id"],"profile_file_sha256":sha256_file(profile_path),"profiles":records,
        "profile_probability_weights":None,"paired_embedding_shape":cfg["paired_embedding_shape"],
        "connectivity_support":"4-neighbour components on point-subsampled grid; thresholds fixed at law mean +/- 1.28155 SD",
        "validation_envelopes":"finite-draw descriptive spread, not confidence or independent geographical validation"}
    _write(marker,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config");parser.add_argument("--output",required=True,help="structural-profile JSON")
    parser.add_argument("--report-dir",required=True);parser.add_argument("--stage",choices=("fit","validate","all"),default="all")
    args=parser.parse_args(argv)
    if args.stage in {"fit","all"}:
        if args.config is None:parser.error("--config required for fitting")
        fit(args.config,args.output,args.report_dir)
    if args.stage in {"validate","all"}:validate(args.output,args.report_dir)


if __name__=="__main__":main()
