"""Fit separate reference-texture models with GSTools; never fit residual sigma_R."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from subsurface_uq.io import git_head, sha256_file, runtime_versions
from subsurface_uq.sampling.calibration import estimate_directional_variograms
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec


def fit_references(config_path, output_path):
    config_path, output_path = Path(config_path).resolve(), Path(output_path).resolve()
    if output_path.exists():
        raise ValueError("fit output exists; choose a fresh path")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cell = float(config["cell_size_m"])
    report = {"runtime": runtime_versions(("gstools", "numpy", "scipy")),
        "source_commit": git_head(Path(__file__).parents[2]),
        "fit_source_sha256": sha256_file(Path(__file__)),
        "reference_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "interpretation": "separate reference-texture fits; residual sigma_R and covariance remain uncertainty assumptions",
        "settings": {"cell_size_m": cell, "spatial_stride": 8, "max_lag_cells": 96,
                     "lag_windows_m": [1280., 2560.], "rotation_rad": 0., "nugget": 0.,
                     "sill": "reference spatial log10 variance; normalization only",
                     "fit_loss": "linear least squares", "detrending": None,
                     "held_out": "both diagonals; axis fitting only", "support": "point subsampling, no block averaging"},
        "references": {}}
    for run, raw_path in config["references"].items():
        path = (config_path.parent / raw_path).resolve()
        reference = np.load(path, allow_pickle=False).astype(float)
        logs = np.log10(reference)
        sill = float(np.var(logs))
        if not np.isfinite(sill) or sill <= 0:
            raise ValueError("positive reference spatial variance required")
        variograms = estimate_directional_variograms(reference, cell_size_m=cell, spatial_stride=8, max_lag_cells=96)
        sampled = logs[::8, ::8]
        distances = variograms["x"][0]
        minus = np.array([np.mean((sampled[lag:,:-lag]-sampled[:-lag,lag:])**2)/2
                          for lag in range(1,len(distances)+1)]) / sill
        plus = variograms["diag"][1] / sill
        empirical = np.array([variograms["x"][1],variograms["y"][1]]) / sill
        item = {"reference_path": str(path), "reference_sha256": sha256_file(path),
                "reference_shape": list(reference.shape), "spatial_sill_log10": sill, "fits": [],
                "empirical": {"axis_lags_m": distances.tolist(), "normalized_semivariogram_col": empirical[0].tolist(),
                              "normalized_semivariogram_row": empirical[1].tolist(), "diagonal_lags_m": (distances*np.sqrt(2)).tolist(),
                              "normalized_semivariogram_diag_plus": plus.tolist(), "normalized_semivariogram_diag_minus": minus.tolist()}}
        for window in (1280.,2560.):
            mask = distances <= window
            for family in ("gaussian","matern32_radial","exponential_radial"):
                model = GSToolsReferenceSpec(family,1000.,1000.).gstools_model()
                model.set_arg_bounds(len_scale=[20.,51200.],anis=[.02,50.])
                select = {"var":False,"nugget":False}
                if family == "matern32_radial":
                    select["nu"] = False
                model.fit_variogram(distances[mask],empirical[:,mask],init_guess="current",loss="linear",max_eval=5000,**select)
                axes = np.array([model.vario_axis(distances,axis=0),model.vario_axis(distances,axis=1)])
                held = distances*np.sqrt(2) <= window
                dp = model.vario_spatial([distances,distances])
                dm = model.vario_spatial([distances,-distances])
                item["fits"].append({"family":family,"fit_max_lag_m":window,
                    "ell_row_m":float(model.len_scale_vec[1]),"ell_col_m":float(model.len_scale_vec[0]),
                    "gstools_rescale":model.rescale,"gstools_nu":1.5 if family=="matern32_radial" else None,
                    "normalized_axis_fit_rmse":float(np.sqrt(np.mean((axes[:,mask]-empirical[:,mask])**2))),
                    "normalized_heldout_diagonal_rmse":float(np.sqrt(np.mean(np.concatenate([dp[held]-plus[held],dm[held]-minus[held]])**2))),
                    "gstools_model":str(model)})
        report["references"][run] = item
        print(f"Fitted six radial texture models for {run}",flush=True)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-config",required=True,help="YAML with references mapping and cell_size_m")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    fit_references(args.reference_config,args.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
