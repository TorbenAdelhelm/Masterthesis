"""Frozen release25 context for studies that supply their own permeability law.

Only release25, qoi and streamlines settings are read from the model YAML. The
RQ1 sampling/GRF configuration is deliberately not interpreted as an FFT law.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..io import sha256_file
from .bounded_streamlines import configure_release25_streamlines
from .release25 import Release25Surrogate
from .release25_runtime import Release25Runtime, _checkpoint, _one_file


def _mapping(value, name, allowed):
    if not isinstance(value, dict) or not set(value) <= set(allowed):
        raise ValueError(f"{name} must be a mapping with known keys: {sorted(allowed)}")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass
class FrozenTemperatureModel:
    surrogate: Release25Surrogate
    input_shape: tuple[int, int]
    cell_size_m: float
    background_temperature: float
    receptors: tuple[tuple[int, int], ...]
    mean_anomaly_roi: tuple[int, int, int, int] | None
    metadata: dict
    normalization_bounds: dict

    def predict_temperature(self, field):
        if np.asarray(field).shape != self.input_shape:
            raise ValueError("permeability grid does not match fixed model inputs")
        return self.surrogate.predict_temperature(field)

    def normalization_diagnostics(self, field):
        # CNN3 permeability is cropped to the intermediate velocity grid.
        velocity_shape = self.surrogate.adapter.last_velocity_shape
        result = {}
        for role, values in (("cnn1", field), ("cnn3", _crop(field, velocity_shape))):
            lower, upper = self.normalization_bounds[role]
            result[role] = {"minimum": float(values.min()), "maximum": float(values.max()),
                            "fraction_below": float(np.mean(values < lower)),
                            "fraction_above": float(np.mean(values > upper)),
                            "bounds": [lower, upper]}
        result["interpretation"] = "checkpoint normalization ranges; not a training-support certificate"
        return result


def _crop(field, shape):
    if len(shape) != 2 or any(n > m for n, m in zip(shape, field.shape)):
        raise ValueError("velocity crop does not fit permeability grid")
    r, c = ((m-n)//2 for m, n in zip(field.shape, shape))
    return field[r:r+shape[0], c:c+shape[1]]


class _CaptureVelocityShape:
    def __init__(self, adapter):
        self.adapter = adapter

    def predict(self, permeability, fixed_inputs):
        result = self.adapter.predict(permeability, fixed_inputs)
        self.last_velocity_shape = tuple(int(v) for v in result["velocity"].shape[-2:])
        return result


def load_frozen_temperature_model(path, *, asset_root=None, torch_threads=8):
    """Load actual checkpoints and fixed inputs; hash the evaluated context.

    Assets resolve relative to ``asset_root`` or the model YAML's directory.
    An explicit asset root permits reuse of older repository-relative RQ1 YAML.
    All RUN references share the declared pressure/material/source scenario.
    """
    import torch

    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("model configuration must be a mapping")
    release = _mapping(raw.get("release25"), "release25", {
        "repo", "cnn1_dir", "cnn2_dir", "prepared_pki_dir", "fixed_run_id", "device",
        "cell_size_m", "background_temperature", "random_k",
    })
    qoi = _mapping(raw.get("qoi"), "qoi", {"receptors", "mean_anomaly_roi"})
    streams = _mapping(raw.get("streamlines", {}), "streamlines", {
        "mode", "method", "max_nfev", "diagnostics", "slow_seconds",
    })
    root = source.parent if asset_root is None else Path(asset_root).expanduser().resolve()

    def asset(name):
        value = release.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"release25.{name} must be a path")
        target = Path(value).expanduser()
        return (root/target).resolve() if not target.is_absolute() else target.resolve()

    paths = {name: asset(name) for name in ("repo", "cnn1_dir", "cnn2_dir", "prepared_pki_dir")}
    if release.get("random_k") is not False or release.get("cell_size_m") != 5.:
        raise ValueError("conditional reference studies require explicit real-K semantics and the native 5 m grid")
    background = release.get("background_temperature")
    if isinstance(background, bool) or not isinstance(background, (int, float)) or not np.isfinite(background):
        raise ValueError("background_temperature must be finite")
    run = release.get("fixed_run_id")
    if not isinstance(run, str) or not run or Path(run).name != run or "/" in run or "\\" in run:
        raise ValueError("fixed_run_id must be a single RUN name")
    receptors = qoi.get("receptors", [])
    if not isinstance(receptors, list) or any(not isinstance(v, list) or len(v) != 2 for v in receptors):
        raise ValueError("receptors must be a list of [row,column] output-grid cells")
    receptors = tuple(tuple(_integer(v, "receptor cell") for v in pair) for pair in receptors)
    if len(set(receptors)) != len(receptors):
        raise ValueError("receptors must be distinct")
    roi = qoi.get("mean_anomaly_roi")
    if roi is not None:
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError("mean_anomaly_roi must be [r0,r1,c0,c1] or null")
        roi = tuple(_integer(v, "ROI boundary") for v in roi)
        if roi[0] >= roi[1] or roi[2] >= roi[3]:
            raise ValueError("mean_anomaly_roi must have positive extent")
    if not receptors and roi is None:
        raise ValueError("configure at least one receptor or mean-anomaly ROI")
    mode, method = streams.get("mode", "bounded"), streams.get("method", "RK45")
    if mode not in {"bounded", "release25"} or method not in {"RK45", "RK23", "Radau"}:
        raise ValueError("unsupported streamline mode/method")
    max_nfev = _integer(streams.get("max_nfev", 100000), "max_nfev")
    diagnostics = streams.get("diagnostics", False)
    slow = streams.get("slow_seconds", 2.)
    if not isinstance(diagnostics, bool) or isinstance(slow, bool) or not np.isfinite(slow) or slow < 0:
        raise ValueError("invalid streamline diagnostics/slow_seconds")
    info = paths["prepared_pki_dir"]/"info.yaml"
    prepared = yaml.safe_load(info.read_text(encoding="utf-8"))
    shape = tuple(prepared["CellsNumber"])
    if len(shape) != 2 or any(_integer(v, "grid dimension", 3) != v for v in shape):
        raise ValueError("prepared inputs must describe a 2-D grid")
    if not np.array_equal(prepared["CellsSize"][:2], [5., 5.]):
        raise ValueError("prepared input spacing must be 5 m")
    tensors = [p for p in (paths["prepared_pki_dir"]/"Inputs"/run,
                           paths["prepared_pki_dir"]/"Inputs"/f"{run}.pt") if p.is_file()]
    if len(tensors) != 1:
        raise ValueError("exactly one fixed input tensor required")
    files = [source, info, tensors[0]]
    for role in ("cnn1_dir", "cnn2_dir"):
        files.extend([_checkpoint(paths[role]), _one_file(paths[role], "info.yaml"),
                      _one_file(paths[role], "HPS_options.yaml")])
    code = paths["repo"]/"code" if (paths["repo"]/"code").is_dir() else paths["repo"]
    files.extend(sorted(code.rglob("*.py")))
    if not list(code.rglob("*.py")):
        raise ValueError("upstream model code is missing")
    metadata = {"kind": "frozen_release25_lgcnn", "fixed_run_id": run,
                "paths": {k: str(v) for k,v in paths.items()}, "asset_root": str(root),
                "files_sha256": {str(p): sha256_file(p) for p in files},
                "release25": release, "qoi": qoi, "streamlines": streams,
                "torch_threads": _integer(torch_threads, "torch_threads", 1),
                "uncertainty": "permeability input only; model weights and pressure/material/sources fixed",
                "input_law": "supplied by the conditional FFT study; model YAML grf/sampling blocks ignored"}
    torch.set_num_threads(torch_threads)
    runtime = Release25Runtime.from_paths(release25_repo=paths["repo"], cnn1_dir=paths["cnn1_dir"],
        cnn2_dir=paths["cnn2_dir"], prepared_pki_dir=paths["prepared_pki_dir"], run_id=run,
        device=release.get("device", "cpu"), random_k=False, streamline_method=method)
    if tuple(runtime.scenario.shape) != shape:
        raise ValueError("prepared tensor shape differs from its metadata")
    configure_release25_streamlines(runtime.adapter, mode=mode, max_nfev=max_nfev,
                                   diagnostics=diagnostics, slow_streamline_seconds=slow)
    bounds = {}
    for role, bundle in (("cnn1", runtime.adapter.cnn1), ("cnn3", runtime.adapter.cnn2)):
        channels = [v for k,v in bundle.info["Inputs"].items() if "permeability" in k.lower()]
        if len(channels) != 1:
            raise ValueError("one permeability normalization channel required per checkpoint")
        bounds[role] = (float(channels[0]["min"]), float(channels[0]["max"]))
        if not np.all(np.isfinite(bounds[role])) or bounds[role][0] >= bounds[role][1]:
            raise ValueError("invalid checkpoint permeability normalization range")
    surrogate = Release25Surrogate(_CaptureVelocityShape(runtime.adapter), runtime.scenario.fixed,
                                   device=release.get("device", "cpu"))
    return FrozenTemperatureModel(surrogate, shape, 5., float(background), receptors, roi, metadata, bounds)
