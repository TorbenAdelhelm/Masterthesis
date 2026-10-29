"""Prepare full-resolution reference fields and one fixed real-K pki scenario.

Run from the repository root in an environment with the package and h5py:

    python examples/reference_pilot/prepare_data.py \
      --dataset-root "C:/path/to/dataset_100hp_giant_real_fixP0_0025" \
      --cnn1-dir "C:/path/to/LGCNN_step1_realK/model_folder" \
      --release25-repo "C:/path/to/Heat-Plume-Prediction" \
      --output-dir data/reference_pilot

This prepares inputs only. It neither loads checkpoints nor evaluates the CNNs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import yaml

from subsurface_uq.io import git_head, runtime_versions, sha256_file
from subsurface_uq.sampling.darus_real import (
    RELEASE25_INITIAL_TIME_GROUP, RELEASE25_PERMEABILITY_DATASET,
    load_release25_raw_permeability_run,
)
from subsurface_uq.surrogates.release25_runtime import (
    PreparedLGCNNScenario, Release25Normalizer,
)


def _provenance(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "bytes": path.stat().st_size}


def _code_dir(repo: Path) -> Path:
    for path in (repo, repo / "code"):
        if (path / "preprocessing" / "preprocessing.py").is_file():
            return path
    raise FileNotFoundError(f"release25 preprocessing code not found below {repo}")


def _hparam(hps: dict, name: str):
    value = hps[name]
    if isinstance(value, dict) and "values" in value:
        values = value["values"]
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"pilot requires one explicit standard-model value for {name}")
        return values[0]
    return value


def prepare_data(args) -> dict[str, object]:
    dataset = Path(args.dataset_root).expanduser().resolve()
    cnn1 = Path(args.cnn1_dir).expanduser().resolve()
    release25 = Path(args.release25_repo).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    code = _code_dir(release25)
    settings_path = dataset / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    sizes = np.asarray(settings["grid"]["size [m]"], dtype=float)
    resolution = float(settings["grid"]["resolution"])
    if resolution != 5.0 or sizes.shape != (2,) or not np.allclose(sizes, (12800., 12800.)):
        raise ValueError("pilot requires the original 12800 m square, 5 m grid")
    # This is the upstream 2025 preparation shape before the singleton is removed.
    raw_shape = (2560, 2560, 1)
    expected_shape = raw_shape[:2]
    info_path, hps_path = cnn1 / "info.yaml", cnn1 / "HPS_options.yaml"
    model_info = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    hps = yaml.safe_load(hps_path.read_text(encoding="utf-8"))
    input_code = str(_hparam(hps, "inputs"))
    if input_code != "pki" or not np.allclose(model_info["CellsSize"][:2], (5., 5.)):
        raise ValueError("pilot requires a real-K pki CNN1 model on the 5 m grid")
    names = ("Liquid Pressure [Pa]", RELEASE25_PERMEABILITY_DATASET, "Material ID")
    if (len(model_info["Inputs"]) != 3
            or any(model_info["Inputs"][name]["index"] != i for i, name in enumerate(names))):
        raise ValueError("CNN1 normalization indices must match upstream pki channel order")
    if len(set(args.reference_runs)) != len(args.reference_runs):
        raise ValueError("reference runs must be unique")
    fixed_source = dataset / args.fixed_run / "pflotran.h5"
    if not fixed_source.is_file():
        raise FileNotFoundError(f"fixed pressure/material scenario requires {fixed_source}")
    model_files = [cnn1 / "model.pt", info_path, hps_path]
    command_file = cnn1 / "command_line_arguments.yaml"
    if command_file.is_file():
        model_files.append(command_file)
    upstream_files = [code / relative for relative in (
        "preprocessing/preprocessing.py", "preprocessing/transforms.py",
        "preprocessing/preparing_datasets/raw_data_loading.py",
    )]
    for path in [*model_files, *upstream_files]:
        if not path.is_file():
            raise FileNotFoundError(path)
    target_paths = [output / f"{run}.reference.npy" for run in args.reference_runs]
    target_paths += [output / "prepared_pki" / "Inputs" / f"{args.fixed_run}.pt",
                     output / "prepared_pki" / "info.yaml", output / "preparation_manifest.json"]
    if any(path.exists() for path in target_paths):
        raise FileExistsError("pilot preparation outputs already exist; select a new output directory")

    # Import the actual upstream extraction/transforms, rather than guessing an
    # orientation or copying channel values from a synthetic prepared scenario.
    sys.path.insert(0, str(code))
    raw_loader = importlib.import_module("preprocessing.preparing_datasets.raw_data_loading")
    transforms = importlib.import_module("preprocessing.transforms")
    upstream = {"code_dir": str(code), "git_sha": git_head(release25),
                "recipe_files": [_provenance(path) for path in upstream_files]}
    report: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_script": _provenance(Path(__file__)),
        "masterthesis_git_sha": git_head(Path(__file__).resolve().parents[2]),
        "runtime": runtime_versions(), "dataset_root": str(dataset),
        "dataset_settings": {**_provenance(settings_path), "content": settings},
        "cell_size_m": 5.0, "full_field_shape": list(expected_shape),
        "coarsening": None, "permeability_convention": "historical-training",
        "source_policy": "RUNs are separate reference scenarios; no geological pooling or inferred weights",
        "upstream": upstream,
        "cnn1_model_files": [_provenance(path) for path in model_files],
        "references": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for run in args.reference_runs:
        field = load_release25_raw_permeability_run(dataset, run, cell_size_m=5.)
        if field.shape != expected_shape:
            raise ValueError(f"unexpected full-resolution shape for {run}: {field.shape}")
        source = dataset / run / "pflotran.h5"
        output_backed = source.is_file()
        if not output_backed:
            source = dataset / run / "permeability.h5"
        destination = output / f"{run}.reference.npy"
        np.save(destination, field, allow_pickle=False)
        row = {
            "reference_run": run, "source": _provenance(source),
            "reference_npy": _provenance(destination), "shape": list(field.shape),
            "dtype": str(field.dtype), "range_m2": [float(field.min()), float(field.max())],
            "extraction": {
                "loader": "load_release25_raw_permeability_run",
                "raw_grid_shape": list(raw_shape), "array_order": "C reshape, singleton squeeze; no transpose/flip",
                "time_group": RELEASE25_INITIAL_TIME_GROUP if output_backed else None,
                "dataset": RELEASE25_PERMEABILITY_DATASET if output_backed else "permeability",
                "cell_ids": None if output_backed else "explicit one-based Cell Ids are reordered before reshape",
                "value_conversion": None,
            },
        }
        run_settings = dataset / run / "settings.yaml"
        if run_settings.is_file():
            row["run_settings"] = {**_provenance(run_settings),
                                   "content": yaml.safe_load(run_settings.read_text(encoding="utf-8"))}
        report["references"].append(row)
        del field

    # The upstream helper reads all pki inputs at the initial time. Its 2-D
    # transform leaves this singleton-depth data in the same C-reshape layout.
    raw_inputs = raw_loader.load_raw_data(
        fixed_source, RELEASE25_INITIAL_TIME_GROUP, list(names), raw_shape,
        "   1 Time  2.75000E+01 y", print_bool=False)
    hp_locations = raw_loader.get_hp_location(raw_inputs)
    raw_inputs = transforms.get_transforms()(raw_inputs, loc_hp=hp_locations)
    physical = transforms.ToTensorTransform()(raw_inputs).float()
    if tuple(physical.shape) != (3, *expected_shape) or not torch.isfinite(physical).all():
        raise ValueError("upstream fixed pki extraction returned unexpected/invalid data")
    if not set(torch.unique(physical[2]).tolist()) <= {1., 2.}:
        raise ValueError("expected upstream Material ID background=1, heat-pump=2")
    if not torch.all(physical[1] > 0):
        raise ValueError("fixed permeability must be positive")
    with h5py.File(fixed_source, "r") as handle:
        initial_temperature = np.asarray(handle[RELEASE25_INITIAL_TIME_GROUP]["Temperature [C]"])
        temperature_range = [float(initial_temperature.min()), float(initial_temperature.max())]
    if not np.allclose(temperature_range, (10.6, 10.6), rtol=0, atol=1e-6):
        raise ValueError(f"expected uniform 10.6 C initial temperature, got {temperature_range}")
    normalizer = Release25Normalizer(model_info)
    normalized = normalizer.transform(physical, "Inputs")
    # Also compare against release25's own normalization, before persisting.
    upstream_normalized = transforms.NormalizeTransform(model_info)(physical.clone(), "Inputs")
    torch.testing.assert_close(normalized, upstream_normalized, rtol=0, atol=0)
    prepared = output / "prepared_pki"
    (prepared / "Inputs").mkdir(parents=True, exist_ok=True)
    tensor_path = prepared / "Inputs" / f"{args.fixed_run}.pt"
    torch.save(normalized, tensor_path)
    info = deepcopy(model_info)
    info["CellsNumberPrior"] = info["CellsNumber"]
    info["PositionHPPrior"] = info["PositionLastHP"]
    info["CellsNumber"] = list(expected_shape)
    info["CellsSize"] = [5., 5., 5.]
    info["PositionLastHP"] = hp_locations.tolist()
    info["goal resolution"] = 5.
    info["reference_pilot_fixed_run"] = args.fixed_run
    info["background_temperature_C"] = 10.6
    prepared_info = prepared / "info.yaml"
    prepared_info.write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    recovered = PreparedLGCNNScenario.from_prepared_pki(prepared, args.fixed_run, device="cpu")
    torch.testing.assert_close(recovered.original_permeability, physical[1], rtol=3e-6, atol=0)
    torch.testing.assert_close(recovered.fixed.pressure, physical[0], rtol=2e-7, atol=0)
    torch.testing.assert_close(recovered.fixed.material_id, physical[2], rtol=0, atol=0)
    if args.fixed_run in args.reference_runs:
        reference = np.load(output / f"{args.fixed_run}.reference.npy", allow_pickle=False)
        np.testing.assert_array_equal(physical[1].numpy(), reference)
    report["fixed_scenario"] = {
        "run_id": args.fixed_run, "prepared_pki_dir": str(prepared),
        "source": _provenance(fixed_source), "input_tensor": _provenance(tensor_path),
        "info_yaml": _provenance(prepared_info), "input_code": input_code,
        "channel_order": list(names), "time_group": RELEASE25_INITIAL_TIME_GROUP,
        "raw_grid_shape": list(raw_shape), "array_order": "upstream C reshape and 2-D squeeze; no transpose/flip",
        "normalization": "real-K CNN1 info.yaml; exact agreement with upstream NormalizeTransform",
        "physical_pressure_range_Pa": [float(physical[0].min()), float(physical[0].max())],
        "physical_permeability_range_m2": [float(physical[1].min()), float(physical[1].max())],
        "heat_pump_pixel_count": int(torch.count_nonzero(physical[2] == 2)),
        "heat_pump_locations_raw_array": np.argwhere(physical[2].numpy() == 2).tolist(),
        "initial_temperature_range_C": temperature_range, "background_temperature_C": 10.6,
        "roundtrip_max_abs_error": {
            "pressure_Pa": float(torch.max(torch.abs(recovered.fixed.pressure - physical[0]))),
            "permeability_m2": float(torch.max(torch.abs(recovered.original_permeability - physical[1]))),
            "material_id": float(torch.max(torch.abs(recovered.fixed.material_id - physical[2]))),
        },
        "fixed_input_policy": "one actual pressure/material/source scenario; only permeability varies downstream",
    }
    manifest = output / "preparation_manifest.json"
    manifest.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Prepared {len(args.reference_runs)} full 5 m reference fields and fixed {args.fixed_run}")
    print(f"Preparation provenance: {manifest}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cnn1-dir", required=True, help="Extracted real-K CNN1 standard model folder")
    parser.add_argument("--release25-repo", required=True, help="Original release25 repo or its code directory")
    parser.add_argument("--reference-runs", nargs="+", default=["RUN_1", "RUN_2", "RUN_3"])
    parser.add_argument("--fixed-run", default="RUN_2")
    parser.add_argument("--output-dir", default="data/reference_pilot")
    args = parser.parse_args(argv)
    prepare_data(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
