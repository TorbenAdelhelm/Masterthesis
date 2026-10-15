"""Exact permeability replay from generalized kernel-screen outputs."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import qmc

from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.reference_field import save_reference_field_input_model
from subsurface_uq.sampling.scenarios import ReferenceScenario
from subsurface_uq.visualization.reference_fields_cli import main, pilot_preview_sources


def _screen(root, method="RQMC", grouped=False):
    pilot = root / "pilot"
    pilot.mkdir()
    rows, artifacts, expected = [], [], {}
    reference = 10. ** np.linspace(-10., -9., 36).reshape(6, 6)
    context = {"study_mode": "kernel-screen", "artifacts": artifacts,
               "method": method, "seed": 13, "n_samples": 4}
    # Different kernels use separate coordinate dimensions. An explicit KL
    # group additionally shares a 5-D design prefix for its smaller member.
    definitions = [("gaussian", 2, "RUN_1"), ("matern32", 3, "RUN_2")]
    if grouped:
        definitions.append(("gaussian", 5, "RUN_1"))
    for kernel, n_modes, run in definitions:
        scenario = ReferenceScenario(reference_run=run, sigma_R=.05, covariance_family=kernel,
                                     ell_y=10., ell_x=20., n_modes=n_modes, energy_threshold=None)
        field_map, _ = scenario.build_maps(reference, cell_size_m=5.)
        sid = scenario.scenario_id
        path = root / "artifacts" / sid / "input.yaml"
        save_reference_field_input_model(path, field_map)
        artifacts.append({"scenario_id": sid, "input_model": {"path": str(path), "sha256": sha256_file(path)}})
        dimension = 5 if grouped and kernel == "gaussian" else n_modes
        rows.append({"scenario_id": sid, "dimension": n_modes, "coordinate_dimension": dimension,
                     "role": "kernel sensitivity", "config": scenario.manifest()["config"]})
        coordinates = (np.random.default_rng(13).normal(size=(4, dimension)) if method == "MC"
                       else ndtri(qmc.Sobol(dimension, scramble=True, seed=13).random_base2(2)))
        expected[sid] = [field_map.map_coordinates(xi[:n_modes]) for xi in coordinates]
    context_id = sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (pilot / "run_context.json").write_text(json.dumps({"context_id": context_id, "context": context}))
    (pilot / "summary.json").write_text(json.dumps({"study_mode": "kernel-screen", "context_id": context_id, "scenarios": rows}))
    for sid, fields in expected.items():
        directory = pilot / sid
        directory.mkdir()
        records = [{"sample_index": i, "context_id": context_id,
                    "physical_field_sha256": sha256(field.tobytes()).hexdigest()}
                   for i, field in enumerate(fields)]
        (directory / "sample_diagnostics.json").write_text(json.dumps(records))
    if grouped:
        group = {"context_id": context_id, "coordinate_dimension": 5,
                 "scenario_ids": [row["scenario_id"] for row in rows if row["config"]["covariance_family"] == "gaussian"]}
        (pilot / "qoi_truncation.json").write_text(json.dumps({"context_id": context_id, "groups": [group]}))
    return pilot, rows, expected


@pytest.mark.parametrize("method", ["MC", "RQMC"])
@pytest.mark.parametrize("grouped", [False, True])
def test_screen_replays_all_selected_kernel_inputs_and_optional_shared_kl_design(tmp_path, method, grouped):
    pilot, rows, expected = _screen(tmp_path, method, grouped)
    output = tmp_path / "plots"
    assert main(["--pilot-output", str(pilot), "--output-dir", str(output),
                 "--sample-indices", "1", "--save-arrays"]) == 0
    report = json.loads((output / "manifest.json").read_text())
    assert len(report["samples"]) == len(rows)
    assert {row["scenario_id"] for row in report["samples"]} == set(expected)
    for row in report["samples"]:
        sid = row["scenario_id"]
        np.testing.assert_array_equal(np.load(output / row["generated_npy"]), expected[sid][1])
        assert row["pilot_input_hash_verified"]
        assert row["coordinate_dimension"] == next(r["coordinate_dimension"] for r in rows if r["scenario_id"] == sid)
        assert (output / row["png"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_screen_filters_by_reference_or_explicit_id_and_uses_group_fallback(tmp_path):
    pilot, rows, _ = _screen(tmp_path, grouped=True)
    gaussian_ids = {row["scenario_id"] for row in rows if row["config"]["reference_run"] == "RUN_1"}
    assert {row["scenario_id"] for row in pilot_preview_sources(pilot, reference_run="RUN_1")} == gaussian_ids
    assert len(pilot_preview_sources(pilot, scenario_ids=[rows[0]["scenario_id"]])) == 1
    summary = json.loads((pilot / "summary.json").read_text())
    for row in summary["scenarios"]:
        row.pop("coordinate_dimension")
    (pilot / "summary.json").write_text(json.dumps(summary))
    assert all(row["coordinate_dimension"] == 5 for row in pilot_preview_sources(pilot, reference_run="RUN_1"))
    with pytest.raises(ValueError, match="unknown scenario"):
        pilot_preview_sources(pilot, scenario_ids=["unknown"])


@pytest.mark.parametrize("corruption", ["context", "artifact", "diagnostics", "sample_index", "field_hash"])
def test_screen_retains_integrity_checks(tmp_path, corruption):
    pilot, rows, _ = _screen(tmp_path)
    sid = rows[0]["scenario_id"]
    if corruption == "context":
        path = pilot / "run_context.json"
        data = json.loads(path.read_text())
        data["context"]["seed"] = 14
        path.write_text(json.dumps(data))
        message = "context checksum"
    elif corruption == "artifact":
        context = json.loads((pilot / "run_context.json").read_text())["context"]
        Path(context["artifacts"][0]["input_model"]["path"]).write_text("changed")
        message = "artifact checksum"
    else:
        path = pilot / sid / "sample_diagnostics.json"
        data = json.loads(path.read_text())
        if corruption == "diagnostics":
            data[0]["context_id"] = "0" * 64
            message = "different pilot context"
        elif corruption == "sample_index":
            data[0]["sample_index"] = 1
            message = "missing or duplicated"
        else:
            data[0]["physical_field_sha256"] = "0" * 64
            message = "does not match the queried pilot input"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        main(["--pilot-output", str(pilot), "--output-dir", str(tmp_path / "bad"), "--sample-indices", "0"])


@pytest.mark.parametrize("corruption", ["record_context", "group_context", "dimension", "unknown_group_id"])
def test_screen_validates_optional_kl_group_context_and_design(tmp_path, corruption):
    pilot, rows, _ = _screen(tmp_path, grouped=True)
    path = pilot / "qoi_truncation.json"
    data = json.loads(path.read_text())
    if corruption == "record_context":
        data["context_id"] = "0" * 64
        message = "different run context"
    elif corruption == "group_context":
        data["groups"][0]["context_id"] = "0" * 64
        message = "different run context"
    elif corruption == "dimension":
        data["groups"][0]["coordinate_dimension"] = 6
        message = "coordinate dimension disagrees"
    else:
        data["groups"][0]["scenario_ids"].append("unknown")
        message = "distinct known scenario IDs"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        pilot_preview_sources(pilot)
