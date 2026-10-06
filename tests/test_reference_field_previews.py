"""Check spatial agreement and exact replay of paired pilot input designs."""
from hashlib import sha256
import json

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import qmc

from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.reference_field import build_reference_field_maps, save_reference_field_input_model
from subsurface_uq.visualization.reference_fields import reference_field_agreement
from subsurface_uq.visualization.reference_fields_cli import export_reference_previews, main


def test_agreement_known_multiplicative_shift_and_identity():
    reference = 10. ** np.linspace(-11., -9., 24).reshape(4, 6)
    shifted = reference_field_agreement(reference, 2 * reference)
    assert shifted["log10_rmse"] == pytest.approx(np.log10(2))
    assert shifted["log10_bias"] == pytest.approx(np.log10(2))
    assert shifted["log10_spatial_correlation"] == pytest.approx(1.)
    assert shifted["spatial_ratio_quantiles_05_50_95"] == pytest.approx([2., 2., 2.])
    assert shifted["fraction_pixels_within_10_percent"] == 0.
    identity = reference_field_agreement(reference, reference)
    assert identity["log10_rmse"] == 0. and identity["fraction_pixels_within_10_percent"] == 1.


def test_constant_reference_has_undefined_correlation_and_rejects_invalid_fields():
    reference = np.full((5, 6), 1e-10)
    result = reference_field_agreement(reference, reference)
    assert result["log10_spatial_correlation"] is None
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="same nonempty 2-D shape"):
        reference_field_agreement(reference, reference[:2])
    with pytest.raises(ValueError, match="strictly positive"):
        reference_field_agreement(reference, np.zeros_like(reference))


def _artifact(root, name, n_modes):
    reference = 10. ** (-10. + np.arange(36).reshape(6, 6) / 100.)
    field_map, _ = build_reference_field_maps(reference, cell_size_m=5., residual_std_log10_k=.05,
        length_scale_m=(10., 20.), n_modes=n_modes, energy_threshold=1.)
    path = root / name / "input.yaml"
    save_reference_field_input_model(path, field_map,
        source_metadata={"reference_run": "RUN_2", "permeability_convention": "physical"})
    return path, field_map


def test_standalone_subset_saves_exact_inputs_png_and_metrics_without_full_ensemble(tmp_path):
    path, field_map = _artifact(tmp_path, "artifact", 3)
    output = tmp_path / "previews"
    assert main(["--input-model", str(path), "--output-dir", str(output), "--method", "MC",
                 "--seed", "7", "--n-samples", "8", "--sample-indices", "1", "5", "--save-arrays"]) == 0
    report = json.loads((output / "manifest.json").read_text())
    expected_coordinates = np.random.default_rng(7).normal(size=(8, 3))
    assert len(report["samples"]) == 2
    for row in report["samples"]:
        expected = field_map.map_coordinates(expected_coordinates[row["sample_index"]])
        saved = np.load(output / row["generated_npy"], allow_pickle=False)
        np.testing.assert_array_equal(saved, expected)
        assert saved.dtype == np.float32
        assert (output / row["png"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert row["agreement"]["log10_rmse"] == pytest.approx(
            np.sqrt(np.mean((np.log10(expected.astype(float)) - np.log10(field_map.reference_permeability)) ** 2)))
    assert report["samples"][0]["shared_log10_limits"] == report["samples"][1]["shared_log10_limits"]
    assert len(list(output.glob("**/*sample-*.npy"))) == 2
    with pytest.raises(ValueError, match="nonempty"):
        main(["--input-model", str(path), "--output-dir", str(output), "--method", "MC", "--seed", "7", "--n-samples", "8"])


def _pilot(root):
    pilot = root / "pilot"
    pilot.mkdir()
    rows, artifacts = [], []
    # Small map uses the prefix of a 5-D scrambled design, not its own 2-D design.
    uniforms = qmc.Sobol(5, scramble=True, seed=13).random_base2(2)
    coordinates = ndtri(uniforms)
    context = {"artifacts": artifacts, "method": "RQMC", "seed": 13, "n_samples": 4}
    for sid, dimension, role in (("anchor", 2, "primary"), ("larger", 5, "KL representation sensitivity")):
        path, field_map = _artifact(root, sid, dimension)
        artifacts.append({"scenario_id": sid, "input_model": {"path": str(path), "sha256": sha256_file(path)}})
        rows.append({"scenario_id": sid, "dimension": dimension, "role": role, "config": {"reference_run": "RUN_2"}})
        directory = pilot / sid
        directory.mkdir()
        records = [{"sample_index": i, "physical_field_sha256": sha256(
            field_map.map_coordinates(xi[:dimension]).tobytes()).hexdigest()} for i, xi in enumerate(coordinates)]
        (directory / "sample_diagnostics.json").write_text(json.dumps(records))
    context_id = sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    (pilot / "run_context.json").write_text(json.dumps({"context_id": context_id, "context": context}))
    (pilot / "summary.json").write_text(json.dumps({"context_id": context_id, "scenarios": rows,
        "paired_anchor_reuse": {"coordinate_dimension": 5}}))
    (pilot / "qoi_truncation.json").write_text(json.dumps({"context_id": context_id, "scenario_ids": ["anchor", "larger"]}))
    for row in rows:
        path = pilot / row["scenario_id"] / "sample_diagnostics.json"
        records = json.loads(path.read_text())
        path.write_text(json.dumps([{**r, "context_id": context_id} for r in records]))
    return pilot


def test_pilot_replay_uses_largest_design_and_checks_queried_input_hash(tmp_path):
    pilot = _pilot(tmp_path)
    output = tmp_path / "plots"
    main(["--pilot-output", str(pilot), "--output-dir", str(output), "--reference-run", "RUN_2",
          "--include-kl", "--sample-indices", "1"])
    report = json.loads((output / "manifest.json").read_text())
    assert {r["scenario_id"] for r in report["samples"]} == {"anchor", "larger"}
    assert all(r["coordinate_dimension"] == 5 and r["pilot_input_hash_verified"] for r in report["samples"])
    assert report["samples"][0]["shared_log10_ratio_limit"] == report["samples"][1]["shared_log10_ratio_limit"]
    assert not list(output.glob("**/*.npy")) and not list(output.glob(".selected-fields-*"))
    path = pilot / "anchor" / "sample_diagnostics.json"
    records = json.loads(path.read_text())
    records[1]["physical_field_sha256"] = "0" * 64
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="does not match the queried pilot input"):
        main(["--pilot-output", str(pilot), "--output-dir", str(tmp_path / "bad-hash"), "--sample-indices", "1"])


def test_preview_selection_cannot_override_frozen_design_or_select_invalid_indices(tmp_path):
    pilot = _pilot(tmp_path)
    with pytest.raises(SystemExit):
        main(["--pilot-output", str(pilot), "--output-dir", str(tmp_path / "bad"), "--seed", "14"])
    path, _ = _artifact(tmp_path, "other", 2)
    source = {"scenario_id": "test", "input_model": str(path), "n_samples": 4, "seed": 1, "method": "MC"}
    for indices in ([4], [-1], [0, 0]):
        with pytest.raises(ValueError):
            export_reference_previews([source], tmp_path / "invalid", sample_indices=indices)
    assert not (tmp_path / "invalid").exists()
