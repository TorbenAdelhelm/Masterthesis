import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments import reference_field_diagnostics, reference_kernel_study as study
from subsurface_uq.experiments.reference_scenarios import run_matrix
from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.scenarios import reference_scenario_from_config, scenario_matrix


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _config(tmp_path, *, full=False, **options):
    reference = (10 ** np.linspace(-10., -9.7, 64)).reshape(8, 8)
    np.save(tmp_path / "reference.npy", reference)
    config = {
        "base": {"reference_run": "RUN_2", "sigma_R": .05,
                 "n_modes": 3, "energy_threshold": None},
        "axes": {"covariance_family": ["gaussian", "matern32", "exponential"] if full else ["matern32"],
                 "parameter_sets": [{"label": label, "correlation_range_row_m": row,
                                     "correlation_range_col_m": col}
                                    for label, row, col in ([("short", 10., 15.), ("medium", 20., 30.),
                                                             ("long", 40., 60.)] if full else [("medium", 20., 30.)])]},
        "references": {"RUN_2": "reference.npy"}, "cell_size_m": 5.,
        "permeability_convention": "historical-training", "n_samples": 4, "seed": 33,
        "diagnostic_factor": 1, "storage": {"retain_generated_fields": False}, **options,
    }
    source = tmp_path / "matrix.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    ids = [s.scenario_id for s in scenario_matrix(reference_scenario_from_config(config["base"]), config["axes"])]
    return source, config, ids


@pytest.fixture(autouse=True)
def lightweight_diagnostic_figures(monkeypatch):
    # Keep the real numerical validator; replace only expensive PNG layout.
    def writer(report, output):
        result = {"report": "validation_report.md", "distribution": "distribution.png",
                  "correlation": "correlation.png", "spectra": "spectra.png", "law": "law.png"}
        for name in result.values():
            (output / name).write_bytes(b"fixed rendered diagnostic placeholder")
        return result
    monkeypatch.setattr(reference_field_diagnostics, "write_reference_validation_report", writer)


def test_prepare_complete_and_resume_cache_preserve_metrics_without_repeated_calls(tmp_path, monkeypatch):
    source, _, ids = _config(tmp_path, full=True)
    output = tmp_path / "study"
    calls = []
    validator = study.validate_reference_fields
    def track(*args, **kwargs):
        calls.append(args[0])
        return validator(*args, **kwargs)
    monkeypatch.setattr(study, "validate_reference_fields", track)
    prepared = study.run_study(output, config_path=source, prepare_only=True, probe_grid_size=2, max_lag_cells=3)
    assert not calls
    assert prepared["stage"] == "prepared_not_generated"
    assert len(prepared["scenarios"]) == 9
    assert all(row["input_status"] == "prepared_not_generated" for row in prepared["scenarios"])
    assert not (output / "diagnostics").exists()
    completed = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert completed["context_id"] == prepared["context_id"]
    assert len(calls) == 9
    assert completed["scenario_pooling"] is False and completed["scenario_weights"] is None
    assert completed["automatic_selection"] is None and completed["sample_filtering"] is None
    assert all(row["input_status"] == "diagnosed_all_fields" for row in completed["scenarios"])
    assert all(row["temperature_status"] == "not_evaluated" for row in completed["scenarios"])
    assert all(row["training_support_status"] == "unassessed_by_reference_comparison" for row in completed["scenarios"])
    resumed = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert len(calls) == 9
    assert resumed == completed
    with (output / "comparison.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["scenario_id"] for row in csv_rows} == set(ids)
    assert {row["setting"] for row in csv_rows} == {"short", "medium", "long"}
    assert all(float(row["mean_log10_rmse"]) > 0 for row in csv_rows)
    assert all(row["input_status"] == "diagnosed_all_fields" for row in csv_rows)
    assert (output / "comparison.md").is_file()
    assert (output / "comparison_RUN_2.png").is_file()


@pytest.mark.parametrize("changed", ["seed", "source_reference", "artifact", "companion", "cached_json"])
def test_resume_rejects_changed_design_inputs_and_cached_evidence(tmp_path, changed):
    source, _, _ = _config(tmp_path)
    output = tmp_path / "study"
    report = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    matrix = json.loads((output / "scenario_matrix/manifest.json").read_text(encoding="utf-8"))
    row = matrix["scenarios"][0]
    kwargs = {}
    if changed == "seed":
        kwargs["seed"] = 34
        expected = "study context changed"
    elif changed == "source_reference":
        np.save(tmp_path / "reference.npy", np.full((8, 8), 1e-10))
        expected = "source reference changed"
    elif changed == "artifact":
        artifact = Path(row["input_model"])
        artifact.write_text(artifact.read_text(encoding="utf-8") + "\n# altered\n", encoding="utf-8")
        expected = "manifest/artifact checksum mismatch"
    elif changed == "companion":
        artifact = Path(row["input_model"])
        payload = yaml.safe_load(artifact.read_text(encoding="utf-8"))
        companion = artifact.parent / payload["reference_field"]["path"]
        companion.write_bytes(companion.read_bytes() + b"changed")
        expected = "reference companion checksum mismatch"
    else:
        cached = output / "diagnostics" / report["scenarios"][0]["scenario_id"] / "validation.json"
        cached.write_text(cached.read_text(encoding="utf-8") + " ", encoding="utf-8")
        expected = "cached diagnostics changed"
    with pytest.raises(ValueError, match=expected):
        study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3, **kwargs)


def test_interrupted_render_does_not_publish_partial_diagnostics_and_retry_completes(tmp_path, monkeypatch):
    source, _, ids = _config(tmp_path)
    output = tmp_path / "study"
    study.run_study(output, config_path=source, prepare_only=True, probe_grid_size=2, max_lag_cells=3)
    validator = study.validate_reference_fields
    def interrupted(input_model, stage, **kwargs):
        stage.mkdir(parents=True)
        (stage / "partial.png").write_bytes(b"partial render")
        raise RuntimeError("interrupted PNG render")
    monkeypatch.setattr(study, "validate_reference_fields", interrupted)
    with pytest.raises(RuntimeError, match="interrupted PNG render"):
        study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert not (output / "diagnostics" / ids[0]).exists()
    assert not list(output.glob(".diagnostics-*"))
    monkeypatch.setattr(study, "validate_reference_fields", validator)
    report = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert report["stage"] == "input_diagnostics_completed"
    assert (output / "diagnostics" / ids[0] / "study_record.json").is_file()
    assert not (output / "diagnostics" / ids[0] / "partial.png").exists()


def test_existing_incomplete_final_diagnostic_directory_is_preserved(tmp_path):
    source, _, ids = _config(tmp_path)
    output = tmp_path / "study"
    study.run_study(output, config_path=source, prepare_only=True, probe_grid_size=2, max_lag_cells=3)
    incomplete = output / "diagnostics" / ids[0]
    incomplete.mkdir(parents=True)
    (incomplete / "previous.txt").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="incomplete diagnostic directory"):
        study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert list(incomplete.iterdir()) == [incomplete / "previous.txt"]
    assert (incomplete / "previous.txt").read_bytes() == b"preserve"


def test_existing_empty_final_diagnostic_directory_can_be_completed(tmp_path):
    source, _, ids = _config(tmp_path)
    output = tmp_path / "study"
    study.run_study(output, config_path=source, prepare_only=True, probe_grid_size=2, max_lag_cells=3)
    empty = output / "diagnostics" / ids[0]
    empty.mkdir(parents=True)
    report = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    assert report["stage"] == "input_diagnostics_completed"
    assert (empty / "study_record.json").is_file()


def test_explicit_subset_and_shortlist_are_separate_unweighted_assumptions(tmp_path):
    source, _, ids = _config(tmp_path, full=True, scenario_weights="equal")
    output = tmp_path / "selected"
    selected = [ids[0], ids[3], ids[6]]
    report = study.run_study(output, config_path=source, scenario_ids=selected,
                             shortlist_ids=selected[:2], probe_grid_size=2, max_lag_cells=3)
    assert {row["scenario_id"] for row in report["scenarios"]} == set(selected)
    assert len(list((output / "diagnostics").iterdir())) == 3
    assert report["scenario_pooling"] is False
    assert set(report["scenario_weights"]) == set(ids)
    shortlist = json.loads((output / "shortlist_manifest.json").read_text(encoding="utf-8"))
    assert {row["scenario_id"] for row in shortlist["scenarios"]} == set(selected[:2])
    assert shortlist["scenario_weights"] is None
    assert shortlist["source_scenario_weights"] == report["scenario_weights"]
    assert shortlist["selection"]["mode"] == "explicit_user_ids"
    assert all(Path(row["input_model"]).is_absolute() for row in shortlist["scenarios"])


def test_external_manifest_can_use_relative_artifact_paths(tmp_path):
    source, _, _ = _config(tmp_path)
    external = tmp_path / "external"
    report = run_matrix(source, external, prepare_only=True)
    for row in report["scenarios"]:
        row["input_model"] = str(Path(row["input_model"]).relative_to(external))
    manifest = external / "manifest.json"
    _write_json(manifest, report)
    result = study.run_study(tmp_path / "review", matrix_manifest=manifest,
                             probe_grid_size=2, max_lag_cells=3)
    assert len(result["scenarios"]) == 1
    assert result["scenarios"][0]["input_status"] == "diagnosed_all_fields"
    assert result["sampling"]["n_samples"] == 4 and result["sampling"]["seed"] == 33
    assert result["scenarios"][0]["support"]["diagnostic_factor"] == 1


def test_temperature_context_join_passes_full_artifact_provenance_and_preserves_separate_outputs(tmp_path, monkeypatch):
    source, _, _ = _config(tmp_path)
    output = tmp_path / "study"
    before = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    artifacts = json.loads((output / "study_context.json").read_text(encoding="utf-8"))["context"]["artifacts"]
    summary = {"scenarios": [{"scenario_id": artifacts[0]["scenario_id"], "config": artifacts[0]["config"],
                              "qoi_mean": [11.], "qoi_std": [.2]}],
               "nominal_references": [{"reference_run": "RUN_2", "qoi_values": [11.1]}],
               "qoi_names": ["temperature_receptor"]}
    summary_path = tmp_path / "temperature_summary.json"
    _write_json(summary_path, summary)
    calls = []
    # The dedicated attachment tests exercise persisted prediction/QoI caches
    # and numerical tamper rejection. Here verify the batch orchestration sends
    # full scientific provenance and retains the validated result per scenario.
    def validated_attachment(path, actual_artifacts):
        assert Path(path) == summary_path
        assert actual_artifacts == artifacts
        for artifact in actual_artifacts:
            for kind in ("input_model", "reference_companion"):
                assert artifact[kind]["sha256"] == sha256_file(artifact[kind]["path"])
        calls.append(path)
        return summary, {"fixed_run_id": "RUN_2", "summary": {"sha256": sha256_file(path)}}
    monkeypatch.setattr(study, "_temperature_attachment", validated_attachment)
    report = study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3,
                             temperature_summary=summary_path)
    assert report["context_id"] == before["context_id"]
    assert report["scenarios"][0]["temperature_status"] == "frozen_model_evaluated"
    assert report["scenarios"][0]["temperature"] == summary["scenarios"][0]
    assert report["nominal_references"] == summary["nominal_references"]
    assert report["temperature_attachment"]["fixed_run_id"] == "RUN_2"
    assert report["temperature_attachment"]["summary"]["sha256"] == sha256_file(summary_path)
    assert report["scenario_pooling"] is False and report["automatic_selection"] is None
    assert calls == [summary_path]


def test_temperature_join_validation_failure_does_not_replace_existing_comparison(tmp_path, monkeypatch):
    source, _, _ = _config(tmp_path)
    output = tmp_path / "study"
    study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3)
    before = (output / "comparison.json").read_bytes()
    def rejected(*args):
        raise ValueError("temperature scenario provenance disagrees")
    monkeypatch.setattr(study, "_temperature_attachment", rejected)
    with pytest.raises(ValueError, match="provenance disagrees"):
        study.run_study(output, config_path=source, probe_grid_size=2, max_lag_cells=3,
                         temperature_summary=tmp_path / "invalid_summary.json")
    assert (output / "comparison.json").read_bytes() == before


def test_rqmc_dimension_limit_is_checked_before_diagnostics(tmp_path, monkeypatch):
    source, _, _ = _config(tmp_path)
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported RQMC representation must not draw fields")
    monkeypatch.setattr(study, "validate_reference_fields", unexpected)
    monkeypatch.setattr(study.qmc, "Sobol", SimpleNamespace(MAXDIM=2))
    with pytest.raises(ValueError, match="exceeds scrambled Sobol limit"):
        study.run_study(tmp_path / "study", config_path=source, method="RQMC", prepare_only=True)
    assert not (tmp_path / "study/diagnostics").exists()


@pytest.mark.parametrize("kwargs", [
    {"n_samples": True}, {"n_samples": 1}, {"seed": -1}, {"seed": True},
    {"method": "bad"}, {"method": "RQMC", "n_samples": 3},
    {"diagnostic_factor": 0}, {"probe_grid_size": 17}, {"max_lag_cells": 0},
    {"prepare_only": True, "shortlist_ids": ["ref-placeholder"]},
    {"prepare_only": True, "temperature_summary": "placeholder.json"},
])
def test_invalid_parameters_reject_before_creating_study_outputs(tmp_path, kwargs):
    source, _, _ = _config(tmp_path)
    with pytest.raises(ValueError):
        study.run_study(tmp_path / "study", config_path=source, **kwargs)
    assert not (tmp_path / "study").exists()


@pytest.mark.parametrize("kwargs", [{}, {"config_path": "x", "matrix_manifest": "y"}])
def test_exactly_one_source_is_required(tmp_path, kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        study.run_study(tmp_path / "study", **kwargs)


@pytest.mark.parametrize("contents", ["", "[]", "scalar"])
def test_config_must_be_a_mapping(tmp_path, contents):
    source = tmp_path / "invalid.yaml"
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        study.run_study(tmp_path / "study", config_path=source)
    assert not (tmp_path / "study").exists()


@pytest.mark.parametrize("selection", ["unknown", "duplicate", "shortlist_unknown", "shortlist_duplicate"])
def test_invalid_selection_and_shortlist_ids_are_rejected(tmp_path, selection):
    source, _, ids = _config(tmp_path)
    kwargs = {
        "unknown": {"scenario_ids": ["ref-unknown"]},
        "duplicate": {"scenario_ids": [ids[0], ids[0]]},
        "shortlist_unknown": {"shortlist_ids": ["ref-unknown"]},
        "shortlist_duplicate": {"shortlist_ids": [ids[0], ids[0]]},
    }[selection]
    with pytest.raises(ValueError, match="distinct"):
        study.run_study(tmp_path / "study", config_path=source, **kwargs)


def test_nonempty_directory_without_context_is_preserved(tmp_path):
    source, _, _ = _config(tmp_path)
    output = tmp_path / "study"
    output.mkdir()
    (output / "existing.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty study directory has no context"):
        study.run_study(output, config_path=source)
    assert (output / "existing.txt").read_text(encoding="utf-8") == "preserve"
    assert list(output.iterdir()) == [output / "existing.txt"]


def test_cli_help_exposes_staged_kernel_study(capsys):
    with pytest.raises(SystemExit) as exc:
        study.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--config", "--matrix-manifest", "--prepare-only", "--shortlist-ids", "--temperature-summary"):
        assert flag in help_text


def test_external_manifest_cannot_relabel_different_arrays_as_the_same_run(tmp_path):
    source, config, _ = _config(tmp_path)
    first = run_matrix(source, tmp_path / "first", prepare_only=True)
    np.save(tmp_path / "reference.npy", np.load(tmp_path / "reference.npy") * 1.1)
    config["axes"]["covariance_family"] = ["gaussian"]
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    second = run_matrix(source, tmp_path / "second", prepare_only=True)
    combined = tmp_path / "combined.json"
    _write_json(combined, {**first, "scenarios": first["scenarios"] + second["scenarios"]})
    with pytest.raises(ValueError, match="one reference RUN"):
        study.run_study(tmp_path / "study", matrix_manifest=combined)
    assert not (tmp_path / "study/diagnostics").exists()
