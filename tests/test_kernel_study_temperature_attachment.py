"""Attach saved frozen-screen predictions, not unverifiable summary numbers."""
from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from subsurface_uq.experiments.reference_kernel_study import _canonical, _temperature_attachment
from subsurface_uq.io import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _moments(values):
    return {"sample_count": len(values), "mean": values.mean(axis=0).tolist(),
            "std_ddof1": values.std(axis=0, ddof=1).tolist(),
            "quantiles_05_50_95": np.quantile(values, [.05, .5, .95], axis=0).tolist()}


@pytest.fixture
def saved_screen(tmp_path):
    root = tmp_path / "screen"
    root.mkdir()
    artifact = {"scenario_id": "ref-one", "config": {"reference_run": "RUN_2", "sigma_R": .05},
                "dimension": 2, "input_model": {"path": "input.yaml", "sha256": "a" * 64},
                "reference_companion": {"path": "reference.npy", "sha256": "b" * 64}}
    context = {"study_mode": "kernel-screen", "artifacts": [artifact], "method": "RQMC",
               "seed": 5901, "n_samples": 4, "qoi_names": ["temperature_0_0", "temperature_0_1"],
               "rq1_config_resolved": {"release25": {"fixed_run_id": "RUN_2"}}}
    context_id = sha256(_canonical(context).encode()).hexdigest()
    _write(root / "run_context.json", {"context_id": context_id, "context": context})
    cache = root / "sample_cache" / context_id
    cache.mkdir(parents=True)

    def prediction(label, values):
        digest = sha256(label.encode()).hexdigest()
        temperature = np.asarray([values, values], dtype=np.float32)
        np.save(cache / f"{digest}.temperature.npy", temperature)
        record = {"context_id": context_id, "physical_field_sha256": digest,
                  "physical_field_shape": [8, 8], "physical_field_dtype": "float32",
                  "temperature_grid_shape": [2, 2], "temperature_sha256": sha256_file(cache / f"{digest}.temperature.npy"),
                  "qoi": list(values), "prediction_seconds": .01,
                  "temperature_descriptors": {"mean_anomaly_C": float(np.mean(values) - 10.)},
                  "flow_descriptors": {"speed_max": .05},
                  "normalization_range_diagnostic": {"fraction_outside": 0.}}
        _write(cache / f"{digest}.json", {**record, "result_sha256": sha256(_canonical(record).encode()).hexdigest()})
        return record, temperature

    baseline_values = np.array([11., 12.])
    baseline_record, baseline = prediction("nominal", baseline_values.tolist())
    nominal = root / "nominal_references/RUN_2"
    nominal.mkdir(parents=True)
    np.save(nominal / "temperature.npy", baseline)
    values = np.array([[11., 12.], [11.2, 12.3], [10.9, 11.9], [11.1, 12.1]])
    diagnostics = [{"sample_index": index, **prediction(f"field-{index}", row.tolist())[0]}
                   for index, row in enumerate(values)]
    scenario = root / "ref-one"
    scenario.mkdir()
    np.save(scenario / "qoi_samples.npy", values)
    _write(scenario / "sample_diagnostics.json", diagnostics)
    row = {"scenario_id": "ref-one", "config": artifact["config"], "dimension": 2,
           "qoi_samples": str(scenario / "qoi_samples.npy"), "temperature_qoi": _moments(values),
           "nominal_reference_qoi": baseline_values.tolist(), "qoi_delta_to_nominal": _moments(values - baseline_values),
           "temperature_descriptors": [r["temperature_descriptors"] for r in diagnostics],
           "flow_descriptors": [r["flow_descriptors"] for r in diagnostics],
           "normalization_range_diagnostics": [r["normalization_range_diagnostic"] for r in diagnostics]}
    summary = {"study_mode": "kernel-screen", "context_id": context_id, "fixed_run_id": "RUN_2",
               "method": "RQMC", "seed": 5901, "n_samples_per_scenario": 4,
               "qoi_names": context["qoi_names"], "fixed_context": "RUN_2 pressure and sources held fixed",
               "training_support_status": "unassessed; output plausibility does not establish membership",
               "nominal_reference_count": 1,
               "nominal_references": {"RUN_2": {"reference_run": "RUN_2", "source": artifact["reference_companion"],
                   "qoi": baseline_values.tolist(), "diagnostics": baseline_record}},
               "scenarios": [row]}
    path = root / "summary.json"
    _write(path, summary)
    return path, summary, [artifact], cache


def test_attachment_verifies_saved_outputs_and_keeps_separate_temperature_design(saved_screen):
    path, summary, artifacts, cache = saved_screen
    actual, metadata = _temperature_attachment(path, artifacts)
    assert actual == summary
    assert metadata["sampling"] == {"method": "RQMC", "seed": 5901, "n_samples_per_scenario": 4,
                                    "qoi_names": ["temperature_0_0", "temperature_0_1"]}
    assert "different samples, seeds, budgets or integration methods" in metadata["design_association"]
    assert "no sample-wise pairing" in metadata["design_association"]
    assert len(metadata["verified_output_files"]) == 13
    assert all(sha256_file(item["path"]) == item["sha256"] for item in metadata["verified_output_files"])


@pytest.mark.parametrize("key,value", [("method", "MC"), ("seed", 4901), ("n_samples_per_scenario", 32),
                                      ("fixed_run_id", "RUN_1"), ("qoi_names", ["different_qoi"])])
def test_attachment_rejects_summary_design_claims_not_bound_by_context(saved_screen, key, value):
    path, summary, artifacts, _ = saved_screen
    summary[key] = value
    _write(path, summary)
    with pytest.raises(ValueError, match="sampling/QoI/fixed context"):
        _temperature_attachment(path, artifacts)


@pytest.mark.parametrize("moment", ["mean", "std_ddof1", "quantiles_05_50_95"])
@pytest.mark.parametrize("group", ["temperature_qoi", "qoi_delta_to_nominal"])
def test_attachment_rejects_modified_numerical_moments(saved_screen, moment, group):
    path, summary, artifacts, _ = saved_screen
    values = summary["scenarios"][0][group][moment]
    if moment == "quantiles_05_50_95":
        values[0][0] += .1
    else:
        values[0] += .1
    _write(path, summary)
    with pytest.raises(ValueError, match="moments disagree with saved QoI samples"):
        _temperature_attachment(path, artifacts)


@pytest.mark.parametrize("descriptor", ["temperature_descriptors", "flow_descriptors", "normalization_range_diagnostics"])
def test_attachment_rejects_modified_per_prediction_descriptors(saved_screen, descriptor):
    path, summary, artifacts, _ = saved_screen
    summary["scenarios"][0][descriptor][0] = {"changed": 123.}
    _write(path, summary)
    with pytest.raises(ValueError, match="descriptors disagree"):
        _temperature_attachment(path, artifacts)


def test_attachment_rejects_changed_qoi_array_even_with_unchanged_summary(saved_screen):
    path, _, artifacts, _ = saved_screen
    qoi_path = path.parent / "ref-one/qoi_samples.npy"
    values = np.load(qoi_path)
    values[0, 0] += 1.
    np.save(qoi_path, values)
    with pytest.raises(ValueError, match="saved QoI samples disagree"):
        _temperature_attachment(path, artifacts)


def test_attachment_rejects_changed_diagnostics_even_if_summary_is_adjusted(saved_screen):
    path, summary, artifacts, _ = saved_screen
    diagnostic_path = path.parent / "ref-one/sample_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic[0]["qoi"][0] += 1.
    _write(diagnostic_path, diagnostic)
    values = np.asarray([row["qoi"] for row in diagnostic])
    np.save(path.parent / "ref-one/qoi_samples.npy", values)
    summary["scenarios"][0]["temperature_qoi"] = _moments(values)
    _write(path, summary)
    with pytest.raises(ValueError, match="checksummed prediction cache"):
        _temperature_attachment(path, artifacts)


def test_attachment_rejects_changed_prediction_cache_checksum(saved_screen):
    path, _, artifacts, cache = saved_screen
    cache_path = next(cache.glob("*.json"))
    record = json.loads(cache_path.read_text())
    record["qoi"][0] += 1.
    _write(cache_path, record)
    with pytest.raises(ValueError, match="checksummed prediction cache"):
        _temperature_attachment(path, artifacts)


def test_attachment_rejects_changed_temperature_array(saved_screen):
    path, _, artifacts, cache = saved_screen
    array_path = next(cache.glob("*.temperature.npy"))
    np.save(array_path, np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="temperature array disagrees"):
        _temperature_attachment(path, artifacts)


@pytest.mark.parametrize("change", ["values", "source", "scenario_delta"])
def test_attachment_rejects_changed_nominal_reference_values_or_source(saved_screen, change):
    path, summary, artifacts, _ = saved_screen
    if change == "values":
        summary["nominal_references"]["RUN_2"]["qoi"][0] += .1
    elif change == "source":
        summary["nominal_references"]["RUN_2"]["source"] = {"sha256": "c" * 64}
    else:
        summary["scenarios"][0]["nominal_reference_qoi"][0] += .1
    _write(path, summary)
    with pytest.raises(ValueError, match="nominal"):
        _temperature_attachment(path, artifacts)


def test_attachment_requires_every_completed_context_scenario(saved_screen):
    path, summary, artifacts, _ = saved_screen
    summary["scenarios"] = []
    _write(path, summary)
    with pytest.raises(ValueError, match="distinct scenarios"):
        _temperature_attachment(path, artifacts)


@pytest.mark.parametrize("kind", ["artifact", "reference", "config", "summary_context", "context_checksum"])
def test_attachment_retains_original_artifact_and_context_provenance_guards(saved_screen, kind):
    path, summary, artifacts, _ = saved_screen
    artifacts = deepcopy(artifacts)
    if kind == "artifact":
        artifacts[0]["input_model"]["sha256"] = "c" * 64
    elif kind == "reference":
        artifacts[0]["reference_companion"]["sha256"] = "c" * 64
    elif kind == "config":
        artifacts[0]["config"]["sigma_R"] = .2
    elif kind == "summary_context":
        summary["context_id"] = "c" * 64
        _write(path, summary)
    else:
        context_path = path.parent / "run_context.json"
        record = json.loads(context_path.read_text())
        record["context"]["seed"] += 1
        _write(context_path, record)
    with pytest.raises(ValueError, match="provenance disagrees|checksum or study-mode mismatch"):
        _temperature_attachment(path, artifacts)
