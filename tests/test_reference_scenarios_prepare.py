import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments import reference_scenarios
from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.scenarios import reference_scenario_from_config, scenario_matrix


def _config(tmp_path, *, kernels=("gaussian", "matern32", "exponential")):
    reference = (10 ** np.linspace(-10., -9.8, 64)).reshape(8, 8)
    np.save(tmp_path / "reference.npy", reference)
    config = {
        "base": {"reference_run": "RUN_2", "sigma_R": .05,
                 "n_modes": 3, "energy_threshold": None},
        "axes": {"covariance_family": list(kernels), "parameter_sets": [
            {"label": "paired", "correlation_range_row_m": 20., "correlation_range_col_m": 30.}]},
        "references": {"RUN_2": "reference.npy"}, "cell_size_m": 5.,
        "permeability_convention": "historical-training", "n_samples": 4, "seed": 33,
        "diagnostic_factor": 1, "storage": {"retain_generated_fields": True},
    }
    source = tmp_path / "matrix.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    return source, config, reference


def _files(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_prepare_only_writes_portable_laws_and_representation_without_draws(tmp_path, monkeypatch):
    source, _, reference = _config(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("prepare-only must not draw fields or compute ensemble diagnostics")

    monkeypatch.setattr(reference_scenarios, "GaussianCoordinatePermeabilitySampler", unexpected)
    monkeypatch.setattr(reference_scenarios, "compare_reference_ensemble", unexpected)
    output = tmp_path / "prepared"
    report = reference_scenarios.run_matrix(source, output, prepare_only=True)
    assert report["execution"] == "artifacts_only"
    assert report["pooling"] is False and report["scenario_weights"] is None
    assert len(report["scenarios"]) == 3
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == report
    for row in report["scenarios"]:
        directory = output / row["scenario_id"]
        assert row["execution"] == "artifacts_only"
        assert row["dimension"] == 3
        assert 0 < row["retained_energy"] <= 1
        assert row["parameter_set"]["label"] == "paired"
        assert row["parameter_set"] == report["parameter_sets"][row["scenario_id"]]
        assert row["storage"] == {"retain_generated_fields": False, "generated_fields": None}
        assert not (directory / "generated_fields.npy").exists()
        assert not (directory / "diagnostics.json").exists()
        assert not (directory / "distribution_validation.json").exists()
        artifact = Path(row["input_model"])
        assert sha256_file(artifact) == row["input_model_sha256"]
        # A copied scenario directory reconstructs without the original YAML's
        # reference path. The NPZ is relative to the portable input artifact.
        portable = tmp_path / "copied" / row["scenario_id"]
        shutil.copytree(directory, portable)
        model = load_conditional_kl_input_model(portable / artifact.name)
        np.testing.assert_array_equal(model.unconditional.reference_permeability, reference)
        assert model.conditional.dimension == row["dimension"]
        variance = model.prior.pointwise_log10_variance()
        assert row["retained_log10_variance"] == pytest.approx({
            "min": float(variance.min()), "mean": float(variance.mean()), "max": float(variance.max())})
        assert row["source_metadata"]["reference_npy_sha256"] == sha256_file(tmp_path / "reference.npy")


def test_normal_execution_after_preparation_preserves_generated_input_contract(tmp_path):
    source, _, _ = _config(tmp_path, kernels=("gaussian",))
    report = reference_scenarios.run_matrix(source, tmp_path / "prepared", prepare_only=True)
    prepared = load_conditional_kl_input_model(report["scenarios"][0]["input_model"])
    # Complete the same prepared matrix through the unchanged normal path.
    completed = reference_scenarios.run_matrix(source, tmp_path / "prepared")
    assert completed["execution"] == "generated_input_ensembles"
    row = completed["scenarios"][0]
    assert row["scenario_id"] == report["scenarios"][0]["scenario_id"]
    assert row["scenario_instance_id"] == report["scenarios"][0]["scenario_instance_id"]
    assert row["storage"]["retain_generated_fields"] is True
    fields = np.load(row["storage"]["generated_fields"], allow_pickle=False)
    assert fields.shape == (4, 8, 8)
    coordinates = np.random.default_rng(33).normal(size=(4, prepared.conditional.dimension))
    expected = np.stack([prepared.conditional.map_coordinates(eta) for eta in coordinates]).astype(np.float32)
    np.testing.assert_array_equal(fields, expected)
    directory = Path(row["input_model"]).parent
    diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["compatibility_assessment"]["status"] == "unassessed_training_support"
    marginal = json.loads((directory / "distribution_validation.json").read_text(encoding="utf-8"))
    assert marginal["field_count"] == 4 and marginal["sample_filtering"] is None


def test_prepare_only_cannot_overwrite_existing_prepared_or_generated_artifacts(tmp_path):
    source, _, _ = _config(tmp_path, kernels=("exponential",))
    for label, preparation in (("prepared", True), ("generated", False)):
        output = tmp_path / label
        reference_scenarios.run_matrix(source, output, prepare_only=preparation)
        before = _files(output)
        with pytest.raises(ValueError, match="fresh scenario directories"):
            reference_scenarios.run_matrix(source, output, prepare_only=True)
        assert _files(output) == before


def test_prepare_only_does_not_overwrite_orphaned_nonempty_scenario_directory(tmp_path):
    source, config, _ = _config(tmp_path, kernels=("matern32",))
    scenario = scenario_matrix(reference_scenario_from_config(config["base"]), config["axes"])[0]
    output = tmp_path / "orphaned"
    directory = output / scenario.scenario_id
    directory.mkdir(parents=True)
    (directory / "existing.txt").write_text("preserve", encoding="utf-8")
    before = _files(output)
    with pytest.raises(ValueError, match="non-empty but has no manifest"):
        reference_scenarios.run_matrix(source, output, prepare_only=True)
    assert _files(output) == before


def test_prepare_only_rejects_rq1_propagation_before_reading_or_writing(tmp_path):
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="cannot request RQ1 propagation"):
        reference_scenarios.run_matrix(tmp_path / "absent.yaml", output,
                                       rq1_config=tmp_path / "absent-rq1.yaml", prepare_only=True)
    assert not output.exists()


@pytest.mark.parametrize("value", [1, "true", None])
def test_prepare_only_requires_boolean_value(tmp_path, value):
    with pytest.raises(ValueError, match="must be boolean"):
        reference_scenarios.run_matrix(tmp_path / "absent.yaml", tmp_path / "not-created", prepare_only=value)
    assert not (tmp_path / "not-created").exists()
