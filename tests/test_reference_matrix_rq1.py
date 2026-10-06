"""Exercise artifact-only RQ1 templates and replayable matrix configuration."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments.reference_scenarios import run_matrix
from subsurface_uq.rq1.config import load_rq1_config
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model


def _rq1_template(tmp_path, grf):
    source = tmp_path / "rq1.yaml"
    source.write_text(yaml.safe_dump({
        "experiment": {"id": "matrix", "output_root": str(tmp_path / "unused")},
        "release25": {"repo": "release25", "cnn1_dir": "real_cnn1", "cnn2_dir": "real_cnn3",
                      "prepared_pki_dir": "prepared", "fixed_run_id": "RUN_FIXED", "random_k": False},
        "sampling": {"budgets": [2, 4], "comparison_budgets": [2], "repetitions": 1,
                     "main_seed": 1, "repetition_seeds": [2]},
        "qoi": {"receptors": [[0, 0]], "mean_anomaly_roi": None},
        "grf": grf,
    }), encoding="utf-8")
    return source


def _manual_grf():
    return {"mean_log10_k": -9.5, "std_log10_k": .3, "covariance_model": "matern32",
            "length_scale_y_m": 100., "length_scale_x_m": 100., "n_modes": 2,
            "observations": [[0, 0, 1e-10]]}


@pytest.mark.parametrize("template_grf", [{"input_model": None}, _manual_grf()])
def test_explicit_artifact_override_removes_template_grf_and_roundtrips(tmp_path, template_grf):
    template = _rq1_template(tmp_path, template_grf)
    artifact = tmp_path / "scenario" / "stochastic_input_model.yaml"
    config = load_rq1_config(template, input_model_override=artifact)
    assert config.input_model == artifact.resolve()
    assert config.observations == () and config.mean_log10_k is None
    payload = config.to_dict()
    assert payload["grf"] == {"input_model": str(artifact.resolve())}
    replay = tmp_path / "replay.yaml"
    replay.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert load_rq1_config(replay) == config


def test_direct_artifact_still_rejects_duplicate_stochastic_parameters(tmp_path):
    grf = {**_manual_grf(), "input_model": str(tmp_path / "input.yaml")}
    template = _rq1_template(tmp_path, grf)
    with pytest.raises(ValueError, match="remove duplicated manual keys"):
        load_rq1_config(template)
    explicit = load_rq1_config(template, input_model_override=tmp_path / "selected.yaml")
    assert explicit.to_dict()["grf"] == {"input_model": str((tmp_path / "selected.yaml").resolve())}


def test_legacy_manual_rq1_config_remains_replayable(tmp_path):
    config = load_rq1_config(_rq1_template(tmp_path, _manual_grf()))
    source = tmp_path / "legacy_replay.yaml"
    source.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    assert load_rq1_config(source) == config


def _matrix_config(tmp_path, *, training_support=False):
    reference = 10 ** (-9.5 + np.arange(36).reshape(6, 6) / 100.)
    np.save(tmp_path / "reference.npy", reference)
    source = tmp_path / "matrix.yaml"
    payload = {
        "base": {"reference_run": "RUN_1", "sigma_R": .05},
        "axes": {"truncation": [{"n_modes": 2}, {"n_modes": 4}]},
        "references": {"RUN_1": "reference.npy"}, "cell_size_m": 5.,
        "permeability_convention": "historical-training", "n_samples": 2,
    }
    if training_support:
        np.save(tmp_path / "training.npy", reference * 2.)
        payload["training_support"] = {"reference_npy": ["training.npy"], "box_size": 3,
                                       "skip_per_dir": 1, "feature_stride": 1, "max_patches": 8}
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return source


@pytest.mark.parametrize("template_grf", [{"input_model": None}, _manual_grf()])
def test_matrix_supplies_artifact_before_loading_rq1_template(tmp_path, monkeypatch, template_grf):
    template = _rq1_template(tmp_path, template_grf)
    calls = []
    def run_stub(config):
        config.output_root.mkdir(parents=True, exist_ok=True)
        replay = config.output_root / "config.yaml"
        replay.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
        restored = load_rq1_config(replay)
        assert restored == config
        assert config.fixed_run_id == "RUN_FIXED" and config.random_k is False
        assert config.input_model.is_file()
        assert config.observations == () and config.n_modes is None
        calls.append(config)
    monkeypatch.setattr("subsurface_uq.rq1.workflow.run_rq1", run_stub)
    report = run_matrix(_matrix_config(tmp_path), tmp_path / "out", rq1_config=template)
    assert len(calls) == len(report["scenarios"]) == 2
    assert {c.experiment_id for c in calls} == {row["scenario_id"] for row in report["scenarios"]}
    assert all(Path(row["rq1_output"]).is_dir() for row in report["scenarios"])


def test_matrix_records_explicit_training_support_not_nominal_range(tmp_path):
    report = run_matrix(_matrix_config(tmp_path, training_support=True), tmp_path / "out")
    nominal = np.load(tmp_path / "reference.npy")
    for row in report["scenarios"]:
        model = load_conditional_kl_input_model(row["input_model"])
        assert model.training_k_range == pytest.approx((nominal.min() * 2., nominal.max() * 2.))
        provenance = model.payload["training_reference"]["provenance"]
        assert provenance["kind"] == "explicit_training_fields"
        assert provenance["sources"][0]["path"] == str((tmp_path / "training.npy").resolve())
        assert len(provenance["sources"][0]["sha256"]) == 64


def test_matrix_rejects_training_shape_mismatch_before_writing(tmp_path):
    source = _matrix_config(tmp_path, training_support=True)
    np.save(tmp_path / "training.npy", np.full((4, 4), 1e-10))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="share the full field shape"):
        run_matrix(source, output)
    assert not output.exists()
