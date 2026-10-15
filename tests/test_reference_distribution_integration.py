import json

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments.reference_scenarios import run_matrix
from subsurface_uq.sampling.hydraulic_conductivity import (
    hydraulic_conductivity_to_permeability,
)
from subsurface_uq.sampling.scenarios import (
    ReferenceScenario,
    reference_scenario_from_config,
    scenario_matrix,
)


def test_row_column_length_aliases_preserve_legacy_mapping():
    scenario = reference_scenario_from_config({
        "reference_run": "RUN_1",
        "sigma_R": 0.05,
        "ell_row_m": 40.0,
        "ell_col_m": 70.0,
    })
    assert scenario.ell_y == 40.0
    assert scenario.ell_x == 70.0
    assert scenario.ell_row_m == 40.0
    assert scenario.ell_col_m == 70.0
    assert scenario.manifest()["correlation_lengths_m"]["axis_interpretation"].startswith("array axes")

    rows = scenario_matrix(
        ReferenceScenario("RUN_1", sigma_R=0.05),
        {"ell_row_m": [30.0], "ell_col_m": [60.0]},
    )
    assert len(rows) == 1
    assert rows[0].ell_y == 30.0 and rows[0].ell_x == 60.0


def test_matrix_writes_hydraulic_distribution_validation_without_filtering(tmp_path):
    # Constant median near the documented target center in hydraulic-conductivity
    # space. The real generator still receives intrinsic permeability in m2.
    kh_reference = np.full((8, 8), 1.0e-3)
    permeability = hydraulic_conductivity_to_permeability(
        kh_reference, convention="historical-training"
    ).astype(np.float32)
    np.save(tmp_path / "run.npy", permeability)

    config = {
        "base": {
            "reference_run": "RUN_1",
            "sigma_R": 0.05,
            "ell_row_m": 20.0,
            "ell_col_m": 30.0,
        },
        "axes": {"truncation": [{"n_modes": 3}]},
        "references": {"RUN_1": "run.npy"},
        "cell_size_m": 5.0,
        "permeability_convention": "historical-training",
        "distribution_target": {
            "log10_mean": -3.0,
            "log10_std": 0.5,
            "interval_m_s": [1.0e-4, 5.0e-2],
            "minimum_interval_fraction": 0.95,
        },
        "distribution_validation": {
            "max_abs_log10_mean_error": 0.3,
            "max_abs_log10_std_error": 0.6,
            "max_qq_rmse_standardized": 2.0,
        },
        "n_samples": 4,
        "seed": 4,
        "diagnostic_factor": 1,
        "storage": {"retain_generated_fields": False},
    }
    source = tmp_path / "matrix.yaml"
    source.write_text(yaml.safe_dump(config), encoding="utf-8")
    report = run_matrix(source, tmp_path / "out")
    row = report["scenarios"][0]
    assert row["correlation_lengths_m"]["row"] == 20.0
    assert row["correlation_lengths_m"]["column"] == 30.0
    validation_path = tmp_path / "out" / row["scenario_id"] / "distribution_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert validation["no_filtering"] is True
    assert validation["sample_filtering"] is None
    assert validation["hydraulic_conversion"]["convention"] == "historical-training"
    assert validation["target"]["log10_std"] == 0.5
    assert validation["target"]["log10_variance"] == 0.25
    assert validation["field_count"] == 4
    assert 0.0 <= validation["assumption_coverage"] <= 1.0
    assert "log10_sigma_error" in validation
    assert "qq_rmse_standardized" in validation["normality_deviation"]
    assert "spatial_support_deviation" in validation
    assert not (validation_path.parent / "generated_fields.npy").exists()


def _small_matrix(tmp_path, **options):
    reference = (10 ** np.linspace(-10, -9.8, 64)).reshape(8, 8)
    np.save(tmp_path / "reference.npy", reference)
    return {
        "base": {"reference_run": "RUN_2", "sigma_R": .05,
                 "ell_row_m": 20, "ell_col_m": 30, "n_modes": 3, "energy_threshold": None},
        "references": {"RUN_2": "reference.npy"}, "cell_size_m": 5,
        "permeability_convention": "historical-training", "n_samples": 4, "seed": 33,
        "storage": {"retain_generated_fields": True}, **options,
    }


@pytest.mark.parametrize("target_config", [None, {}])
def test_global_target_is_opt_in_and_does_not_change_generated_inputs(tmp_path, target_config):
    outputs = []
    for label, target in (("off", None), ("chosen", target_config)):
        config = _small_matrix(tmp_path, distribution_target=target)
        source = tmp_path / f"{label}.yaml"
        source.write_text(yaml.safe_dump(config))
        report = run_matrix(source, tmp_path / label)
        row = report["scenarios"][0]
        directory = tmp_path / label / row["scenario_id"]
        validation = json.loads((directory / "distribution_validation.json").read_text())
        assert (report["distribution_target"] is None) == (target is None)
        assert (validation["normality_deviation"] is None) == (target is None)
        assert validation["pooled"]["log10_std"] > 0
        assert validation["sample_filtering"] is None
        outputs.append(np.load(directory / "generated_fields.npy", allow_pickle=False))
    np.testing.assert_array_equal(*outputs)


def test_matrix_partial_thresholds_write_json_and_require_explicit_target(tmp_path):
    config = _small_matrix(tmp_path, distribution_target={},
                           distribution_validation={"max_abs_log10_mean_error": .25})
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config))
    report = run_matrix(source, tmp_path / "valid")
    row = report["scenarios"][0]
    validation = json.loads((tmp_path / "valid" / row["scenario_id"] / "distribution_validation.json").read_text())
    assert validation["pooled"]["heuristic_assessment"]["tolerances"]["max_abs_log10_std_error"] is None
    config["distribution_target"] = None
    source.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="explicit distribution target"):
        run_matrix(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
