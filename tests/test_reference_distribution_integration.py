import json

import numpy as np
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
        "base": {"reference_run": "RUN_1", "sigma_R": 0.05},
        "axes": {
            "ell_row_m": [20.0],
            "ell_col_m": [30.0],
            "truncation": [{"n_modes": 3}],
        },
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
