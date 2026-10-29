from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from subsurface_uq.sampling.scenarios import (
    ReferenceScenario,
    correlation_range_to_length_scale,
    reference_scenario_from_config,
    scenario_matrix,
    scenario_matrix_with_metadata,
)


@pytest.mark.parametrize("family", ["gaussian", "matern32", "exponential"])
def test_effective_range_matches_one_over_e_for_each_kernel(family):
    distance = 500.
    length = correlation_range_to_length_scale(distance, family)
    h = distance / length
    correlation = {
        "gaussian": lambda: np.exp(-h**2 / 2),
        "exponential": lambda: np.exp(-h),
        "matern32": lambda: (1 + np.sqrt(3) * h) * np.exp(-np.sqrt(3) * h),
    }[family]()
    assert correlation == pytest.approx(np.exp(-1), rel=1e-14)
    assert correlation_range_to_length_scale(2 * distance, family) == 2 * length


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "500"])
def test_effective_range_rejects_nonpositive_or_non_numeric(value):
    with pytest.raises(ValueError, match="positive and finite"):
        correlation_range_to_length_scale(value, "gaussian")


def test_effective_range_rejects_unknown_kernel():
    with pytest.raises(ValueError, match="unsupported"):
        correlation_range_to_length_scale(500., "unknown")


def test_coupled_ranges_expand_to_27_scenarios_and_keep_labels_out_of_ids():
    base = ReferenceScenario("RUN_1", sigma_R=.05)
    axes = {
        "reference_run": ["RUN_1", "RUN_2", "RUN_3"],
        "covariance_family": ["gaussian", "matern32", "exponential"],
        "parameter_sets": [
            {"label": name, "correlation_range_row_m": row, "correlation_range_col_m": col}
            for name, row, col in [("short", 250., 400.), ("medium", 500., 800.), ("long", 1000., 1600.)]
        ],
    }
    scenarios, metadata = scenario_matrix_with_metadata(base, axes)
    assert len(scenarios) == 27
    assert len({s.scenario_id for s in scenarios}) == 27
    assert len(metadata) == 27
    for scenario in scenarios:
        info = metadata[scenario.scenario_id]
        declared = info["declared_parameters"]
        assert scenario.ell_row_m == correlation_range_to_length_scale(
            declared["correlation_range_row_m"], scenario.covariance_family)
        assert scenario.ell_col_m == correlation_range_to_length_scale(
            declared["correlation_range_col_m"], scenario.covariance_family)
        assert "label" not in scenario.manifest()["config"]
        assert info["resolved_parameters"]["covariance_family"] == scenario.covariance_family
    renamed = dict(axes)
    renamed["parameter_sets"] = [dict(item, label="renamed-" + item["label"]) for item in axes["parameter_sets"]]
    assert [s.scenario_id for s in scenario_matrix(base, renamed)] == [s.scenario_id for s in scenarios]
    reversed_axes = dict(reversed(list(axes.items())))
    assert {s.scenario_id for s in scenario_matrix(base, reversed_axes)} == {s.scenario_id for s in scenarios}


def test_coupled_amplitude_lengths_and_truncation_do_not_cross_product():
    base = ReferenceScenario("RUN_2", sigma_R=.05)
    axes = {"parameter_sets": [
        {"label": "small", "sigma_R": .025, "ell_row_m": 50., "ell_col_m": 80.,
         "truncation": {"n_modes": 3}},
        {"label": "large", "sigma_R": .1, "ell_row_m": 100., "ell_col_m": 160.,
         "truncation": {"energy_threshold": .99}},
    ]}
    scenarios = scenario_matrix(base, axes)
    assert len(scenarios) == 2
    assert [(s.sigma_R, s.ell_row_m, s.ell_col_m) for s in scenarios] == [(.025, 50., 80.), (.1, 100., 160.)]
    assert scenarios[0].n_modes == 3 and scenarios[0].energy_threshold is None
    assert scenarios[1].n_modes is None and scenarios[1].energy_threshold == .99
    independent_sigma = scenario_matrix(base, {"sigma_R": [.025, .1], "parameter_sets": [
        {"label": "range", "correlation_range_row_m": 100.}]})
    assert len(independent_sigma) == 2


@pytest.mark.parametrize("entry,axis", [
    ({"ell_row_m": 50.}, {"ell_y": [50.]}),
    ({"correlation_range_row_m": 50.}, {"ell_row_m": [50.]}),
    ({"ell_x": 80.}, {"ell_col_m": [80.]}),
    ({"sigma_R": .05}, {"sigma_R": [.05]}),
    ({"truncation": {"n_modes": 3}}, {"truncation": [{"n_modes": 3}]}),
])
def test_parameter_sets_reject_conflicting_independent_axes(entry, axis):
    with pytest.raises(ValueError, match="conflict with independent"):
        scenario_matrix(ReferenceScenario("RUN_1", sigma_R=.05), {
            **axis, "parameter_sets": [{"label": "setting", **entry}]})


@pytest.mark.parametrize("entry", [
    {"label": "range", "correlation_range_row_m": 50., "ell_y": 50.},
    {"label": "range", "correlation_range_col_m": 80., "ell_col_m": 80.},
    {"label": ""}, {"label": "empty"}, {"ell_row_m": 50.},
    {"label": "unknown", "unrecognized": 1.},
    {"label": "bad-mode", "truncation": {"n_modes": 3, "energy_threshold": .99}},
])
def test_invalid_parameter_sets_are_rejected(entry):
    with pytest.raises(ValueError):
        scenario_matrix(ReferenceScenario("RUN_1", sigma_R=.05), {"parameter_sets": [entry]})


def test_duplicate_labels_and_scientific_settings_are_rejected():
    base = ReferenceScenario("RUN_1", sigma_R=.05)
    with pytest.raises(ValueError, match="labels must be unique"):
        scenario_matrix(base, {"parameter_sets": [
            {"label": "same", "ell_y": 50.}, {"label": "same", "ell_y": 100.}]})
    with pytest.raises(ValueError, match="duplicate scenarios"):
        scenario_matrix(base, {"parameter_sets": [
            {"label": "first", "ell_y": 50.}, {"label": "second", "ell_row_m": 50.}]})


def test_existing_scientific_ids_and_direct_kernel_settings_remain_unchanged():
    base = ReferenceScenario("RUN_1", sigma_R=.05)
    assert base.scenario_id == "ref-f4e822aff1c0275c68f0"
    rows, metadata = scenario_matrix_with_metadata(base, {
        "reference_run": ["RUN_1", "RUN_2"], "ell_row_m": [500.], "ell_col_m": [800.],
        "truncation": [{"n_modes": 20}],
    })
    assert [row.scenario_id for row in rows] == ["ref-2a3cd5668aa74e3138d2", "ref-be8c19a151f323feb731"]
    assert metadata == {}
    gaussian = reference_scenario_from_config({"reference_run": "RUN_1", "sigma_R": .05,
                                              "covariance_family": "gaussian"})
    assert gaussian == replace(base, covariance_family="gaussian")


def test_committed_kernel_config_is_a_27_scenario_assumption_matrix():
    path = Path(__file__).resolve().parents[1] / "configs/reference_scenarios.kernels.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows, metadata = scenario_matrix_with_metadata(reference_scenario_from_config(config["base"]), config["axes"])
    assert len(rows) == len(metadata) == 27
    assert config["n_samples"] == 32
    assert config["diagnostic_factor"] == 8
    assert config["scenario_weights"] is None and config["distribution_target"] is None
    assert config["storage"]["retain_generated_fields"] is False
    assert {s.sigma_R for s in rows} == {.05}
    assert {s.energy_threshold for s in rows} == {.95}
