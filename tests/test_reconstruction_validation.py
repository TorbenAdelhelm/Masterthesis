from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.special import ndtri
import yaml

from subsurface_uq.experiments.measurement_density import run
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec
from subsurface_uq.sampling.measurement_density import coverage_locations, native_ensembles
from subsurface_uq.validation.reconstruction import (
    spatial_test_partition, block_folds, select_covariance, predictive_scores, withheld_diagnostics,
)
from test_measurement_density import _config, FakeModel


RULES = {"minimum_test_blocks": 2, "coverage90_min": .8, "coverage90_max": .98,
         "standardized_rmse_min": .5, "standardized_rmse_max": 2., "absolute_standardized_bias_max": .5}


def test_reserved_test_blocks_and_buffer_never_supply_readings():
    test, excluded, blocks, metadata = spatial_test_partition((40, 45), block_cells=10,
        fraction=.2, buffer_cells=2, seed=12)
    indices, _ = coverage_locations((40, 45), 40, stride=2, seed=13, kind="balanced", excluded_mask=excluded)
    assert not excluded[tuple(indices.T)].any()
    assert len(np.unique(blocks[test])) == len(metadata["test_block_ids"])
    for b in metadata["test_block_ids"]:
        assert test[blocks == b].all()
    assert not metadata["reference_values_used"]
    with pytest.raises(ValueError):
        coverage_locations((40,45), 10, stride=2, seed=1, kind="balanced", excluded_mask=np.ones((2,2),dtype=bool))


def test_spatial_selection_folds_keep_colocated_blocks_together():
    indices = np.array([[0,0], [1,2], [12,2], [11,1], [25,25], [31,39]])
    folds, blocks = block_folds(indices, 10, 3, 12)
    for b in np.unique(blocks):
        assert len(np.unique(folds[blocks == b])) == 1
    assert len(np.unique(folds)) == 3


def test_selection_uses_only_the_supplied_prefix_not_future_readings():
    indices, _ = coverage_locations((24,24), 24, stride=2, seed=4, kind="balanced")
    readings = -9.+indices[:,0]/100+indices[:,1]/200
    candidates = [GSToolsReferenceSpec(f, length, length, sigma_R=sigma)
                  for f in ["matern32_radial", "exponential_radial"] for length in [5.,20.] for sigma in [.1,.2]]
    def select(values):
        return select_covariance((24,24), indices[:12], values[:12], candidates,
            cell_size_m=5., block_cells=6, n_folds=3, seed=13)
    first = select(readings)
    future = readings.copy(); future[12:] += 1000
    assert select(future) == first
    assert not first["test_values_used"] and not first["future_readings_used"]
    assert first["selection_measurements"] == 12


def test_reliability_diagnostics_detect_over_and_under_confidence():
    # Deterministic normal quantiles have approximately the declared coverage,
    # without a stochastic test failure or a correlated-pixel significance test.
    z = ndtri((np.arange(2000)+.5)/2000)
    blocks = np.arange(len(z)) % 10
    for sigma, expected in [(1., True), (.1, False), (10., False)]:
        d = withheld_diagnostics(z, np.zeros_like(z), np.full_like(z,sigma), blocks, rules=RULES)
        assert d["adequate_for_exploratory_propagation"] is expected
    score = predictive_scores(np.array([1.,-1.]), np.zeros(2), np.ones(2))
    assert score["rmse_log10"] == score["rms_predictive_sd_log10"] == score["standardized_rmse"] == 1
    zero = predictive_scores(np.ones(2), np.zeros(2), np.zeros(2))
    assert zero["standardized_rmse"] is None and zero["reliability"]["0.9"] == 0


def _validated_config(tmp_path, *, fail=False):
    source, cfg = _config(tmp_path)
    for key in ["kernel_fits", "fit_max_lag_m", "families", "sigma_log10", "screen_sigma_log10"]:
        cfg.pop(key)
    cfg["covariance_candidates"] = [{"family": "matern32_radial", "ell_row_m": 5., "ell_col_m": 5., "sigma_R": .001 if fail else .2}]
    cfg.update(measurement_counts=[6,10], coverage_scenarios=["balanced"], candidate_stride_cells=1)
    # Permissive numerical-test criteria are deliberately different from the
    # scientific study's stricter rules. This fixture tests the complete path.
    rules = {**RULES, "minimum_test_blocks":1, "coverage90_min":0., "coverage90_max":1.,
             "standardized_rmse_min":0., "standardized_rmse_max":1.e6,
             "absolute_standardized_bias_max":1.e6}
    if fail:
        rules.update(coverage90_min=.99, standardized_rmse_max=.01)
    cfg["validation"] = {"block_cells":3, "test_fraction":.2, "buffer_cells":0,
                         "selection_folds":2, "n_designs":2, "rules":rules}
    source.write_text(yaml.safe_dump(cfg))
    return source, cfg


def test_failed_withheld_assessment_never_loads_model_or_prepares_ensembles(tmp_path):
    source, _ = _validated_config(tmp_path, fail=True)
    def forbidden(*args, **kwargs):
        pytest.fail("failed validation must stop before the model is loaded")
    output = tmp_path/"study"
    result = run(source, output, model_factory=forbidden)
    assert result["status"] == "validation_failed"
    assert not list(output.rglob("permeability_ensemble.npy"))
    assert not list(output.rglob("temperature_ensemble.npy"))
    assert (output/"reconstruction_validation.md").exists()
    with pytest.raises(ValueError, match="withheld reconstruction uncertainty"):
        run(source, output, stage="inputs", model_factory=forbidden)


def test_validated_multidesign_pipeline_and_input_prediction_recovery(tmp_path):
    source, _ = _validated_config(tmp_path)
    output = tmp_path/"study"
    model = FakeModel()
    factory = lambda *args, **kwargs: model
    result = run(source, output, model_factory=factory)
    assert result["status"] == "complete"
    assert len(result["designs"]) == 2 and model.calls == 17
    run(source, output, stage="temperatures", model_factory=factory)
    assert model.calls == 17
    before = {str(p): p.stat().st_mtime_ns for p in output.rglob("permeability_ensemble.npy")}
    # Exercise input recovery directly; completed arrays are verified/reused.
    from subsurface_uq.experiments.measurement_density import load_config, prepare_inputs
    src, cfg, refs = load_config(source)
    prepare_inputs(src, cfg, refs, output, result)
    assert before == {str(p): p.stat().st_mtime_ns for p in output.rglob("permeability_ensemble.npy")}
    case = output/"RUN_1/balanced/design-000/n-0006/withheld_predictions.npz"
    with case.open("ab") as f: f.write(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        prepare_inputs(src, cfg, refs, output, result)


def test_native_complete_count_recovery_rejects_changed_arrays(tmp_path):
    shape = (5,5)
    spec = GSToolsReferenceSpec("exponential_radial", 5., 5., sigma_R=.2)
    reference = np.full(shape, 1.e-9)
    indices = np.array([[0,0], [4,4]])
    # Zero prior is enough to verify recovery integrity, not ensemble covariance.
    prior = np.zeros((4,*shape))
    first = native_ensembles(reference, spec, indices, [1,2], prior, tmp_path, cell_size_m=5.)
    path = tmp_path/"n-0001/permeability_ensemble.npy"
    timestamp = path.stat().st_mtime_ns
    assert native_ensembles(reference, spec, indices, [1,2], prior, tmp_path, cell_size_m=5.) == first
    assert path.stat().st_mtime_ns == timestamp
    values = np.load(path, mmap_mode="r+"); values[0,0,0] *= 2; values.flush(); del values
    with pytest.raises(ValueError, match="cache changed"):
        native_ensembles(reference, spec, indices, [1,2], prior, tmp_path, cell_size_m=5.)


def test_changing_only_test_truth_changes_scores_but_not_selection(tmp_path):
    source, cfg = _validated_config(tmp_path)
    cfg["covariance_candidates"].append({"family":"exponential_radial", "ell_row_m":10., "ell_col_m":10., "sigma_R":.1})
    source.write_text(yaml.safe_dump(cfg))
    first = tmp_path/"first"
    run(source, first, stage="validate")
    selections = [p.read_text() for p in sorted(first.rglob("covariance_selection.json"))]
    reference_path = tmp_path/"reference.npy"
    reference = np.load(reference_path)
    with np.load(first/"RUN_1/test_partition.npz") as partition:
        test_mask = partition["test_mask"]
    reference[test_mask] *= 100.
    # Source files can remain memory-mapped until cyclic garbage is collected
    # on Windows. A new immutable input also matches the study replay contract.
    changed_path = tmp_path/"changed-reference.npy"
    np.save(changed_path, reference)
    cfg["references"]["RUN_1"] = changed_path.name
    source.write_text(yaml.safe_dump(cfg))
    second = tmp_path/"second"
    result = run(source, second, stage="validate")
    assert [p.read_text() for p in sorted(second.rglob("covariance_selection.json"))] == selections
    assert result["status"] == "validated"  # permissive numerical fixture rules
    a = json.loads((first/"reconstruction_validation.json").read_text())["rows"]
    b = json.loads((second/"reconstruction_validation.json").read_text())["rows"]
    assert all(y["rmse_log10"] > x["rmse_log10"]+1 for x,y in zip(a,b))


def test_unavailable_large_budget_is_reported_without_reselecting(tmp_path, monkeypatch):
    source, _ = _validated_config(tmp_path)
    from subsurface_uq.experiments import measurement_density_validation as module
    original = module.predict_points
    def fail_large(shape, spec, indices, readings, queries, **kwargs):
        if len(indices) == 10:
            raise ValueError("declared numerical rank loss")
        return original(shape, spec, indices, readings, queries, **kwargs)
    monkeypatch.setattr(module, "predict_points", fail_large)
    output = tmp_path/"study"
    result = run(source, output, stage="validate")
    assert result["status"] == "validation_failed"
    report = json.loads((output/"reconstruction_validation.json").read_text())
    failed = [c for c in report["cases"] if c["n_measurements"] == 10]
    assert len(failed) == 2
    assert all(c["diagnostics"]["reason"] == "declared numerical rank loss" for c in failed)
    assert "unavailable" in (output/"reconstruction_validation.md").read_text()
