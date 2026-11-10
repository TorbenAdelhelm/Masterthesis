"""Numerical, leakage, provenance and end-to-end output-field checks."""
import json
from pathlib import Path

import numpy as np
import pytest

from subsurface_uq.experiments.output_field_assessment import (
    area_event, file_hash, inventory, run, verified_source, write_json,
)
from subsurface_uq.statistics.output_field import (
    FieldKL, assess_fields, fit_kl, gaussian_scores, native_boundary,
    native_exceedance, normality_screen, project, score_law, split_indices,
)


def low_rank_fields(n=60):
    rng = np.random.default_rng(42)
    spatial, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    scores = rng.normal(size=(n, 3)) * [1.5, .7, .2]
    return (10.6 + scores @ spatial.T).reshape(n, 4, 5)


def test_snapshot_fit_matches_dense_svd_and_reload(tmp_path):
    fields = low_rank_fields()
    train, _ = split_indices(len(fields))
    model = fit_kl(fields, train, tmp_path / "model", max_rank=10, variance_fraction=1., chunk_cells=7)
    centered = fields[train].reshape(len(train), -1) - fields[train].mean(axis=0).ravel()
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    assert model.rank == 3
    np.testing.assert_allclose(model.info["eigenvalues_C2"][:3], singular[:3] ** 2 / (len(train) - 1), rtol=1e-12)
    scores = project(fields, model, chunk_cells=7)
    reconstructed = model.block(scores, 0, 20)
    np.testing.assert_allclose(reconstructed, fields.reshape(len(fields), -1), atol=2e-7)
    reloaded = FieldKL.load(tmp_path / "model")
    np.testing.assert_array_equal(project(fields, reloaded, 7), scores)
    np.testing.assert_allclose(model.modes @ model.modes.T, np.eye(3), atol=1e-7)


def test_holdout_cannot_change_fitted_basis_or_gaussian_law(tmp_path):
    fields = low_rank_fields()
    train, test = split_indices(len(fields))
    model = fit_kl(fields, train, tmp_path / "a", max_rank=2)
    altered = fields.copy(); altered[test] += 100
    second = fit_kl(altered, train, tmp_path / "b", max_rank=2)
    np.testing.assert_array_equal(model.mean, second.mean)
    np.testing.assert_array_equal(model.modes, second.modes)
    a = project(fields, model)[train]
    b = project(altered, second)[train]
    np.testing.assert_array_equal(gaussian_scores(a, 50, 12), gaussian_scores(b, 50, 12))
    assert not model.info["variance_target_met"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_field_is_a_valid_degenerate_gaussian(tmp_path, dtype):
    fields = np.full((40, 3, 4), 10.6, dtype=dtype)
    train, test = split_indices(len(fields))
    model = fit_kl(fields, train, tmp_path / "model")
    assert model.rank == 0
    assert model.info["variance_fraction_retained"] == 1.
    scores = project(fields, model)
    candidate = gaussian_scores(scores[train], 40, 1)
    result = assess_fields(fields, model, scores, train, test, candidate, tmp_path,
                           background=10.6, thresholds=[.1, 1.], cell_size_m=5)
    assert result["heldout_temperature_rmse_C"] == 0
    assert result["thresholds"][0]["gaussian_expected_area_analytic_m2"] == 0
    assert normality_screen(scores[train], scores[test])["active_dimensions"] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threshold", [.1, 1.])
def test_native_boundary_matches_strict_rounded_temperature_event(dtype, threshold):
    boundary = native_boundary(10.6, threshold, dtype)
    # Check each native representable value around the continuous event boundary.
    values = np.array([dtype(boundary)], dtype=dtype)
    below = np.nextafter(values, dtype(-np.inf))
    above = np.nextafter(values, dtype(np.inf))
    assert not native_exceedance(below, 10.6, threshold, dtype)[0]
    assert native_exceedance(above, 10.6, threshold, dtype)[0]
    # Nominal threshold addition must not silently replace native subtraction.
    assert np.isfinite(boundary)


def test_joint_screen_detects_non_gaussian_dependence_with_normal_marginals():
    rng = np.random.default_rng(9)
    training = rng.normal(size=(5000, 2))
    x = rng.normal(size=1500)
    heldout = np.column_stack((x, x * rng.choice([-1, 1], size=len(x))))
    screen = normality_screen(training, heldout)
    assert screen["rejected_at_0.05"] > 0
    assert any(item.get("holm_p_value", 1) < 1e-8 for item in screen["checks"])
    assert any(item["projection"].startswith("pair_") for item in screen["checks"])


def test_gaussian_sampling_matches_training_moments_and_is_reproducible():
    rng = np.random.default_rng(4)
    training = rng.normal(size=(200, 3)) @ np.array([[1., .4, .2], [0, 2, .1], [0, 0, .7]]) + 2
    candidate = gaussian_scores(training, 40000, 11)
    mean, covariance, _ = score_law(training)
    np.testing.assert_allclose(candidate.mean(axis=0), mean, atol=.025)
    np.testing.assert_allclose(np.cov(candidate.T), covariance, atol=.04)
    np.testing.assert_array_equal(candidate[:10], gaussian_scores(training, 10, 11))


def test_native_risk_reduction_preserves_area_identity_and_no_clipping(tmp_path):
    fields = low_rank_fields(100).astype(np.float32)
    train, test = split_indices(len(fields))
    model = fit_kl(fields, train, tmp_path / "model", max_rank=4, variance_fraction=1.)
    scores = project(fields, model)
    candidate = gaussian_scores(scores[train], 1500, 12)
    metrics = assess_fields(fields, model, scores, train, test, candidate, tmp_path,
                            background=10.6, thresholds=[.1], cell_size_m=5, chunk_cells=6,
                            area_events=[(.1, 200.)])
    area = np.load(tmp_path / "area_samples.npz")
    expected = native_exceedance(fields[test], 10.6, .1, fields.dtype).sum(axis=(1, 2)) * 25
    np.testing.assert_array_equal(area["direct_m2"][:, 0], expected)
    direct_probability = np.load(tmp_path / "probability_direct.npy")
    np.testing.assert_allclose(direct_probability.sum(dtype=float) * 25, expected.mean(), atol=1e-5)
    entry = metrics["thresholds"][0]
    assert abs(entry["gaussian_expected_area_analytic_m2"] - area["gaussian_m2"].mean()) < 4 * entry["area_m2"]["gaussian"]["mean_mcse"]
    assert 0 < metrics["below_background_cell_draw_fraction"]["gaussian"] < 1
    risk = entry["area_events"][0]["risk"]["direct"]
    assert risk["probability"] == np.mean(expected > 200)
    assert risk["ci95"][0] <= risk["probability"] <= risk["ci95"][1]


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    manifest = {"status": "complete", "study_id": "synthetic-test", "reference_id": "RUN_4",
                "configuration": {"scenarios": {"RUN_4_gaussian": {}}, "measurement_counts": [64, 128],
                                  "cell_size_m": 5., "risk_thresholds_C": [.1, 1.]},
                "model": {"release25": {"background_temperature": 10.6}}}
    write_json(root / "manifest.json", manifest)
    for count in [64, 128]:
        case = root / f"RUN_4_gaussian/truth-0000/balanced-000/n-{count:04d}"
        case.mkdir(parents=True)
        fields = low_rank_fields(40).astype(np.float32)
        np.save(case / "temperature_ensemble.npy", fields)
        write_json(case / "summary.json", {"scenario": "RUN_4_gaussian", "trial": 0,
                   "design": "balanced-000", "n_measurements": count, "n_samples": len(fields)})
        write_json(case / "prediction_records.json", [{"sample_index": i, "input_sha256": f"{i:064x}",
                   "temperature_sha256": f"{i+1:064x}", "seed": i + 10} for i in range(len(fields))])
        write_json(case / "result_files.json", {name: file_hash(case / name) for name in
                   ("summary.json", "prediction_records.json", "temperature_ensemble.npy")})
    return root


def test_end_to_end_report_model_exports_filters_and_source_unchanged(study, tmp_path):
    before = {p.relative_to(study): file_hash(p) for p in study.rglob("*") if p.is_file()}
    output = tmp_path / "assessment"
    result = run(study, output, profiles=["RUN_4_gaussian"], counts=[128], gaussian_draws=32,
                 max_rank=3, variance_fraction=1., chunk_cells=7, area_events=[(.1, 250)], progress=lambda _: None)
    assert result["status"] == "complete"
    assert len(result["cases"]) == 1
    case = output / result["cases"][0]["path"]
    saved = np.load(case / "latent_scores.npz")
    np.testing.assert_array_equal(saved["sample_index"], np.arange(40))
    assert saved["source_input_sha256"][3] == f"{3:064x}"
    assert saved["is_train"].sum() == 28
    for filename in ("index.html", "assessment.json", "model/modes.npy", "latent_qq.png",
                     "spatial_probabilities.png", "area_distributions.png", "sample_links.json"):
        assert (case / filename).stat().st_size > 0
    assert "direct" in (case / "index.html").read_text(encoding="utf-8")
    assert before == {p.relative_to(study): file_hash(p) for p in study.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fresh"):
        run(study, output)
    with pytest.raises(ValueError, match="outside"):
        run(study, study / "analysis")


def test_source_tampering_fails_and_failure_is_visible(study, tmp_path):
    case = study / "RUN_4_gaussian/truth-0000/balanced-000/n-0064"
    with (case / "temperature_ensemble.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        run(study, tmp_path / "failed", counts=[64], progress=lambda _: None)
    assert json.loads((tmp_path / "failed/manifest.json").read_text())["status"] == "failed"


def test_inventory_rejects_unknown_or_missing_selection(study):
    manifest = json.loads((study / "manifest.json").read_text())
    with pytest.raises(ValueError, match="profile"):
        inventory(study, manifest, profiles=["other"])
    with pytest.raises(ValueError, match="count"):
        inventory(study, manifest, counts=[512])
    manifest["configuration"]["scenarios"]["absent"] = {}
    with pytest.raises(ValueError, match="missing selected"):
        inventory(study, manifest)


def test_inventory_catches_missing_truth_even_when_profile_count_exists(study):
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["configuration"].update(n_truths=2, n_designs=1, coverage_scenarios=["balanced"])
    with pytest.raises(ValueError, match="declared truths"):
        inventory(study, manifest)


def test_split_and_numerical_validation(tmp_path):
    a, b = split_indices(500)
    np.testing.assert_array_equal(a, split_indices(500)[0])
    assert len(set(a) & set(b)) == 0
    assert len(a) == 350 and len(b) == 150
    with pytest.raises(ValueError):
        split_indices(20, .9)
    with pytest.raises(ValueError):
        fit_kl(low_rank_fields(), [0, 0, 1], tmp_path / "duplicate")
    broken = low_rank_fields(); broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        fit_kl(broken, np.arange(30), tmp_path / "nonfinite")
    model = fit_kl(low_rank_fields(), np.arange(30), tmp_path / "valid")
    with pytest.raises(ValueError, match="nonfinite"):
        project(broken, model)
    assert area_event("0.1:9000000") == (.1, 9000000)
