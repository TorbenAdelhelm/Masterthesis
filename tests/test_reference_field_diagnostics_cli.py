"""Exercise portable-field validation, provenance and informative failure cases."""
from hashlib import sha256
import json

import numpy as np
import pytest
import yaml
from scipy.special import ndtri
from scipy.stats import qmc

from subsurface_uq.experiments import reference_field_diagnostics as diagnostics
from subsurface_uq.sampling.reference_field import (
    build_reference_field_maps,
    save_reference_field_input_model,
)


@pytest.fixture
def artifact(tmp_path):
    row, column = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
    reference = np.power(10., -10. + .08 * row + .03 * column + .03 * np.sin(column))
    field_map, _ = build_reference_field_maps(
        reference, cell_size_m=5., residual_std_log10_k=.05,
        length_scale_m=(15., 25.), n_modes=4, energy_threshold=1.,
    )
    path = tmp_path / "source" / "stochastic_input_model.yaml"
    save_reference_field_input_model(path, field_map,
        source_metadata={"reference_run": "RUN_2", "permeability_convention": "historical-training"})
    return path, field_map


@pytest.fixture
def fast_reports(monkeypatch):
    # The smoke test exercises actual PNGs. Other cases isolate design/validation
    # behavior from the cost of re-rendering identical presentation layouts.
    def write(report, root):
        (root / "validation_report.md").write_text("Reference validation", encoding="utf-8")
        return {"markdown": "validation_report.md"}
    monkeypatch.setattr(diagnostics, "write_reference_validation_report", write)


def _existing(tmp_path, artifact, fields, **options):
    path, _ = artifact
    source = tmp_path / "fields.npy"
    np.save(source, fields)
    before = source.read_bytes()
    report = diagnostics.validate_reference_fields(path, tmp_path / "validation",
        generated_fields=source, diagnostic_factor=1, **options)
    assert source.read_bytes() == before
    assert report["source"]["generated_fields_sha256"] == sha256(before).hexdigest()
    return report


def test_existing_fields_write_strict_json_pngs_and_preserve_native_sources(tmp_path, artifact):
    path, field_map = artifact
    coordinates = np.random.default_rng(41).normal(size=(4, field_map.dimension))
    fields = field_map.map_coordinates(coordinates)
    artifact_bytes = {source: source.read_bytes() for source in path.parent.iterdir()}
    report = _existing(tmp_path, artifact, fields, method="MC")
    assert all(source.read_bytes() == before for source, before in artifact_bytes.items())
    assert report["source"]["input_model_sha256"] == sha256(path.read_bytes()).hexdigest()
    assert report["source"]["reference_run"] == "RUN_2"
    assert report["source"]["sampling_method"] == "MC"
    assert "user assertion" in report["source"]["design_provenance"]
    assert report["support"]["native_shape"] == [8, 8]
    assert report["support"]["diagnostic_cell_size_m"] == 5.
    assert report["native_marginals"]["reference"]["pixel_count"] == 64
    assert len(report["native_marginals"]["generated_per_field"]) == 4
    assert len(report["native_agreement"]) == 4
    assert report["no_filtering"] is True
    assert report["source"]["sample_filtering"] is None
    assert report["source"]["scenario_pooling"] is False
    assert report["assessment"]["status"] == "reference_review_required"
    assert report["assessment"]["training_support"]["status"] == "unassessed_by_reference_comparison"
    assert report["assessment"]["temperature_qoi_convergence"] == "not_evaluated"
    for index, row in enumerate(report["source"]["fields"]):
        assert row["sample_index"] == index
        assert row["physical_field_sha256"] == sha256(fields[index].tobytes()).hexdigest()
        assert row["dtype"] == str(fields.dtype)
    for key in ("distribution_png", "correlation_png", "spectra_png", "probe_law_png"):
        assert (tmp_path / "validation" / report["outputs"][key]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved = json.loads((tmp_path / "validation" / "validation.json").read_text())
    assert saved == report
    json.dumps(report, allow_nan=False)


def test_one_field_does_not_claim_covariance_resolution(tmp_path, artifact, fast_reports):
    _, field_map = artifact
    fields = field_map.map_coordinates(np.zeros((1, field_map.dimension)))
    report = _existing(tmp_path, artifact, fields)
    assert report["assessment"]["ensemble_resolution"] == "single_realization_only"
    assert report["probe_law"]["ensemble_status"] == "insufficient_fields_for_empirical_covariance"
    assert report["probe_law"]["covariance_relative_frobenius_error"] is None
    assert all(value is None for row in report["probe_law"]["empirical_covariance_log10"] for value in row)
    assert report["coherence"]["ensemble_envelope"]["status"] == "single_field_no_ensemble_spread"
    json.dumps(report, allow_nan=False)


def test_permutation_preserves_marginals_but_reveals_spatial_incoherence(tmp_path, artifact, fast_reports):
    _, field_map = artifact
    reference = field_map.reference_permeability
    permuted = np.random.default_rng(31).permutation(reference.ravel()).reshape(reference.shape)
    report = _existing(tmp_path, artifact, permuted[None])
    marginal = report["native_marginals"]
    assert marginal["reference"]["quantiles_m2"] == marginal["generated_per_field"][0]["quantiles_m2"]
    assert marginal["reference"]["log10_std"] == pytest.approx(marginal["generated_per_field"][0]["log10_std"])
    fidelity = report["coherence"]["reference_fidelity"]
    assert fidelity["marginal"]["wasserstein_distance_log10"] == pytest.approx(0.)
    assert report["native_agreement"][0]["log10_spatial_correlation"] < .5
    assert max(curve["relative_rmse"] for curve in fidelity["directional_variograms"].values()) > .5
    assert fidelity["spectra"]["radial_power_relative_l2"] > .1


@pytest.mark.parametrize("method", ["MC", "RQMC"])
def test_new_design_reproduces_coordinates_fields_and_diagnostics(tmp_path, artifact, fast_reports, method):
    path, field_map = artifact
    if method == "MC":
        coordinates = np.random.default_rng(19).normal(size=(8, field_map.dimension))
    else:
        coordinates = ndtri(qmc.Sobol(field_map.dimension, scramble=True, seed=19).random_base2(3))
    reports = [diagnostics.validate_reference_fields(path, tmp_path / f"run-{index}",
        n_samples=8, seed=19, method=method, diagnostic_factor=1) for index in range(2)]
    assert reports[0] == reports[1]
    report = reports[0]
    assert report["source"]["coordinate_design_sha256"] == sha256(coordinates.tobytes()).hexdigest()
    assert report["source"]["mode"] == "validation_design"
    assert report["source"]["n_samples"] == 8
    assert report["probe_law"]["sampling_method"] == method
    if method == "RQMC":
        sampling = report["probe_law"]["mean_sampling_error"]
        assert sampling["status"] == "not_assessed_without_independent_replications"
        assert sampling["expected_log10_mean_standard_error"] is None
    for coordinate, row in zip(coordinates, report["source"]["fields"]):
        expected = field_map.map_coordinates(coordinate)
        assert row["physical_field_sha256"] == sha256(expected.tobytes()).hexdigest()


def test_spatial_coarsening_and_native_probes_are_distinct(tmp_path, artifact, fast_reports):
    path, _ = artifact
    report = diagnostics.validate_reference_fields(path, tmp_path / "coarse", n_samples=4,
        seed=1, method="MC", diagnostic_factor=2, probe_grid_size=3)
    assert report["support"]["native_shape"] == [8, 8]
    assert report["support"]["diagnostic_shape"] == [4, 4]
    assert report["support"]["native_cell_size_m"] == 5.
    assert report["support"]["diagnostic_cell_size_m"] == 10.
    assert report["probe_law"]["probe_count"] >= 9
    selected = {tuple(index) for index in report["support"]["probe_indices_yx"]}
    assert {tuple(index) for index in diagnostics._probe_indices((8, 8), 3)} <= selected
    assert report["probe_law"]["indices_yx"] == report["support"]["probe_indices_yx"]
    assert report["native_marginals"]["reference"]["pixel_count"] == 64
    selection = report["support"]["probe_selection"]
    assert selection["strategy"] == "global_uniform_grid_plus_declared_length_scale_stencil"
    assert selection["selected_before_seeing_fields"] is True
    assert selection["declared_row_column_length_scales_m"] == [15., 25.]
    pairs = list(map(tuple, report["probe_law"]["indices_yx"]))
    center, nearby = pairs.index((4, 4)), pairs.index((2, 4))
    assert report["probe_law"]["expected_correlation"][center][nearby] > .5


def test_native_probe_stencil_resolves_anisotropic_lengths_before_drawing_fields():
    shape, lengths, spacing = (100, 160), (20., 50.), 5.
    indices = diagnostics._probe_indices(shape, 4, length_scale_m=lengths, cell_size_m=spacing)
    selected = {tuple(index) for index in indices}
    assert len(selected) == len(indices)
    assert np.issubdtype(indices.dtype, np.integer)
    # Half-length row and column probes lie close to the same native center.
    assert {(50, 80), (48, 80), (52, 80), (50, 75), (50, 85),
            (46, 80), (50, 70), (42, 80), (50, 60), (48, 75), (54, 90)} <= selected
    global_grid = diagnostics._probe_indices(shape, 4)
    assert {tuple(index) for index in global_grid} <= selected
    assert len(indices) <= 16 + 21
    np.testing.assert_array_equal(indices, diagnostics._probe_indices(shape, 4,
        length_scale_m=lengths, cell_size_m=spacing))


def test_probe_stencil_clips_deduplicates_and_keeps_native_cells_on_small_grids():
    indices = diagnostics._probe_indices((3, 4), 2,
        length_scale_m=(500., 800.), cell_size_m=5.)
    assert np.issubdtype(indices.dtype, np.integer)
    assert np.all(indices >= 0) and np.all(indices < [3, 4])
    assert len({tuple(index) for index in indices}) == len(indices)
    assert {(0, 0), (0, 3), (2, 0), (2, 3), (1, 2)} <= {tuple(index) for index in indices}


def test_cli_criteria_pass_does_not_promote_scientific_approval(tmp_path, artifact, fast_reports):
    path, _ = artifact
    criteria = tmp_path / "criteria.yaml"
    criteria.write_text(yaml.safe_dump({"max_log10_rmse": 100.}))
    output = tmp_path / "cli"
    assert diagnostics.main(["--input-model", str(path), "--output-dir", str(output),
        "--n-samples", "4", "--seed", "3", "--method", "MC", "--diagnostic-factor", "1",
        "--criteria", str(criteria)]) == 0
    report = json.loads((output / "validation.json").read_text())
    assert report["assessment"]["configured_reference_checks"]["status"] == "configured_checks_satisfied"
    assert report["assessment"]["status"] == "reference_review_required"


def test_failed_criterion_is_reported_without_filtering_fields(tmp_path, artifact, fast_reports):
    path, _ = artifact
    report = diagnostics.validate_reference_fields(path, tmp_path / "failed-review",
        n_samples=4, seed=7, method="MC", diagnostic_factor=1,
        criteria={"max_log10_rmse": 0.})
    checks = report["assessment"]["configured_reference_checks"]
    assert checks["status"] == "deviations_observed"
    assert checks["checks"]["max_log10_rmse"]["satisfied"] is False
    assert report["source"]["n_samples"] == 4
    assert len(report["source"]["fields"]) == len(report["native_agreement"]) == 4
    assert report["no_filtering"] is True


@pytest.mark.parametrize("options,match", [
    ({"diagnostic_factor": 0}, "diagnostic_factor"),
    ({"diagnostic_factor": 3}, "divisible"),
    ({"diagnostic_factor": 4}, "at least three"),
    ({"probe_grid_size": 17}, "must not exceed"),
    ({"max_lag_cells": 0}, "max_lag_cells"),
    ({"n_samples": 0}, "n_samples"),
    ({"seed": None}, "explicit"),
    ({"method": None}, "explicit"),
    ({"method": "RQMC", "n_samples": 3}, "power-of-two"),
    ({"criteria": {"invented": 1.}}, "unknown"),
    ({"criteria": {"max_log10_rmse": float("inf")}}, "finite"),
])
def test_invalid_design_options_create_no_output(tmp_path, artifact, fast_reports, options, match):
    path, _ = artifact
    params = {"n_samples": 4, "seed": 1, "method": "MC", "diagnostic_factor": 1, **options}
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=match):
        diagnostics.validate_reference_fields(path, output, **params)
    assert not output.exists()


@pytest.mark.parametrize("options", [{"n_samples": 2}, {"seed": 1}])
def test_existing_fields_forbid_design_overrides(tmp_path, artifact, fast_reports, options):
    path, field_map = artifact
    source = tmp_path / "existing.npy"
    np.save(source, field_map.reference_permeability[None])
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="do not override"):
        diagnostics.validate_reference_fields(path, output, generated_fields=source,
            diagnostic_factor=1, **options)
    assert not output.exists()


@pytest.mark.parametrize("shape", [(0, 8, 8), (8, 8), (2, 8, 7)])
def test_invalid_array_shapes_create_no_output(tmp_path, artifact, fast_reports, shape):
    path, _ = artifact
    source = tmp_path / "shape.npy"
    np.save(source, np.full(shape, 1.e-10))
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="nonempty shape"):
        diagnostics.validate_reference_fields(path, output, generated_fields=source, diagnostic_factor=1)
    assert not output.exists()


@pytest.mark.parametrize("invalid", [0., -1., float("nan"), float("inf"), 1.e-10 + 1j])
def test_invalid_numeric_fields_leave_sources_and_output_untouched(tmp_path, artifact, fast_reports, invalid):
    path, field_map = artifact
    fields = field_map.reference_permeability[None].astype(complex if np.iscomplexobj(invalid) else float)
    fields[0, 0, 0] = invalid
    source = tmp_path / "invalid.npy"
    np.save(source, fields)
    before = source.read_bytes()
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="positive|finite|real"):
        diagnostics.validate_reference_fields(path, output, generated_fields=source, diagnostic_factor=1)
    assert source.read_bytes() == before
    assert not output.exists()


def test_nonempty_output_is_rejected_without_overwriting(tmp_path, artifact, fast_reports):
    path, _ = artifact
    output = tmp_path / "occupied"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="nonempty"):
        diagnostics.validate_reference_fields(path, output, n_samples=4, seed=1,
            method="MC", diagnostic_factor=1)
    assert marker.read_text() == "preserve"
    assert list(output.iterdir()) == [marker]


def test_native_structure_reveals_checkerboard_hidden_by_coarse_plots(tmp_path, artifact, fast_reports):
    path, field_map = artifact
    reference = field_map.reference_permeability
    rows, columns = np.indices(reference.shape)
    changed = reference * 10. ** (.2 * (-1.) ** (rows + columns))
    source = tmp_path / "checkerboard.npy"
    np.save(source, np.stack((reference, changed)))
    report = diagnostics.validate_reference_fields(path, tmp_path / "coarse_plots",
        generated_fields=source, diagnostic_factor=2)
    assert report["coherence"]["reference_fidelity"]["marginal"]["wasserstein_distance_log10"] < 1e-13
    native = report["native_short_range"]
    assert native["cell_size_m"] == 5.
    changed_metrics = native["generated_per_field"][1]
    for direction in ("row", "column"):
        gradient = f"gradient_rms_{direction}_log10_per_m"
        correlation = f"lag1_correlation_{direction}"
        assert changed_metrics[gradient] > 3 * native["reference"][gradient]
        assert changed_metrics[correlation] < native["reference"][correlation] - .3
