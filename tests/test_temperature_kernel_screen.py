"""Frozen-model screen wiring, without loading research model checkpoints."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def screen():
    path = Path(__file__).resolve().parents[1] / "examples/reference_pilot/run_temperature_pilot.py"
    spec = importlib.util.spec_from_file_location("temperature_kernel_screen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path, monkeypatch, screen):
    shape = (8, 8)
    prepared = tmp_path / "prepared"
    (prepared / "Inputs").mkdir(parents=True)
    (prepared / "Inputs/RUN_2.pt").write_bytes(b"fixed-pressure-material-source")
    (prepared / "info.yaml").write_text(yaml.safe_dump({"CellsNumber": list(shape), "CellsSize": [5., 5.]}))
    for directory in (tmp_path / "cnn1", tmp_path / "cnn3"):
        directory.mkdir()
        for filename in ("model.pt", "info.yaml", "HPS_options.yaml"):
            (directory / filename).write_bytes(b"immutable-model")
    release = tmp_path / "release"
    release.mkdir()
    (release / "model.py").write_text("# fixed upstream source\n")
    rq1 = tmp_path / "rq1.yaml"
    rq1.write_text("fixed model configuration\n")
    config = SimpleNamespace(random_k=False, cell_size_m=5., receptors=[(1, 1)],
        mean_anomaly_roi=(0, 3, 0, 3), prepared_pki_dir=prepared, fixed_run_id="RUN_2",
        release25_repo=release, cnn1_dir=tmp_path / "cnn1", cnn2_dir=tmp_path / "cnn3",
        device="cpu", streamline_method="RK45", streamline_mode="bounded", streamline_max_nfev=500,
        streamline_diagnostics=False, streamline_slow_seconds=2., background_temperature=10.,
        to_dict=lambda: {"fixed_run_id": "RUN_2", "pressure": "held fixed"})
    counts = {"runtime": 0, "predict": 0, "maps": 0}
    failures = {"temperature": None, "velocity": None}
    bundle = SimpleNamespace(info={"Inputs": {"permeability": {"min": 1.e-12, "max": 1.e-8}}})

    class Adapter:
        cnn1 = cnn2 = bundle

        def predict(self, permeability, fixed):
            counts["predict"] += 1
            field = permeability.detach().cpu().numpy()
            temperature = 10. + np.log10(field[1:-1, 1:-1]) + 10.
            if failures["temperature"] == "nan":
                temperature[0, 0] = np.nan
            if failures["temperature"] == "shape":
                temperature = temperature[:1]
            if failures["temperature"] == "changing" and counts["predict"] > 1:
                temperature = temperature[:-1]
            velocity = np.stack([field[1:-1, 1:-1] * 1.e5, field[1:-1, 1:-1] * -2.e5])
            if failures["velocity"] == "nan":
                velocity[0, 0, 0] = np.nan
            if failures["velocity"] == "shape":
                velocity = velocity[:1]
            return {"temperature": torch.as_tensor(temperature[None]),
                    "velocity": torch.as_tensor(velocity), "streamlines": torch.zeros((2, 6, 6))}

    def runtime_factory(**kwargs):
        counts["runtime"] += 1
        assert kwargs["run_id"] == "RUN_2"
        return SimpleNamespace(adapter=Adapter(), scenario=SimpleNamespace(shape=shape, fixed=object()))

    monkeypatch.setattr(screen, "load_rq1_config", lambda *a, **k: config)
    monkeypatch.setattr(screen.Release25Runtime, "from_paths", runtime_factory)
    monkeypatch.setattr(screen, "configure_release25_streamlines", lambda *a, **k: None)
    monkeypatch.setattr(screen, "git_head", lambda path: "unchanged-git-head")
    monkeypatch.setattr(screen, "_temperature_png", lambda path, *a, **k: Path(path).write_bytes(b"PNG fixture"))

    def make_row(sid, run, family="matern32", dimension=2):
        root = tmp_path / sid
        root.mkdir(exist_ok=True)
        reference = 10. ** (-10. + np.arange(64).reshape(shape) / 200 + (0. if run == "RUN_1" else .2))
        np.save(root / "reference.npy", reference)
        config = {"reference_run": run, "sigma_R": .05, "covariance_family": family,
                  "ell_row_m": 30., "ell_col_m": 40., "n_modes": dimension, "energy_threshold": None}
        payload = {"input_law": "reference-centered-lognormal", "coordinate_distribution": "iid_standard_normal",
            "conditioning": None, "field_shape": list(shape), "coordinate_dimension": dimension,
            "prior": {"domain_size_m": [40., 40.]},
            "scenario": {"scenario_id": sid, "config": config},
            "reference_field": {"path": "reference.npy", "sha256": screen.sha256_file(root / "reference.npy")}}
        artifact = root / "input_model.yaml"
        artifact.write_text(yaml.safe_dump(payload))
        return {"scenario_id": sid, "config": config, "dimension": dimension,
                "retained_energy": .95, "input_model": str(artifact),
                "input_model_sha256": screen.sha256_file(artifact)}

    rows = [make_row("one", "RUN_1"), make_row("two", "RUN_2", "gaussian"), make_row("three", "RUN_2", "exponential")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"scenarios": rows}))

    def load_model(path):
        counts["maps"] += 1
        payload = yaml.safe_load(Path(path).read_text())
        reference = np.load(Path(path).parent / "reference.npy", allow_pickle=False)
        config = payload["scenario"]["config"]
        dimension = payload["coordinate_dimension"]
        basis = np.stack([(np.arange(64).reshape(shape) / 64. + .1) ** (index + 1) * .01
                          for index in range(dimension)])
        prior = SimpleNamespace(field_shape=shape, std_log10_k=.05, length_scale_m=(30., 40.),
                                covariance_model=config["covariance_family"], retained_energy_fraction=.95)
        mapping = SimpleNamespace(field_shape=shape, domain_size_m=(40., 40.), center="median",
            prior=prior, gaussian_map=prior, dimension=dimension,
            metadata={"reference_array_sha256": payload["reference_field"]["sha256"]},
            map_coordinates=lambda xi: reference * 10. ** np.einsum("i,ijk->jk", xi, basis))
        return SimpleNamespace(prior=prior, unconditional=mapping)

    monkeypatch.setattr(screen, "load_conditional_kl_input_model", load_model)
    args = SimpleNamespace(study_mode="kernel-screen", matrix_manifest=str(manifest), kl_manifest=None,
        scenario_ids=["one", "two"], rq1_config=str(rq1), output_dir=str(tmp_path / "output"),
        n_samples=2, seed=33, method="MC", reference_arrays=None, save_temperature_samples=1, plume_threshold_c=.5)
    return SimpleNamespace(args=args, rows=rows, counts=counts, failures=failures,
                           make_row=make_row, manifest=manifest, config=config)


def test_screen_selection_nominal_baselines_one_runtime_and_reports(screen, case):
    result = screen.run_pilot(case.args)
    assert case.counts == {"runtime": 1, "predict": 6, "maps": 2}
    assert result["primary_scenario_count"] == 2
    assert result["nominal_reference_count"] == 2
    assert set(result["nominal_references"]) == {"RUN_1", "RUN_2"}
    assert {row["scenario_id"] for row in result["scenarios"]} == {"one", "two"}
    assert not result["scenario_pooling"] and result["scenario_weights"] is None and result["sample_filtering"] is None
    assert "uncoupled" in result["coordinate_comparison"]
    assert "actual training membership" in result["training_support_status"]
    root = Path(case.args.output_dir)
    assert not (root / "qoi_truncation.json").exists()
    for row in result["scenarios"]:
        directory = root / row["scenario_id"]
        baseline = np.asarray(row["nominal_reference_qoi"])
        values = np.load(directory / "qoi_samples.npy")
        np.testing.assert_allclose(row["qoi_delta_to_nominal"]["mean"], (values - baseline).mean(axis=0))
        assert np.load(directory / "temperature_std.npy").shape == (6, 6)
        assert np.all(np.load(directory / "temperature_std.npy") >= 0)
        assert (directory / "temperature_sample_000.png").exists()
        assert not (directory / "temperature_sample_001.png").exists()
        assert len(row["flow_descriptors"]) == 2 and len(row["temperature_descriptors"]) == 2
        assert row["flow_descriptors"][0]["velocity_grid_shape"] == [2, 6, 6]
        assert row["temperature_descriptors"][0]["plume"]["threshold_C"] == .5
        assert row["normalization_range_diagnostics"][0]["cnn1"]["evaluated_grid_shape"] == [8, 8]
        assert "not a full training-support test" in row["normalization_range_diagnostics"][0]["interpretation"]


def test_screen_resume_checksums_and_immutable_context(screen, case):
    first = screen.run_pilot(case.args)
    second = screen.run_pilot(case.args)
    assert case.counts["runtime"] == 2 and case.counts["predict"] == 6
    assert second["new_model_predictions_this_invocation"] == 0 and second["cache_hits_this_invocation"] == 6
    assert first["scenarios"] == second["scenarios"]
    case.args.seed += 1
    with pytest.raises(ValueError, match="different immutable run context"):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 2
    case.args.seed -= 1
    cache = next((Path(case.args.output_dir) / "sample_cache").rglob("*.temperature.npy"))
    cache.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="invalid cached prediction"):
        screen.run_pilot(case.args)


def test_one_nominal_per_run_and_identical_fields_share_prediction_cache(screen, case):
    case.args.scenario_ids = ["two", "three"]
    result = screen.run_pilot(case.args)
    assert result["nominal_reference_count"] == 1
    # The stub ignores kernel shape, so these two laws deliberately return
    # identical physical arrays: preserve both scenarios, reuse their outputs.
    assert case.counts["runtime"] == 1 and case.counts["predict"] == 3
    assert result["cache_hits_this_invocation"] == 2
    assert result["total_evaluated_sample_occurrences"] == 4
    assert set(result["nominal_references"]) == {"RUN_2"}


@pytest.mark.parametrize("change,pattern", [
    ({"method": "RQMC", "n_samples": 3}, "power of two"),
    ({"scenario_ids": ["missing"]}, "present in the matrix"),
    ({"scenario_ids": ["one", "one"]}, "distinct"),
    ({"save_temperature_samples": -1}, "non-negative"),
    ({"plume_threshold_c": float("nan")}, "positive and finite"),
])
def test_screen_preflight_rejects_bad_design_before_model_load(screen, case, change, pattern):
    for key, value in change.items():
        setattr(case.args, key, value)
    with pytest.raises(ValueError, match=pattern):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 0
    assert not Path(case.args.output_dir).exists()


def test_sobol_dimension_preflight_before_models_or_map_construction(screen, case):
    rows = json.loads(case.manifest.read_text())
    rows["scenarios"][0]["dimension"] = screen.qmc.Sobol.MAXDIM + 1
    case.manifest.write_text(json.dumps(rows))
    case.args.method = "RQMC"
    with pytest.raises(ValueError, match="exceeds scipy Sobol maximum"):
        screen.run_pilot(case.args)
    assert case.counts == {"runtime": 0, "predict": 0, "maps": 0}


def test_artifact_shape_and_declared_dimension_checked_before_model_load(screen, case):
    row = case.rows[0]
    artifact = Path(row["input_model"])
    raw = yaml.safe_load(artifact.read_text())
    raw["field_shape"] = [4, 16]
    artifact.write_text(yaml.safe_dump(raw))
    row["input_model_sha256"] = screen.sha256_file(artifact)
    case.manifest.write_text(json.dumps({"scenarios": case.rows}))
    with pytest.raises(ValueError, match="full-resolution 5 m grid"):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 0
    raw["field_shape"] = [8, 8]
    raw["coordinate_dimension"] = 4
    artifact.write_text(yaml.safe_dump(raw))
    row["input_model_sha256"] = screen.sha256_file(artifact)
    case.manifest.write_text(json.dumps({"scenarios": case.rows}))
    with pytest.raises(ValueError, match="coordinate dimension mismatch"):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 0


def test_rqmc_design_is_replayable_and_requires_no_extra_model_initialization(screen, case):
    case.args.method = "RQMC"
    case.args.save_temperature_samples = 0
    result = screen.run_pilot(case.args)
    assert case.counts["runtime"] == 1 and case.counts["predict"] == 6
    assert result["method"] == "RQMC"
    assert not (Path(case.args.output_dir) / "one/temperature_sample_000.png").exists()
    coordinates = screen._coordinates(2, 2, case.args.seed, "RQMC")
    model = screen.load_conditional_kl_input_model(case.rows[0]["input_model"])
    field = np.ascontiguousarray(model.unconditional.map_coordinates(coordinates[0]), dtype=np.float32)
    record = json.loads((Path(case.args.output_dir) / "one/sample_diagnostics.json").read_text())[0]
    assert record["physical_field_sha256"] == screen.sha256(field.tobytes(order="C")).hexdigest()


def test_screen_uses_real_serialized_npz_reference_and_gaussian_map(screen, case, tmp_path, monkeypatch):
    from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
    from subsurface_uq.sampling.reference_field import save_reference_field_input_model
    from subsurface_uq.sampling.scenarios import ReferenceScenario

    reference = np.load(Path(case.rows[0]["input_model"]).parent / "reference.npy")
    scenario = ReferenceScenario(reference_run="RUN_1", sigma_R=.05, covariance_family="gaussian",
                                 ell_y=30., ell_x=40., n_modes=2, energy_threshold=None)
    field_map, _ = scenario.build_maps(reference, cell_size_m=5.)
    artifact = tmp_path / "real_serialized/input_model.yaml"
    payload = save_reference_field_input_model(artifact, field_map)
    assert payload["reference_field"]["path"].endswith(".reference.npz")
    row = {**scenario.manifest(), "dimension": field_map.dimension,
           "retained_energy": field_map.prior.retained_energy_fraction,
           "input_model": str(artifact), "input_model_sha256": screen.sha256_file(artifact)}
    case.manifest.write_text(json.dumps({"scenarios": [row]}))
    case.args.scenario_ids = [row["scenario_id"]]
    monkeypatch.setattr(screen, "load_conditional_kl_input_model", load_conditional_kl_input_model)
    result = screen.run_pilot(case.args)
    assert result["nominal_reference_count"] == 1
    assert case.counts["runtime"] == 1 and case.counts["predict"] == 3
    np.testing.assert_array_equal(screen._reference_array(artifact.parent / payload["reference_field"]["path"]), reference)
    nominal = np.load(Path(result["nominal_references"]["RUN_1"]["temperature_array"]))
    expected = 20. + np.log10(reference.astype(np.float32)[1:-1, 1:-1])
    np.testing.assert_allclose(nominal, expected, atol=1.e-6)


@pytest.mark.parametrize("output,kind", [("temperature", "nan"), ("temperature", "shape"),
                                         ("temperature", "changing"), ("velocity", "nan"), ("velocity", "shape")])
def test_invalid_model_outputs_never_produce_completed_summary(screen, case, output, kind):
    case.failures[output] = kind
    with pytest.raises(ValueError, match="temperature|velocity"):
        screen.run_pilot(case.args)
    assert not (Path(case.args.output_dir) / "summary.json").exists()


def test_screen_kl_comparisons_are_explicit_same_law_and_cached(screen, case, tmp_path):
    extra = case.make_row("extra", "RUN_1", dimension=3)
    kl = tmp_path / "kl_manifest.json"
    kl.write_text(json.dumps({"scenarios": [extra]}))
    case.args.kl_manifest = str(kl)
    result = screen.run_pilot(case.args)
    assert len(result["kl_comparisons"]) == 1
    assert result["kl_comparisons"][0]["scenario_ids"] == ["one", "extra"]
    assert result["kl_comparisons"][0]["coordinate_dimension"] == 3
    assert case.counts["predict"] == 8  # two nominal baselines, three times two samples
    assert case.counts["runtime"] == 1
    assert result["cache_hits_this_invocation"] == 4  # truncation comparison reuses all paired evaluations
    assert result["scenarios"][0]["coordinate_dimension"] == 3
    assert result["scenarios"][1]["coordinate_dimension"] == 2


def test_kl_different_law_is_not_silently_mode_paired(screen, case, tmp_path):
    extra = case.make_row("extra", "RUN_1", family="exponential", dimension=3)
    kl = tmp_path / "kl_manifest.json"
    kl.write_text(json.dumps({"scenarios": [extra]}))
    case.args.kl_manifest = str(kl)
    with pytest.raises(ValueError, match="differing only in truncation"):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 0


def test_legacy_guards_remain_and_new_mode_is_explicit(screen, case):
    case.args.study_mode = "legacy-pilot"
    with pytest.raises(ValueError, match="requires --kl-manifest"):
        screen.run_pilot(case.args)
    case.args.kl_manifest = case.args.matrix_manifest
    with pytest.raises(ValueError, match="selection requires"):
        screen.run_pilot(case.args)
    case.args.scenario_ids = None
    with pytest.raises(ValueError, match="twelve primary scenarios"):
        screen.run_pilot(case.args)
    assert case.counts["runtime"] == 0


def test_temperature_descriptors_and_actual_png(screen, tmp_path):
    temperature = np.arange(36, dtype=float).reshape(6, 6) / 100. + 10.
    descriptors = screen._temperature_descriptors(temperature, 10., .2)
    assert descriptors["plume"]["fraction_output_cells"] == pytest.approx(np.mean(temperature - 10. >= .2))
    assert 0 <= descriptors["centered_spectral_power_below_0_1_cycles_per_cell"] <= 1
    path = tmp_path / "temperature.png"
    screen._temperature_png(path, [temperature], ["Physical temperature"])
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
