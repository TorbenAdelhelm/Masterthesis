import json

import numpy as np
import pytest

from subsurface_uq.experiments.reference_field_permeability import main, coarsen_reference
from subsurface_uq.sampling.input_model import load_conditional_kl_input_model
from subsurface_uq.sampling.munich_measurements import hydraulic_conductivity_to_intrinsic_permeability


def test_reference_cli_writes_reconstructible_law_and_truncation_comparison(tmp_path):
    reference = 10**(-9.5 + np.arange(64).reshape(8, 8) / 500)
    source = tmp_path / "reference.npy"
    np.save(source, reference)
    root = tmp_path / "results"
    assert main(["--reference-npy", str(source), "--permeability-convention", "historical-training",
                 "--cell-size-m", "20", "--residual-std-log10-k", ".05",
                 "--length-scale-y-m", "60", "--length-scale-x-m", "80",
                 "--energy-thresholds", ".95", ".99", ".999", "--n-samples", "4",
                 "--output-dir", str(root)]) == 0
    summary = json.loads((root / "summary.json").read_text())
    dimensions = [row["dimension"] for row in summary["truncation_sensitivity"]]
    assert dimensions == sorted(dimensions)
    loaded = load_conditional_kl_input_model(root / "energy_0.99" / "stochastic_input_model.yaml")
    np.testing.assert_allclose(loaded.unconditional.map_coordinates(np.zeros(loaded.prior.dimension)), reference, rtol=1e-6)
    assert np.load(root / "energy_0.99" / "generated_fields.npy").shape == (4, 8, 8)
    fidelity = json.loads((root / "energy_0.99" / "fidelity.json").read_text())
    assert np.asarray(fidelity["spectra"]["generated"]["power_2d"]).shape == (8, 8)


def test_geometric_coarsening_and_conversion_convention_are_explicit():
    reference = 10**np.arange(16).reshape(4, 4)
    expected = 10**np.array([[2.5, 4.5], [10.5, 12.5]])
    np.testing.assert_allclose(coarsen_reference(reference, 2), expected)
    kh = np.array([1.e-3, 1.e-4])
    historical = hydraulic_conductivity_to_intrinsic_permeability(kh, permeability_convention="historical-training")
    np.testing.assert_allclose(historical, kh / 7.5e6)
    physical = hydraulic_conductivity_to_intrinsic_permeability(kh)
    np.testing.assert_allclose(physical/historical, .7676985915176447)
    with pytest.raises(ValueError, match="convention"):
        hydraulic_conductivity_to_intrinsic_permeability(kh, permeability_convention="ambiguous")


@pytest.mark.parametrize("option", ["--max-abs-log10-mean-error", "--max-abs-log10-std-error", "--max-qq-rmse-standardized"])
def test_cli_partial_thresholds_are_json_safe_with_explicit_target(tmp_path, option):
    source = tmp_path / "ref.npy"
    np.save(source, 10 ** np.linspace(-10, -9.8, 64).reshape(8, 8))
    common = ["--reference-npy", str(source), "--permeability-convention", "historical-training",
              "--residual-std-log10-k", ".05", "--length-scale-row-m", "20", "--length-scale-col-m", "30",
              "--n-modes", "3", "--n-samples", "2", option, ".25"]
    with pytest.raises(ValueError, match="explicit distribution target"):
        main([*common, "--output-dir", str(tmp_path / "disabled")])
    assert not (tmp_path / "disabled").exists()
    assert main([*common, "--distribution-target", "--output-dir", str(tmp_path / "enabled")]) == 0
    validation = json.loads((tmp_path / "enabled/modes_3/distribution_validation.json").read_text())
    assert validation["target"]["log10_std"] == .5
    assert validation["sample_filtering"] is None
    limits = validation["pooled"]["heuristic_assessment"]["tolerances"]
    assert sum(v is None for v in limits.values()) == 2


def test_cli_target_options_enable_only_requested_diagnostics(tmp_path):
    source = tmp_path / "ref.npy"
    np.save(source, 10 ** np.linspace(-10, -9.8, 64).reshape(8, 8))
    common = ["--reference-npy", str(source), "--permeability-convention", "historical-training",
              "--residual-std-log10-k", ".05", "--length-scale-row-m", "20", "--length-scale-col-m", "30",
              "--n-modes", "3", "--n-samples", "2", "--seed", "33"]
    fields = []
    for name, flags in (("default", []), ("explicit", ["--target-log10-variance", ".5"])):
        assert main([*common, *flags, "--output-dir", str(tmp_path / name)]) == 0
        report = json.loads((tmp_path / name / "modes_3/distribution_validation.json").read_text())
        if not flags:
            assert report["target"] is None and report["pooled"]["heuristic_assessment"]["status"] == "not_assessed"
        else:
            assert report["target"]["log10_variance"] == pytest.approx(.5)
        fields.append(np.load(tmp_path / name / "modes_3/generated_fields.npy", allow_pickle=False))
    np.testing.assert_array_equal(*fields)
