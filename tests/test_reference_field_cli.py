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
