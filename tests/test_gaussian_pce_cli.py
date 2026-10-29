import json
from types import SimpleNamespace

import numpy as np
import torch

from subsurface_uq.experiments import release25_gaussian_pce as cli
from subsurface_uq.sampling.reference_field import build_reference_field_maps, save_reference_field_input_model


def test_gaussian_cli_connects_serialized_law_to_forward_adapter_and_saves_validation(tmp_path, monkeypatch):
    reference = np.full((4, 4), 10**-9.5)
    model, _ = build_reference_field_maps(reference, cell_size_m=20, residual_std_log10_k=.1,
                                           length_scale_m=(100, 120), n_modes=2)
    input_path = tmp_path / "input.yaml"
    save_reference_field_input_model(input_path, model)
    class Adapter:
        def predict(self, permeability, fixed):
            return {"temperature": 10. + torch.log10(permeability)}
    runtime = SimpleNamespace(adapter=Adapter(), scenario=SimpleNamespace(shape=(4, 4), fixed=None))
    monkeypatch.setattr(cli, "Release25Runtime", SimpleNamespace(from_paths=lambda **kwargs: runtime))
    monkeypatch.setattr(cli, "configure_release25_streamlines", lambda *args, **kwargs: None)
    output = tmp_path / "pce.npz"
    assert cli.main(["--input-model", str(input_path), "--release25-repo", "release25",
                     "--cnn1-dir", "cnn1", "--cnn2-dir", "cnn3", "--prepared-pki-dir", "prepared",
                     "--fixed-run-id", "RUN_1", "--cell-size-m", "20", "--degree", "1",
                     "--n-train", "8", "--n-validation", "8", "--background-temperature", "10",
                     "--output", str(output)]) == 0
    with np.load(output, allow_pickle=False) as result:
        np.testing.assert_allclose(result["validation_prediction"], result["validation_qoi"], atol=2e-6)
        metadata = json.loads(str(result["metadata_json"]))
    assert metadata["pce"]["coordinate_distribution"] == "iid_standard_normal"
    assert metadata["random_k"] is False
