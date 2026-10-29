import json

import h5py
import numpy as np
import pytest
import yaml

from subsurface_uq.experiments.real_k_training_support import main
from subsurface_uq.experiments.reference_field_permeability import coarsen_reference
from subsurface_uq.sampling.calibration import estimate_directional_variograms
from subsurface_uq.sampling.darus_real import RELEASE25_INITIAL_TIME_GROUP, RELEASE25_PERMEABILITY_DATASET


def _dataset(root):
    root.mkdir()
    (root / "settings.yaml").write_text(yaml.safe_dump({"grid": {"size [m]": [160, 200]}}))
    for index in range(3):
        run = root / f"RUN_{index}"
        run.mkdir()
        field = (10 ** np.linspace(-10.5 + index / 10, -9.5 + index / 10, 32 * 40)).reshape(32, 40)
        if index == 1:
            with h5py.File(run / "permeability.h5", "w") as handle:
                handle["permeability"] = field.ravel()
                handle["Cell Ids"] = np.arange(1, field.size + 1)
        else:
            with h5py.File(run / "pflotran.h5", "w") as handle:
                handle.create_group(RELEASE25_INITIAL_TIME_GROUP)[RELEASE25_PERMEABILITY_DATASET] = field.ravel()
    return root


def test_profiler_runs_on_real_hdf5_without_fabricating_training_membership(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    output = tmp_path / "support"
    assert main(["--dataset-root", str(dataset), "--output-dir", str(output)]) == 0
    report = json.loads((output / "training_support.json").read_text())
    assert report["status"] == "realistic_dataset_support"
    assert report["training_membership"]["status"] == "unresolved"
    assert report["target"] is None
    assert report["dataset_inventory_validation"]["pooled"]["normality_deviation"] is None
    for row in report["runs"].values():
        assert row["shape"] == [32, 40]
        assert row["diagnostic_support"]["cell_size_m"] == 40
        assert np.asarray(row["spectral"]["power_2d"]).shape == (4, 5)
        assert row["directional_variograms"]["row"]["distance_m"][0] == 40
        assert not any(key.endswith("_error") for key in row)
    # Passing positive conductivity to the existing APIs performs exactly one
    # logarithm and preserves explicitly recorded diagnostic support.
    with h5py.File(dataset / "RUN_0/pflotran.h5") as handle:
        kh = np.asarray(handle[RELEASE25_INITIAL_TIME_GROUP][RELEASE25_PERMEABILITY_DATASET], dtype=np.float32).reshape(32, 40).astype(float) * 7.5e6
    expected = estimate_directional_variograms(coarsen_reference(kh, 8), cell_size_m=40)
    np.testing.assert_allclose(report["runs"]["RUN_0"]["directional_variograms"]["row"]["semivariance_log10"], expected["y"][1])


@pytest.mark.parametrize("selected", [None, "RUN_2"])
def test_profiler_distinguishes_resolved_and_profiled_training_runs(tmp_path, selected):
    dataset = _dataset(tmp_path / "dataset")
    inputs = tmp_path / "prepared/Inputs"
    inputs.mkdir(parents=True)
    for i in range(3):
        (inputs / f"RUN_{i}.pt").write_bytes(b"inventory fixture")
    metadata = tmp_path / "command_line_arguments.yaml"
    metadata.write_text(yaml.safe_dump({"order_data": [[0], [1], [2]]}))
    output = tmp_path / "support"
    args = ["--dataset-root", str(dataset), "--output-dir", str(output),
            "--cnn1-command-line", str(metadata), "--cnn3-command-line", str(metadata),
            "--cnn1-prepared-inputs", str(inputs), "--cnn3-prepared-inputs", str(inputs),
            "--distribution-target"]
    if selected:
        args += ["--runs", selected]
    assert main(args) == 0
    report = json.loads((output / "training_support.json").read_text())
    assert report["common_proven_training_runs"] == ["RUN_0"]
    assert report["target"]["log10_std"] == .5
    if selected:
        assert report["profiled_common_training_runs"] == []
        assert report["status"] == "resolved_but_no_profiled_common_training_runs"
        assert "common_training_validation" not in report
    else:
        assert report["profiled_common_training_runs"] == ["RUN_0"]
        assert report["status"] == "training_support"
        assert report["common_training_validation"]["field_count"] == 1
        assert report["runs"]["RUN_1"]["role"]["cnn1"] == "validation"
        assert report["runs"]["RUN_2"]["role"]["cnn3"] == "test"
