import h5py
import numpy as np
import pytest
import yaml

from subsurface_uq.sampling.darus_real import load_release25_raw_permeability_run, load_release25_raw_permeability_dataset


def test_input_only_run_is_loaded_and_reordered_by_cell_ids(tmp_path):
    (tmp_path / "settings.yaml").write_text(yaml.safe_dump({"grid": {"size [m]": [40, 60]}}))
    run = tmp_path / "RUN_1"
    run.mkdir()
    ids = np.array([6, 1, 4, 2, 5, 3])
    with h5py.File(run / "permeability.h5", "w") as handle:
        handle["Cell Ids"] = ids
        handle["permeability"] = ids * 1.e-10
    expected = np.arange(1, 7).reshape(2, 3) * 1.e-10
    np.testing.assert_allclose(load_release25_raw_permeability_run(tmp_path, "RUN_1", cell_size_m=20), expected)
    fields, names = load_release25_raw_permeability_dataset(tmp_path, cell_size_m=20)
    assert names == ("RUN_1",)
    np.testing.assert_allclose(fields[0], expected)
    with h5py.File(run / "permeability.h5", "a") as handle:
        handle["Cell Ids"][0] = 1
    with pytest.raises(ValueError, match="Cell Id"):
        load_release25_raw_permeability_run(tmp_path, "RUN_1", cell_size_m=20)
