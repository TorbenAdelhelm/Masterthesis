from pathlib import Path

import pytest
import yaml

from subsurface_uq.sampling.training_split_provenance import (
    inventory_from_prepared_inputs,
    resolve_release25_training_split,
)


def test_order_data_maps_indices_to_numeric_run_inventory(tmp_path):
    inputs = tmp_path / "prepared" / "Inputs"
    inputs.mkdir(parents=True)
    for name in ("RUN_10.pt", "RUN_2.pt", "RUN_7.pt", "ignore.txt"):
        (inputs / name).write_bytes(b"x")
    assert inventory_from_prepared_inputs(inputs) == ("RUN_2", "RUN_7", "RUN_10")
    metadata = tmp_path / "command_line_arguments.yaml"
    metadata.write_text(yaml.safe_dump({"order_data": [[0, 2], 1]}), encoding="utf-8")
    split = resolve_release25_training_split(metadata, prepared_inputs_dir=inputs)
    assert split.train_runs == ("RUN_2", "RUN_10")
    assert split.validation_runs == ("RUN_7",)
    assert split.test_runs == ()
    assert split.status.startswith("resolved")


def test_split_refuses_to_guess_without_inventory(tmp_path):
    metadata = tmp_path / "command_line_arguments.yaml"
    metadata.write_text(yaml.safe_dump({"order_data": [[0, 1, 2], 3]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unresolved"):
        resolve_release25_training_split(metadata)
