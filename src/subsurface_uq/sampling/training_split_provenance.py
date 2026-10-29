"""Resolve release25 training/validation RUN membership without guessing.

The release25 code interprets ``order_data`` as indices into the numerically
sorted prepared RUN inventory.  This module reproduces that behavior exactly and
emits explicit provenance rather than assuming that list index ``0`` means
``RUN_1``.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import yaml


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_number(name: str) -> int:
    if not name.startswith("RUN_"):
        raise ValueError(f"invalid RUN name {name!r}")
    return int(name.removeprefix("RUN_"))


def normalize_run_inventory(names: Iterable[str]) -> tuple[str, ...]:
    """Return unique RUN names in release25 numeric order."""

    runs = tuple(sorted({str(name) for name in names}, key=_run_number))
    if not runs:
        raise ValueError("RUN inventory is empty")
    return runs


def inventory_from_prepared_inputs(path: str | Path) -> tuple[str, ...]:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return normalize_run_inventory(
        file.stem for file in directory.iterdir()
        if file.is_file() and file.suffix == ".pt" and file.stem.startswith("RUN_")
    )


def inventory_from_raw_dataset(path: str | Path) -> tuple[str, ...]:
    """Reconstruct the release25 prepared inventory from raw simulation outputs.

    The historical release25 preparation routine only discovered RUN directories
    containing ``pflotran.h5``.  ``permeability.h5`` alone therefore does not prove
    that a RUN belonged to the historical prepared dataset.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    return normalize_run_inventory(
        folder.name for folder in root.iterdir()
        if folder.is_dir() and folder.name.startswith("RUN_") and (folder / "pflotran.h5").is_file()
    )


def _indices(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, int) for v in value):
        return tuple(value)
    raise ValueError("order_data entries must be integer indices or non-empty integer lists")


def _map_indices(inventory: tuple[str, ...], value) -> tuple[str, ...]:
    result = []
    for index in _indices(value):
        if index < 0 or index >= len(inventory):
            raise IndexError(f"order_data index {index} is outside RUN inventory of length {len(inventory)}")
        result.append(inventory[index])
    return tuple(result)


@dataclass(frozen=True)
class ResolvedTrainingSplit:
    status: str
    inventory_source: str
    inventory: tuple[str, ...]
    train_runs: tuple[str, ...]
    validation_runs: tuple[str, ...]
    test_runs: tuple[str, ...]
    metadata_path: str
    metadata_sha256: str
    order_data: object

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "status": self.status,
            "inventory_source": self.inventory_source,
            "inventory": list(self.inventory),
            "train_runs": list(self.train_runs),
            "validation_runs": list(self.validation_runs),
            "test_runs": list(self.test_runs),
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "order_data": self.order_data,
            "interpretation": (
                "RUN membership is resolved by applying release25 order_data indices to the "
                "numeric prepared/raw RUN inventory. It is not inferred from RUN numbers."
            ),
        }


def resolve_release25_training_split(
    command_line_arguments_yaml: str | Path,
    *,
    prepared_inputs_dir: str | Path | None = None,
    raw_dataset_root: str | Path | None = None,
) -> ResolvedTrainingSplit:
    """Resolve train/validation/test membership for one release25 model.

    Prefer the original prepared ``Inputs`` directory.  If it is unavailable,
    the exact raw dataset can reconstruct the historical inventory because the
    release25 preparation routine sorted RUN folders containing ``pflotran.h5``.
    Insufficient evidence raises rather than guessing.
    """

    metadata = Path(command_line_arguments_yaml).expanduser().resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    payload = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "order_data" not in payload:
        raise ValueError("command_line_arguments.yaml must contain order_data")
    order_data = payload["order_data"]
    if not isinstance(order_data, (list, tuple)) or len(order_data) < 2:
        raise ValueError("order_data must contain at least training and validation entries")

    if prepared_inputs_dir is not None:
        inventory = inventory_from_prepared_inputs(prepared_inputs_dir)
        source = f"prepared_inputs:{Path(prepared_inputs_dir).expanduser().resolve()}"
    elif raw_dataset_root is not None:
        inventory = inventory_from_raw_dataset(raw_dataset_root)
        source = f"raw_release25_inventory:{Path(raw_dataset_root).expanduser().resolve()}"
    else:
        raise ValueError(
            "training membership is unresolved without either the original prepared Inputs directory "
            "or the exact raw dataset root"
        )

    train = _map_indices(inventory, order_data[0])
    validation = _map_indices(inventory, order_data[1])
    test = _map_indices(inventory, order_data[2]) if len(order_data) > 2 else ()
    return ResolvedTrainingSplit(
        status="resolved_from_release25_metadata_and_inventory",
        inventory_source=source,
        inventory=inventory,
        train_runs=train,
        validation_runs=validation,
        test_runs=test,
        metadata_path=str(metadata),
        metadata_sha256=_sha256_file(metadata),
        order_data=order_data,
    )
