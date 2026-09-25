from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

Array = np.ndarray

RELEASE25_INITIAL_TIME_GROUP = "   0 Time  0.00000E+00 y"
RELEASE25_PERMEABILITY_DATASET = "Permeability X [m^2]"


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "DaRUS/PFLOTRAN HDF5 loading requires h5py; install "
            "'subsurface-uq[geostat]'"
        ) from exc
    return h5py


def load_pflotran_permeability_h5(
    path: str | Path,
    *,
    shape: tuple[int, ...] | None = None,
    time_group: str = RELEASE25_INITIAL_TIME_GROUP,
    dataset_name: str = RELEASE25_PERMEABILITY_DATASET,
) -> Array:
    """Load the physical permeability field from one release25 PFLOTRAN HDF5 file.

    The group and dataset names match the release25 raw-data loader. When the
    HDF5 dataset is flattened, ``shape`` must be supplied from the dataset
    settings. Singleton dimensions are removed, yielding the 2-D field consumed
    by the release25 preprocessing path.
    """

    h5py = _require_h5py()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as handle:
        if time_group not in handle:
            raise KeyError(f"time group {time_group!r} not found in {source}")
        group = handle[time_group]
        if dataset_name not in group:
            raise KeyError(f"dataset {dataset_name!r} not found in {source}:{time_group}")
        values = np.asarray(group[dataset_name], dtype=np.float64)

    if shape is not None:
        expected = int(np.prod(shape, dtype=np.int64))
        if values.size != expected:
            raise ValueError(
                f"{source} contains {values.size} permeability values, expected {expected}"
            )
        values = values.reshape(tuple(int(v) for v in shape))
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(
            f"permeability field must reduce to 2-D, got shape {values.shape}; "
            "provide the raw grid shape when the HDF5 dataset is flattened"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("raw permeability must contain finite positive values")
    return values.astype(np.float32, copy=False)


def _raw_grid_shape(settings: dict, *, cell_size_m: float) -> tuple[int, ...]:
    try:
        sizes = settings["grid"]["size [m]"]
    except (KeyError, TypeError) as exc:
        raise KeyError("settings.yaml must contain grid -> 'size [m]'") from exc
    sizes = tuple(float(v) for v in sizes)
    if len(sizes) not in {2, 3}:
        raise ValueError("grid size [m] must contain two or three physical dimensions")
    dims = tuple(int(round(value / cell_size_m)) for value in sizes)
    if any(value <= 0 for value in dims):
        raise ValueError("settings.yaml produced a non-positive grid dimension")
    if len(dims) == 2:
        dims = (*dims, 1)
    return dims


def load_release25_raw_permeability_dataset(
    dataset_root: str | Path,
    *,
    cell_size_m: float = 5.0,
) -> tuple[Array, tuple[str, ...]]:
    """Load all same-shape RUN_*/pflotran.h5 permeability fields from DaRUS raw data.

    This mirrors the release25 raw-data convention used for DARUS-5065: grid
    dimensions are recovered from ``settings.yaml`` and permeability is read
    from the initial-time ``Permeability X [m^2]`` dataset.
    """

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    cell_size_m = float(cell_size_m)
    if not np.isfinite(cell_size_m) or cell_size_m <= 0.0:
        raise ValueError("cell_size_m must be finite and positive")
    settings_path = root / "settings.yaml"
    if not settings_path.is_file():
        raise FileNotFoundError(settings_path)
    with settings_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle)
    if not isinstance(settings, dict):
        raise ValueError("settings.yaml must contain a mapping")
    shape = _raw_grid_shape(settings, cell_size_m=cell_size_m)

    runs: list[tuple[int, str, Path]] = []
    for folder in root.iterdir():
        if not folder.is_dir() or not folder.name.startswith("RUN_"):
            continue
        h5_path = folder / "pflotran.h5"
        if not h5_path.is_file():
            continue
        try:
            run_number = int(folder.name.removeprefix("RUN_"))
        except ValueError:
            continue
        runs.append((run_number, folder.name, h5_path))
    runs.sort(key=lambda item: item[0])
    if not runs:
        raise ValueError(f"no RUN_*/pflotran.h5 files found below {root}")

    fields = [load_pflotran_permeability_h5(path, shape=shape) for _, _, path in runs]
    field_shapes = {field.shape for field in fields}
    if len(field_shapes) != 1:
        raise ValueError(
            "raw runs have different permeability shapes; calibrate training and scaling "
            "datasets separately"
        )
    return np.stack(fields, axis=0), tuple(name for _, name, _ in runs)
