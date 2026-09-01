from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from ..surrogates.release25_runtime import Release25Normalizer

Array = np.ndarray


@dataclass(frozen=True)
class TemperatureComparison:
    """Physical-space comparison between one prediction and one reference field."""

    prediction: Array
    reference: Array
    error: Array
    absolute_error: Array
    mae: float
    mse: float
    rmse: float
    max_absolute_error: float
    bias: float

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.prediction.shape)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return value


def _load_torch_tensor(path: Path, *, device: str | torch.device = "cpu") -> torch.Tensor:
    try:
        value = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        value = torch.load(path, map_location=device)
    if isinstance(value, Mapping):
        if "tensor" not in value:
            raise TypeError(f"{path} contains a mapping without key 'tensor'")
        value = value["tensor"]
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _one_run_file(root: Path, directory: str, run_id: str) -> Path:
    candidates = [root / directory / run_id, root / directory / f"{run_id}.pt"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        listing = ", ".join(str(path) for path in existing) if existing else "none"
        raise FileNotFoundError(
            f"expected exactly one {directory} tensor for {run_id!r} below "
            f"{root / directory}; found {listing}"
        )
    return existing[0]


def _temperature_index(info: Mapping[str, Any]) -> int:
    labels = info.get("Labels")
    if not isinstance(labels, Mapping):
        raise KeyError("prepared dataset info.yaml has no Labels mapping")

    exact_names = ("Temperature [C]", "Temperature [°C]")
    for name in exact_names:
        entry = labels.get(name)
        if isinstance(entry, Mapping) and "index" in entry:
            return int(entry["index"])

    candidates: list[int] = []
    for name, entry in labels.items():
        if "temperature" in str(name).lower() and isinstance(entry, Mapping) and "index" in entry:
            candidates.append(int(entry["index"]))
    if len(candidates) != 1:
        raise KeyError(
            "could not identify exactly one temperature label in prepared dataset metadata"
        )
    return candidates[0]


def load_prepared_temperature_label(
    dataset_dir: str | Path,
    run_id: str,
    *,
    device: str | torch.device = "cpu",
) -> Array:
    """Load and reverse-normalize one prepared DaRUS/release25 temperature label.

    ``dataset_dir`` must contain ``info.yaml`` and ``Labels/<RUN_ID>`` (or
    ``Labels/<RUN_ID>.pt``). Returned values are physical temperatures in °C.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    info_path = root / "info.yaml"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)

    info = _load_yaml(info_path)
    label = _load_torch_tensor(_one_run_file(root, "Labels", run_id), device=device)

    if label.ndim == 2:
        label = label.unsqueeze(0)
    if label.ndim != 3:
        raise ValueError(
            f"prepared temperature label must be [H,W] or [C,H,W], got {tuple(label.shape)}"
        )

    physical = Release25Normalizer(info).reverse(label, "Labels")
    index = _temperature_index(info)
    if index >= physical.shape[0]:
        raise ValueError(
            f"temperature metadata refers to channel {index}, label has {physical.shape[0]} channels"
        )
    temperature = physical[index]
    if not torch.isfinite(temperature).all():
        raise ValueError("reference temperature contains non-finite values")
    return temperature.detach().cpu().numpy().astype(np.float32, copy=False)


def load_prediction_field(path: str | Path, *, key: str = "mean") -> Array:
    """Load one 2-D prediction field from a ``.npz`` result or ``.npy`` file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".npz":
        with np.load(source) as archive:
            if key not in archive:
                raise KeyError(
                    f"prediction archive {source} has no key {key!r}; "
                    f"available keys: {sorted(archive.files)}"
                )
            value = archive[key]
    elif source.suffix.lower() == ".npy":
        value = np.load(source)
    else:
        raise ValueError("prediction must be a .npz or .npy file")

    field = np.asarray(value, dtype=np.float64)
    while field.ndim > 2 and field.shape[0] == 1:
        field = field[0]
    if field.ndim != 2:
        raise ValueError(f"prediction field must be 2-D, got {field.shape}")
    if not np.all(np.isfinite(field)):
        raise ValueError("prediction temperature contains non-finite values")
    return field


def center_crop_2d(field: Array, target_shape: tuple[int, int]) -> Array:
    """Return the centered ``target_shape`` window of one 2-D field."""

    values = np.asarray(field)
    if values.ndim != 2:
        raise ValueError(f"center crop expects a 2-D field, got {values.shape}")
    th, tw = (int(target_shape[0]), int(target_shape[1]))
    h, w = values.shape
    if th <= 0 or tw <= 0:
        raise ValueError("target shape must be positive")
    if th > h or tw > w:
        raise ValueError(f"cannot center-crop {values.shape} to {(th, tw)}")
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return values[y0 : y0 + th, x0 : x0 + tw]


def compare_temperature_fields(
    prediction: Array,
    reference: Array,
    *,
    alignment: str = "center-crop-reference",
) -> TemperatureComparison:
    """Compare two physical temperature fields and compute standard errors.

    ``center-crop-reference`` is the release25-friendly default because
    ``UNetNoPad2`` shrinks the predicted spatial field. ``strict`` requires both
    arrays to already have identical shapes.
    """

    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.ndim != 2 or ref.ndim != 2:
        raise ValueError("prediction and reference must both be 2-D")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(ref)):
        raise ValueError("prediction and reference must contain only finite values")

    if alignment == "strict":
        if pred.shape != ref.shape:
            raise ValueError(
                f"strict comparison requires equal shapes, got {pred.shape} and {ref.shape}"
            )
    elif alignment == "center-crop-reference":
        if ref.shape != pred.shape:
            ref = center_crop_2d(ref, pred.shape)
    else:
        raise ValueError("alignment must be 'strict' or 'center-crop-reference'")

    error = pred - ref
    absolute_error = np.abs(error)
    mse = float(np.mean(error * error))
    return TemperatureComparison(
        prediction=pred.astype(np.float32),
        reference=ref.astype(np.float32),
        error=error.astype(np.float32),
        absolute_error=absolute_error.astype(np.float32),
        mae=float(np.mean(absolute_error)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
        max_absolute_error=float(np.max(absolute_error)),
        bias=float(np.mean(error)),
    )


def save_temperature_comparison(
    comparison: TemperatureComparison,
    path: str | Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Store aligned fields and scalar metrics in one compressed NumPy archive."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Array] = {
        "prediction": comparison.prediction,
        "reference": comparison.reference,
        "error": comparison.error,
        "absolute_error": comparison.absolute_error,
        "mae": np.asarray(comparison.mae, dtype=np.float64),
        "mse": np.asarray(comparison.mse, dtype=np.float64),
        "rmse": np.asarray(comparison.rmse, dtype=np.float64),
        "max_absolute_error": np.asarray(comparison.max_absolute_error, dtype=np.float64),
        "bias": np.asarray(comparison.bias, dtype=np.float64),
    }
    if run_id is not None:
        payload["run_id"] = np.asarray(run_id)
    np.savez_compressed(destination, **payload)
    return destination
