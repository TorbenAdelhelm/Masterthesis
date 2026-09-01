from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

Array = np.ndarray


def _select_payload(payload: Any, key: str | None, source: Path) -> Any:
    if isinstance(payload, dict):
        if key is not None:
            if key not in payload:
                raise KeyError(f"{key!r} is not present in {source}")
            return payload[key]
        if len(payload) == 1:
            return next(iter(payload.values()))
        raise ValueError(
            f"{source} contains multiple arrays/tensors; specify an explicit key"
        )
    if key is not None:
        raise ValueError(f"a key was supplied but {source} does not contain a mapping")
    return payload


def load_empirical_fields(path: str | Path, *, key: str | None = None) -> Array:
    """Load a permeability ensemble as ``[N, H, W]``.

    Supported formats are NumPy ``.npy``/``.npz`` and Torch ``.pt``/``.pth``.
    For mapping-like files, ``key`` selects the ensemble. If no key is given,
    the file must contain exactly one array/tensor.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    suffix = source.suffix.lower()
    if suffix == ".npy":
        payload: Any = np.load(source, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as archive:
            payload = {name: archive[name] for name in archive.files}
    elif suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
    else:
        raise ValueError(
            f"unsupported empirical-field format {source.suffix!r}; "
            "expected .npy, .npz, .pt or .pth"
        )

    selected = _select_payload(payload, key, source)
    if isinstance(selected, torch.Tensor):
        selected = selected.detach().cpu().numpy()
    return np.asarray(selected)


@dataclass
class EmpiricalPermeabilitySampler:
    """Deterministic sampler over an existing permeability ensemble.

    Samples retain their original order. ``batch_size`` affects only how many
    fields are yielded at once; it must not affect Monte-Carlo statistics.
    """

    fields: Array
    batch_size: int = 1
    require_positive: bool = True
    copy_batches: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.fields)
        if values.ndim != 3:
            raise ValueError(
                f"fields must have shape [N,H,W], got {tuple(values.shape)}"
            )
        if values.shape[0] == 0:
            raise ValueError("the empirical ensemble must contain at least one field")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.all(np.isfinite(values)):
            raise ValueError("permeability fields must contain only finite values")
        if self.require_positive and np.any(values <= 0.0):
            raise ValueError("permeability fields must be strictly positive")
        self.fields = values.astype(np.float32, copy=False)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        key: str | None = None,
        batch_size: int = 1,
        require_positive: bool = True,
        copy_batches: bool = False,
    ) -> "EmpiricalPermeabilitySampler":
        return cls(
            load_empirical_fields(path, key=key),
            batch_size=batch_size,
            require_positive=require_positive,
            copy_batches=copy_batches,
        )

    @property
    def field_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.fields.shape[1:])

    def __len__(self) -> int:
        return int(self.fields.shape[0])

    def __iter__(self):
        for start in range(0, len(self), self.batch_size):
            batch = self.fields[start : start + self.batch_size]
            yield batch.copy() if self.copy_batches else batch
