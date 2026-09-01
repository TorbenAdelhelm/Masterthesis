from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np

from ..surrogates.release25_runtime import PreparedLGCNNScenario


def load_prepared_permeability_ensemble(
    dataset_dir: str | Path,
    run_ids: Iterable[str],
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Load physical permeability fields from prepared release25 ``pki`` samples.

    The prepared tensors are reverse-normalized using their own ``info.yaml``.
    All selected fields must have the same spatial shape. The result has shape
    ``[N,H,W]`` and can be passed directly to ``EmpiricalPermeabilitySampler``.
    """

    fields: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for run_id in run_ids:
        scenario = PreparedLGCNNScenario.from_prepared_pki(
            dataset_dir, str(run_id), device=device
        )
        field = scenario.original_permeability.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        if expected_shape is None:
            expected_shape = tuple(int(value) for value in field.shape)
        elif tuple(field.shape) != expected_shape:
            raise ValueError(
                f"prepared permeability {run_id!r} has shape {field.shape}, "
                f"expected {expected_shape}"
            )
        fields.append(field)
    if not fields:
        raise ValueError("run_ids must contain at least one prepared datapoint")
    return np.stack(fields, axis=0)
