from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def save_monte_carlo_result(
    result: Any,
    destination: str | Path,
    *,
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    """Save Monte Carlo fields plus human- and machine-readable run metadata.

    Metadata is stored twice: as JSON text inside the NPZ archive and as a
    neighboring ``*.metadata.json`` sidecar. ``sample_count`` is always set from
    the realized Monte Carlo result so truncated/requested runs cannot report an
    inconsistent count.
    """

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    run_metadata = dict(metadata)
    run_metadata["sample_count"] = int(result.count)
    if getattr(result, "background_temperature", None) is not None:
        run_metadata["background_temperature"] = float(result.background_temperature)
    thresholds = tuple(float(value) for value in getattr(result, "exceedance_thresholds", ()))
    if thresholds:
        run_metadata["exceedance_thresholds_delta_temperature"] = list(thresholds)
    metadata_json = json.dumps(run_metadata, indent=2, sort_keys=True)

    payload: dict[str, np.ndarray] = {
        "count": np.asarray(result.count, dtype=np.int64),
        "mean": np.asarray(result.mean),
        "variance": np.asarray(result.variance),
        "std": np.asarray(result.std),
        "minimum": np.asarray(result.minimum),
        "maximum": np.asarray(result.maximum),
        "metadata_json": np.asarray(metadata_json),
    }
    if result.samples is not None:
        payload["samples"] = np.asarray(result.samples)

    probabilities = getattr(result, "exceedance_probabilities", None)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.ndim != 3:
            raise ValueError(
                "exceedance probabilities must have shape [K,H,W], "
                f"got {probabilities.shape}"
            )
        if probabilities.shape[0] != len(thresholds):
            raise ValueError(
                "number of exceedance probability maps does not match thresholds"
            )
        payload["exceedance_thresholds"] = np.asarray(thresholds, dtype=np.float64)
        payload["exceedance_probabilities"] = probabilities

    np.savez_compressed(target, **payload)
    sidecar = target.with_suffix(".metadata.json")
    sidecar.write_text(metadata_json + "\n", encoding="utf-8")
    return target, sidecar
