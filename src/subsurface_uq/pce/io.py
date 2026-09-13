from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .workflow import PCEProofOfConceptResult

PCE_RESULT_SCHEMA_VERSION = 1


def save_pce_proof_of_concept_result(
    result: PCEProofOfConceptResult,
    destination: str | Path,
    *,
    metadata: Mapping[str, object],
) -> tuple[Path, Path]:
    """Save PCE training/validation pairs, coefficients, and metadata."""

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    run_metadata = dict(metadata)
    run_metadata["result_schema_version"] = PCE_RESULT_SCHEMA_VERSION
    run_metadata["pce"] = result.regressor.metadata
    run_metadata["diagnostics"] = result.diagnostics.as_dict()
    run_metadata["n_train"] = int(result.train_coordinates.shape[0])
    run_metadata["n_validation"] = int(result.validation_coordinates.shape[0])
    metadata_json = json.dumps(run_metadata, indent=2, sort_keys=True)

    np.savez_compressed(
        target,
        schema_version=np.asarray(PCE_RESULT_SCHEMA_VERSION, dtype=np.int64),
        train_coordinates=np.asarray(result.train_coordinates, dtype=np.float64),
        train_qoi=np.asarray(result.train_qoi, dtype=np.float64),
        validation_coordinates=np.asarray(
            result.validation_coordinates,
            dtype=np.float64,
        ),
        validation_qoi=np.asarray(result.validation_qoi, dtype=np.float64),
        validation_prediction=np.asarray(
            result.validation_prediction,
            dtype=np.float64,
        ),
        multi_indices=np.asarray(result.regressor.multi_indices_, dtype=np.int64),
        coefficients=np.asarray(result.regressor.coefficients_, dtype=np.float64),
        metadata_json=np.asarray(metadata_json),
    )
    sidecar = target.with_suffix(".metadata.json")
    sidecar.write_text(metadata_json + "\n", encoding="utf-8")
    return target, sidecar
