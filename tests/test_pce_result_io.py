import json

import numpy as np

from subsurface_uq.pce import (
    CoordinateQoIEvaluator,
    MeanTemperatureAnomaly,
    run_uniform_pce_proof_of_concept,
    save_pce_proof_of_concept_result,
)
from subsurface_uq.surrogates import CallableTemperatureSurrogate


class _ToyCoordinateMap:
    dimension = 2
    field_shape = (1, 2)

    def map_coordinates(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.float64)
        single = coordinates.ndim == 1
        if single:
            coordinates = coordinates[None, :]
        fields = np.empty((coordinates.shape[0], 1, 2), dtype=np.float32)
        fields[:, 0, 0] = 2.0 + coordinates[:, 0]
        fields[:, 0, 1] = 2.0 + coordinates[:, 1]
        return fields[0] if single else fields


def _temperature(permeability):
    x = float(permeability[0, 0] - 2.0)
    y = float(permeability[0, 1] - 2.0)
    value = 1.0 + x + y + x * y
    return np.full((1, 2), value, dtype=np.float32)


def test_pce_result_archive_contains_training_validation_and_metadata(tmp_path):
    evaluator = CoordinateQoIEvaluator(
        field_map=_ToyCoordinateMap(),
        surrogate=CallableTemperatureSurrogate(_temperature),
        functional=MeanTemperatureAnomaly(background_temperature=0.0),
        batch_size=3,
    )
    result = run_uniform_pce_proof_of_concept(
        evaluator,
        degree=2,
        n_train=12,
        n_validation=8,
        train_seed=11,
        validation_seed=12,
    )

    archive, sidecar = save_pce_proof_of_concept_result(
        result,
        tmp_path / "pce.npz",
        metadata={"case": "unit-test"},
    )

    assert archive.exists()
    assert sidecar.exists()
    with np.load(archive, allow_pickle=False) as data:
        assert data["train_coordinates"].shape == (12, 2)
        assert data["validation_coordinates"].shape == (8, 2)
        assert data["coefficients"].shape == (6,)
        metadata = json.loads(str(data["metadata_json"].item()))
    assert metadata["case"] == "unit-test"
    assert metadata["pce"]["basis_size"] == 6
    assert metadata["n_train"] == 12
    assert metadata["n_validation"] == 8
