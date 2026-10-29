from types import SimpleNamespace

import numpy as np

from subsurface_uq.pce.qoi import MeanTemperatureAnomaly
from subsurface_uq.pce.workflow import CoordinateQoIEvaluator


class _ContextRecorder:
    def __init__(self):
        self.calls = []

    def set_batch_context(self, entries):
        self.calls.append(entries)


class _FieldMap:
    dimension = 2
    field_shape = (2, 3)

    def map_coordinates(self, coordinates):
        coordinates = np.asarray(coordinates)
        return np.ones((coordinates.shape[0], *self.field_shape), dtype=np.float32)

    def coordinates_to_offsets(self, coordinates):
        return np.asarray(coordinates, dtype=float) + np.array([10.0, 20.0])


class _Surrogate:
    def __init__(self, recorder):
        self.adapter = SimpleNamespace(_make_streamlines=recorder)

    def predict_temperature_batch(self, permeability):
        return np.asarray(permeability, dtype=np.float32) + 10.6


def test_coordinate_evaluator_attaches_phase_xi_and_perlin_offsets():
    recorder = _ContextRecorder()
    evaluator = CoordinateQoIEvaluator(
        field_map=_FieldMap(),
        surrogate=_Surrogate(recorder),
        functional=MeanTemperatureAnomaly(background_temperature=10.6),
        batch_size=2,
    )
    coordinates = np.array([[0.1, -0.2], [0.3, 0.4]])

    values = evaluator.evaluate(coordinates, phase="train")

    np.testing.assert_allclose(values, np.ones(2), rtol=1e-6, atol=1e-6)
    assert len(recorder.calls) == 1
    entries = recorder.calls[0]
    assert entries[0]["phase"] == "train"
    assert entries[0]["design_index"] == 1
    assert entries[0]["xi"] == [0.1, -0.2]
    assert entries[0]["perlin_offset"] == [10.1, 19.8]
    assert entries[1]["design_index"] == 2
