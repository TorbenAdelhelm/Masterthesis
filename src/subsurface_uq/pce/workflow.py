from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sampling.coordinates import StochasticPermeabilityMap
from ..surrogates.base import TemperatureSurrogate
from .design import iid_uniform_design, latin_hypercube_uniform_design
from .qoi import TemperatureFunctional
from .regression import PolynomialChaosRegressor

Array = np.ndarray


@dataclass
class CoordinateQoIEvaluator:
    """Evaluate paired stochastic coordinates and scalar temperature QoIs."""

    field_map: StochasticPermeabilityMap
    surrogate: TemperatureSurrogate
    functional: TemperatureFunctional
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(self.batch_size)

    def evaluate(self, coordinates: Array) -> Array:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.ndim == 1:
            coordinates = coordinates[None, :]
        if coordinates.ndim != 2 or coordinates.shape[1] != self.field_map.dimension:
            raise ValueError(
                "coordinates must have shape "
                f"[N,{self.field_map.dimension}], got {coordinates.shape}"
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")

        values: list[Array] = []
        for start in range(0, coordinates.shape[0], self.batch_size):
            stop = min(start + self.batch_size, coordinates.shape[0])
            batch_coordinates = coordinates[start:stop]
            permeability = np.asarray(
                self.field_map.map_coordinates(batch_coordinates)
            )
            expected = (stop - start, *self.field_map.field_shape)
            if permeability.shape != expected:
                raise ValueError(
                    "stochastic permeability map returned an unexpected shape; "
                    f"expected {expected}, got {permeability.shape}"
                )
            temperature = np.asarray(
                self.surrogate.predict_temperature_batch(permeability)
            )
            if temperature.ndim != 3 or temperature.shape[0] != stop - start:
                raise ValueError(
                    "temperature surrogate must return [B,H,W], "
                    f"got {temperature.shape}"
                )
            qoi = np.asarray(self.functional.evaluate_batch(temperature), dtype=np.float64)
            if qoi.shape != (stop - start,):
                raise ValueError(
                    "temperature functional must return one scalar per sample; "
                    f"expected {(stop - start,)}, got {qoi.shape}"
                )
            values.append(qoi)
        return np.concatenate(values, axis=0)


@dataclass(frozen=True)
class PCEDiagnostics:
    rmse: float
    mae: float
    max_abs_error: float
    relative_l2_error: float
    q2: float
    validation_mean: float
    pce_mean: float
    validation_variance: float
    pce_variance: float

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "max_abs_error": self.max_abs_error,
            "relative_l2_error": self.relative_l2_error,
            "q2": self.q2,
            "validation_mean": self.validation_mean,
            "pce_mean": self.pce_mean,
            "validation_variance": self.validation_variance,
            "pce_variance": self.pce_variance,
        }


def pce_diagnostics(
    truth: Array,
    prediction: Array,
    *,
    pce_mean: float,
    pce_variance: float,
) -> PCEDiagnostics:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape:
        raise ValueError("truth and prediction must have matching shape [N]")
    if truth.size < 2:
        raise ValueError("at least two validation samples are required")
    if not (np.all(np.isfinite(truth)) and np.all(np.isfinite(prediction))):
        raise ValueError("truth and prediction must be finite")

    error = prediction - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    max_abs_error = float(np.max(np.abs(error)))
    truth_norm = float(np.linalg.norm(truth))
    relative_l2_error = (
        float(np.linalg.norm(error) / truth_norm)
        if truth_norm > 0.0
        else (0.0 if np.linalg.norm(error) == 0.0 else float("inf"))
    )
    centered = truth - float(np.mean(truth))
    sst = float(np.sum(centered**2))
    sse = float(np.sum(error**2))
    q2 = 1.0 - sse / sst if sst > 0.0 else (1.0 if sse == 0.0 else float("-inf"))

    return PCEDiagnostics(
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs_error,
        relative_l2_error=relative_l2_error,
        q2=float(q2),
        validation_mean=float(np.mean(truth)),
        pce_mean=float(pce_mean),
        validation_variance=float(np.var(truth, ddof=1)),
        pce_variance=float(pce_variance),
    )


@dataclass
class PCEProofOfConceptResult:
    regressor: PolynomialChaosRegressor
    train_coordinates: Array
    train_qoi: Array
    validation_coordinates: Array
    validation_qoi: Array
    validation_prediction: Array
    diagnostics: PCEDiagnostics



def run_uniform_pce_proof_of_concept(
    evaluator: CoordinateQoIEvaluator,
    *,
    degree: int,
    n_train: int,
    n_validation: int,
    train_seed: int,
    validation_seed: int,
) -> PCEProofOfConceptResult:
    """Fit Legendre PCE and validate it against independent surrogate evaluations.

    Training uses a randomized Latin-hypercube design. Validation uses iid
    ``U(-1,1)`` coordinates so the expensive validation ensemble also serves as
    a direct Monte Carlo reference under exactly the same stochastic law.
    """

    regressor = PolynomialChaosRegressor(
        dimension=evaluator.field_map.dimension,
        degree=degree,
    )
    if n_train < regressor.basis_size:
        raise ValueError(
            "n_train must be at least the PCE basis size; "
            f"got n_train={n_train}, basis_size={regressor.basis_size}"
        )
    if n_validation < 2:
        raise ValueError("n_validation must be at least 2")

    train_coordinates = latin_hypercube_uniform_design(
        n_train,
        evaluator.field_map.dimension,
        seed=train_seed,
    )
    train_qoi = evaluator.evaluate(train_coordinates)
    regressor.fit(train_coordinates, train_qoi)

    validation_coordinates = iid_uniform_design(
        n_validation,
        evaluator.field_map.dimension,
        seed=validation_seed,
    )
    validation_qoi = evaluator.evaluate(validation_coordinates)
    validation_prediction = regressor.predict(validation_coordinates)
    diagnostics = pce_diagnostics(
        validation_qoi,
        validation_prediction,
        pce_mean=regressor.mean,
        pce_variance=regressor.variance,
    )

    return PCEProofOfConceptResult(
        regressor=regressor,
        train_coordinates=train_coordinates,
        train_qoi=train_qoi,
        validation_coordinates=validation_coordinates,
        validation_qoi=validation_qoi,
        validation_prediction=validation_prediction,
        diagnostics=diagnostics,
    )
