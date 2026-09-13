import numpy as np

from subsurface_uq.pce import (
    CoordinateQoIEvaluator,
    MeanTemperatureAnomaly,
    PolynomialChaosRegressor,
    evaluate_orthonormal_legendre,
    iid_uniform_design,
    latin_hypercube_uniform_design,
    run_uniform_pce_proof_of_concept,
    total_degree_indices,
)
from subsurface_uq.surrogates import CallableTemperatureSurrogate


def test_total_degree_basis_size_and_ordering():
    indices = total_degree_indices(dimension=2, degree=3)

    assert indices.shape == (10, 2)
    np.testing.assert_array_equal(indices[0], [0, 0])
    assert np.all(np.sum(indices, axis=1) <= 3)
    assert np.all(np.diff(np.sum(indices, axis=1)) >= 0)


def test_legendre_regression_recovers_exact_polynomial_and_moments():
    regressor = PolynomialChaosRegressor(dimension=2, degree=2)
    coordinates = latin_hypercube_uniform_design(30, 2, seed=123)
    design = evaluate_orthonormal_legendre(coordinates, regressor.multi_indices_)
    true_coefficients = np.asarray([1.5, -0.7, 0.2, 0.4, -0.3, 0.1])
    targets = design @ true_coefficients

    regressor.fit(coordinates, targets)

    np.testing.assert_allclose(
        regressor.coefficients_,
        true_coefficients,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(regressor.mean, true_coefficients[0], atol=1e-11)
    np.testing.assert_allclose(
        regressor.variance,
        np.sum(true_coefficients[1:] ** 2),
        rtol=1e-11,
        atol=1e-11,
    )
    assert regressor.rank_ == regressor.basis_size
    assert regressor.training_rmse_ < 1e-11


def test_uniform_designs_are_reproducible_and_bounded():
    iid_first = iid_uniform_design(20, 3, seed=7)
    iid_second = iid_uniform_design(20, 3, seed=7)
    lhs_first = latin_hypercube_uniform_design(20, 3, seed=8)
    lhs_second = latin_hypercube_uniform_design(20, 3, seed=8)

    np.testing.assert_array_equal(iid_first, iid_second)
    np.testing.assert_array_equal(lhs_first, lhs_second)
    assert np.all(np.abs(iid_first) <= 1.0)
    assert np.all(np.abs(lhs_first) <= 1.0)


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


def _toy_temperature(permeability):
    x = float(permeability[0, 0] - 2.0)
    y = float(permeability[0, 1] - 2.0)
    value = 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * y + 5.0 * x**2
    return np.full((1, 2), value, dtype=np.float32)


def test_end_to_end_coordinate_qoi_pce_workflow_recovers_quadratic_map():
    evaluator = CoordinateQoIEvaluator(
        field_map=_ToyCoordinateMap(),
        surrogate=CallableTemperatureSurrogate(_toy_temperature),
        functional=MeanTemperatureAnomaly(background_temperature=0.0),
        batch_size=4,
    )

    result = run_uniform_pce_proof_of_concept(
        evaluator,
        degree=2,
        n_train=24,
        n_validation=40,
        train_seed=101,
        validation_seed=202,
    )

    assert result.regressor.basis_size == 6
    assert result.train_coordinates.shape == (24, 2)
    assert result.validation_coordinates.shape == (40, 2)
    assert result.diagnostics.rmse < 2e-6
    assert result.diagnostics.max_abs_error < 5e-6
    assert result.diagnostics.q2 > 0.999999999
    np.testing.assert_allclose(
        result.validation_prediction,
        result.validation_qoi,
        rtol=2e-6,
        atol=2e-6,
    )


def test_regression_rejects_undersampled_basis():
    regressor = PolynomialChaosRegressor(dimension=2, degree=3)
    coordinates = iid_uniform_design(9, 2, seed=1)
    targets = np.zeros(9)

    try:
        regressor.fit(coordinates, targets)
    except ValueError as exc:
        assert "at least as many samples as basis terms" in str(exc)
    else:
        raise AssertionError("undersampled regression should raise ValueError")
