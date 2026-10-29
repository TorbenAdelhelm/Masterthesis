from math import comb

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

from subsurface_uq.pce import PolynomialChaosRegressor, total_degree_indices, evaluate_orthonormal_hermite
from subsurface_uq.pce import iid_gaussian_design, latin_hypercube_gaussian_design, run_gaussian_pce, CoordinateQoIEvaluator
from subsurface_uq.surrogates.base import BaseTemperatureSurrogate
from subsurface_uq.pce import MeanTemperatureAnomaly


def test_total_degree_indices_scale_with_admissible_terms_in_twenty_dimensions():
    indices = total_degree_indices(20, 3)
    assert indices.shape == (comb(23, 3), 20)
    assert len({tuple(row) for row in indices}) == len(indices)
    assert np.all(indices.sum(axis=1) <= 3)


def test_hermite_basis_is_orthonormal_and_gaussian_regression_recovers_moments():
    nodes, weights = hermegauss(6)
    indices = total_degree_indices(1, 4)
    basis = evaluate_orthonormal_hermite(nodes[:, None], indices)
    np.testing.assert_allclose(basis.T @ (weights[:, None] * basis) / np.sqrt(2 * np.pi), np.eye(5), atol=1e-13)
    x = latin_hypercube_gaussian_design(80, 2, seed=3)
    q = 4. + 2*x[:, 0] + 3*(x[:, 1]**2 - 1) + .5*x[:, 0]*x[:, 1]
    regressor = PolynomialChaosRegressor(2, 2, basis_family="hermite").fit(x, q)
    np.testing.assert_allclose(regressor.predict(x), q, atol=1e-12)
    np.testing.assert_allclose(regressor.mean, 4., atol=1e-12)
    np.testing.assert_allclose(regressor.variance, 4. + 18. + .25, atol=1e-12)
    assert regressor.metadata["coordinate_distribution"] == "iid_standard_normal"


def test_gaussian_pce_pipeline_uses_independent_gaussian_validation():
    class FieldMap:
        dimension = 2
        field_shape = (1, 1)
        def map_coordinates(self, coordinates):
            return np.exp(np.asarray(coordinates)[:, 0, None, None])
    class Surrogate(BaseTemperatureSurrogate):
        def predict_temperature(self, permeability):
            return 2. + np.log(permeability) + np.log(permeability)**2
    evaluator = CoordinateQoIEvaluator(FieldMap(), Surrogate(), MeanTemperatureAnomaly(0.))
    result = run_gaussian_pce(evaluator, degree=2, n_train=40, n_validation=50, train_seed=4, validation_seed=5)
    np.testing.assert_allclose(result.validation_prediction, result.validation_qoi, atol=1e-12)
    np.testing.assert_allclose(result.regressor.mean, 3., atol=1e-12)
    np.testing.assert_allclose(result.regressor.variance, 3., atol=1e-12)
    np.testing.assert_array_equal(result.validation_coordinates, iid_gaussian_design(50, 2, seed=5))
