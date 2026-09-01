import numpy as np

from subsurface_uq.propagation import MonteCarloRunner
from subsurface_uq.sampling import EmpiricalPermeabilitySampler
from subsurface_uq.surrogates import CallableTemperatureSurrogate


def _ensemble():
    rng = np.random.default_rng(123)
    return rng.uniform(1.0e-11, 5.0e-9, size=(9, 5, 7)).astype(np.float32)


def _surrogate():
    return CallableTemperatureSurrogate(lambda k: 10.0 - 2.5e8 * k)


def test_monte_carlo_is_invariant_to_sampler_batch_size():
    fields = _ensemble()
    a = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=1), _surrogate()
    ).run(store_all=True)
    b = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=4), _surrogate()
    ).run(store_all=True)

    assert a.count == b.count == len(fields)
    np.testing.assert_allclose(a.mean, b.mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.variance, b.variance, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(a.samples, b.samples, rtol=0, atol=0)


def test_repeated_identical_permeability_has_zero_output_variance():
    field = np.full((5, 7), 2.0e-10, dtype=np.float32)
    fields = np.repeat(field[None, ...], 8, axis=0)
    result = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=3), _surrogate()
    ).run()
    np.testing.assert_array_equal(result.variance, np.zeros_like(result.variance))
    np.testing.assert_array_equal(result.std, np.zeros_like(result.std))


def test_n_samples_truncates_ensemble_without_changing_order():
    fields = _ensemble()
    result = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=4), _surrogate()
    ).run(n_samples=5, store_all=True)
    expected = np.stack([_surrogate().predict_temperature(k) for k in fields[:5]])
    assert result.count == 5
    np.testing.assert_allclose(result.samples, expected)
