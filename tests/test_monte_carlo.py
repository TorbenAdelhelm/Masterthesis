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


def test_online_exceedance_matches_direct_sample_counting_without_store_all():
    fields = np.stack(
        [
            np.full((2, 3), 1.0e-10, dtype=np.float32),
            np.full((2, 3), 2.0e-10, dtype=np.float32),
            np.full((2, 3), 4.0e-10, dtype=np.float32),
            np.full((2, 3), 8.0e-10, dtype=np.float32),
        ]
    )
    surrogate = CallableTemperatureSurrogate(lambda k: 10.0 + k * 1.0e9)
    result = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=3), surrogate
    ).run(
        background_temperature=10.0,
        exceedance_thresholds=(0.1, 0.5),
        store_all=False,
    )

    direct = np.stack([surrogate.predict_temperature(k) for k in fields])
    delta = direct - np.asarray(10.0, dtype=direct.dtype)
    expected = np.stack(
        [
            np.mean(delta >= np.asarray(0.1, dtype=delta.dtype), axis=0),
            np.mean(delta >= np.asarray(0.5, dtype=delta.dtype), axis=0),
        ]
    ).astype(np.float32)

    assert result.samples is None
    assert result.background_temperature == 10.0
    assert result.exceedance_thresholds == (0.1, 0.5)
    np.testing.assert_allclose(result.exceedance_probabilities, expected)


def test_exceedance_probabilities_are_invariant_to_sampler_batch_size():
    fields = _ensemble()
    kwargs = {
        "background_temperature": 10.0,
        "exceedance_thresholds": (-1.0, -0.5),
    }
    a = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=1), _surrogate()
    ).run(**kwargs)
    b = MonteCarloRunner(
        EmpiricalPermeabilitySampler(fields, batch_size=4), _surrogate()
    ).run(**kwargs)
    np.testing.assert_array_equal(a.exceedance_probabilities, b.exceedance_probabilities)
