import numpy as np

from subsurface_uq.statistics import OnlineFieldStatistics


def test_online_statistics_match_direct_numpy():
    rng = np.random.default_rng(7)
    fields = rng.normal(size=(11, 4, 6)).astype(np.float32)

    online = OnlineFieldStatistics()
    online.update(fields[:3])
    online.update(fields[3:8])
    online.update(fields[8:])
    result = online.finalize(ddof=1)

    np.testing.assert_allclose(result.mean, fields.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.variance, fields.var(axis=0, ddof=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.minimum, fields.min(axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(result.maximum, fields.max(axis=0), rtol=0, atol=0)


def test_identical_fields_have_zero_variance():
    field = np.arange(12, dtype=np.float32).reshape(3, 4)
    online = OnlineFieldStatistics()
    online.update(np.repeat(field[None, ...], 6, axis=0))
    result = online.finalize()
    np.testing.assert_array_equal(result.mean, field)
    np.testing.assert_array_equal(result.variance, np.zeros_like(field))
    np.testing.assert_array_equal(result.std, np.zeros_like(field))
