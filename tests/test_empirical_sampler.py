import numpy as np

from subsurface_uq.sampling import EmpiricalPermeabilitySampler


def _fields(n=5):
    return np.stack(
        [np.full((3, 4), (i + 1) * 1.0e-10, dtype=np.float32) for i in range(n)]
    )


def test_empirical_sampler_preserves_order_across_batches():
    fields = _fields(5)
    sampler = EmpiricalPermeabilitySampler(fields, batch_size=2)
    rebuilt = np.concatenate(list(sampler), axis=0)
    np.testing.assert_array_equal(rebuilt, fields)
    assert len(sampler) == 5
    assert sampler.field_shape == (3, 4)


def test_empirical_sampler_loads_npy(tmp_path):
    fields = _fields(3)
    path = tmp_path / "fields.npy"
    np.save(path, fields)
    sampler = EmpiricalPermeabilitySampler.from_file(path, batch_size=2)
    np.testing.assert_array_equal(np.concatenate(list(sampler)), fields)
