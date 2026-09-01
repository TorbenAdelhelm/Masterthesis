import numpy as np
import torch

from subsurface_uq.surrogates import Release25Surrogate


class FakeRelease25Adapter:
    def __init__(self):
        self.calls = 0
        self.fixed_seen = None

    def predict(self, permeability, fixed_inputs):
        self.calls += 1
        self.fixed_seen = fixed_inputs
        return {
            "velocity": torch.stack([permeability, permeability]),
            "streamlines": torch.stack([permeability, permeability]),
            "temperature": (10.0 - permeability / 1.0e-10).unsqueeze(0),
        }


def test_release25_surrogate_delegates_to_verified_adapter_exactly():
    adapter = FakeRelease25Adapter()
    fixed = object()
    surrogate = Release25Surrogate(adapter=adapter, fixed_inputs=fixed)
    k = np.full((4, 6), 2.0e-10, dtype=np.float32)

    temperature = surrogate.predict_temperature(k)

    assert adapter.calls == 1
    assert adapter.fixed_seen is fixed
    np.testing.assert_allclose(temperature, np.full((4, 6), 8.0, dtype=np.float32))
