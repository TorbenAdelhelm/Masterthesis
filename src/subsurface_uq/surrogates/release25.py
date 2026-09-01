from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .base import BaseTemperatureSurrogate

Array = np.ndarray


@dataclass
class Release25Surrogate(BaseTemperatureSurrogate):
    """Thin UQ bridge around a verified deterministic release25 adapter.

    ``adapter`` must expose ``predict(permeability_tensor, fixed_inputs)`` and
    return a mapping containing the physical ``"temperature"`` field. This is
    intentionally compatible with the verified ``Release25Adapter`` developed
    in the predecessor prototype without copying release25 source code here.

    The wrapper performs no alternative streamline calculation and therefore
    preserves the deterministic reference path used for Monte Carlo.
    """

    adapter: Any
    fixed_inputs: Any
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    def predict_temperature(self, permeability: Array) -> Array:
        values = np.asarray(permeability)
        if values.ndim != 2:
            raise ValueError(f"permeability must have shape [H,W], got {values.shape}")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("permeability must contain finite positive values")

        k = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        result = self.adapter.predict(k, self.fixed_inputs)
        if "temperature" not in result:
            raise KeyError("release25 adapter result does not contain 'temperature'")
        temperature = result["temperature"]
        if isinstance(temperature, torch.Tensor):
            output = temperature.detach().cpu().numpy()
        else:
            output = np.asarray(temperature)

        while output.ndim > 2 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2:
            raise ValueError(
                "release25 adapter must expose one physical 2-D temperature field; "
                f"got {output.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("release25 temperature prediction contains non-finite values")
        return output.astype(np.float32, copy=False)
