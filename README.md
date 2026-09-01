# Master Thesis — Subsurface UQ

This repository contains the thesis-specific uncertainty-quantification (UQ) framework around deterministic subsurface heat-plume surrogate models.

## Phase A: empirical Monte Carlo baseline

The first implementation deliberately keeps the UQ core small and model-agnostic:

```text
EmpiricalPermeabilitySampler
        -> TemperatureSurrogate
        -> MonteCarloRunner
        -> OnlineFieldStatistics
```

The baseline treats an existing ensemble of permeability fields as uncertain realizations. This allows the propagation, statistics, batching, validation and later QoI layers to be developed before a conditional geostatistical sampler is introduced.

### Design rules

- The original `Heat-Plume-Prediction` release25 implementation remains an external deterministic reference; its source is not copied into this repository.
- `VampireMan` remains an external data-generation/PFLOTRAN reference.
- `Release25Surrogate` is a thin bridge around an already verified release25 adapter and fixed scenario.
- Monte Carlo is the baseline propagation method. First-order/JVP propagation and conditional kriging are intentionally not part of the Phase-A core.
- Scientific data, model checkpoints and generated outputs are not committed to this repository.

## Installation

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Minimal use

```python
import numpy as np
from subsurface_uq.propagation import MonteCarloRunner
from subsurface_uq.sampling import EmpiricalPermeabilitySampler
from subsurface_uq.surrogates import CallableTemperatureSurrogate

k = np.stack([
    np.full((8, 12), 1.0e-10),
    np.full((8, 12), 2.0e-10),
])

sampler = EmpiricalPermeabilitySampler(k, batch_size=2)
surrogate = CallableTemperatureSurrogate(lambda field: 10.0 - field / 1.0e-10)
result = MonteCarloRunner(sampler, surrogate).run()

print(result.count)
print(result.mean.shape)
print(result.std.shape)
```

For release25, construct the verified deterministic adapter and fixed scenario, then wrap them with `subsurface_uq.surrogates.Release25Surrogate`. The wrapper does not modify the deterministic LGCNN path.

## Current validation guarantees

The automated tests cover:

- empirical sample order and batching;
- `.npy` empirical-field loading;
- online Welford statistics against direct NumPy statistics;
- zero variance for identical realizations;
- Monte-Carlo invariance to sampler batch size;
- exact delegation of a permeability field to a release25-compatible adapter.

## Next phases

The next implementation layers are an extensible QoI framework and a borehole-conditioned geostatistical sampler. Advanced propagation approaches such as JVP/first-order Gaussian propagation remain optional comparison methods rather than baseline dependencies.
