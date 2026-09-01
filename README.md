# Master Thesis — Subsurface UQ

This repository contains the thesis-specific uncertainty-quantification (UQ)
framework around deterministic subsurface heat-plume surrogate models.

## Phase A: empirical Monte Carlo baseline

The first implementation deliberately keeps the UQ core small and model-agnostic:

```text
EmpiricalPermeabilitySampler
        -> TemperatureSurrogate
        -> MonteCarloRunner
        -> OnlineFieldStatistics
```

The baseline treats an existing ensemble of permeability fields as uncertain
realizations. This allows propagation, statistics, batching, validation and
later QoI layers to be developed before a conditional geostatistical sampler is
introduced.

### Design rules

- The original `Heat-Plume-Prediction` release25 implementation remains an
  external deterministic reference; its source is not copied into this repository.
- `VampireMan` remains an external data-generation/PFLOTRAN reference.
- PFLOTRAN is not a runtime dependency of surrogate inference or Monte Carlo UQ.
- Monte Carlo is the baseline propagation method. First-order/JVP propagation
  and conditional kriging are intentionally not part of the Phase-A core.
- Scientific data, model checkpoints and generated outputs are not committed.

## Installation

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Real release25 + DaRUS execution

The repository now contains an executable integration layer for the published
LGCNN pipeline:

```text
physical p,k,i
      -> pretrained Step-1 CNN
      -> physical vx,vy
      -> original release25 Step-2 streamline solver
      -> [i,vx,vy,s,k,s_outer]
      -> pretrained Step-3 CNN
      -> physical temperature
      -> MonteCarloRunner
```

`Release25Runtime` loads the standard pretrained model folders from their
`model.pt`, `info.yaml`, and `HPS_options.yaml` files and dynamically imports the
network definitions and Step-2 routine from an external release25 checkout.
`PreparedLGCNNScenario` reverse-normalizes a prepared DaRUS `pki` sample and
uses its pressure and material/heat-pump fields as the fixed scenario.

After downloading/extracting the external assets, a single deterministic smoke
run is:

```bash
subsurface-uq-release25 \
  --release25-repo external/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_0 \
  --output run_output/release25_single.npz
```

An empirical Monte Carlo run can use physical permeability fields from several
prepared datapoints while keeping pressure and heat-pump positions fixed:

```bash
subsurface-uq-release25 \
  --release25-repo external/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_0 \
  --permeability-run-ids RUN_0,RUN_1,RUN_2 \
  --device cuda \
  --output run_output/release25_empirical_mc.npz
```

See `docs/release25_darus.md` for the required DaRUS assets, directory layout,
and model-loading contract.

## Minimal model-agnostic use

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

## Validation guarantees

Normal CI covers:

- empirical sample order and batching;
- `.npy` empirical-field loading;
- online Welford statistics against direct NumPy statistics;
- zero variance for identical realizations;
- Monte-Carlo invariance to sampler batch size;
- delegation through `Release25Surrogate`;
- release25 normalization round-trips;
- physical recovery of a prepared `pki` scenario;
- standard-model reconstruction from release25 metadata using a small fixture;
- execution of the complete CNN1 -> Step 2 -> CNN3 adapter contract with a
  deterministic lightweight Step-2 fixture.

The large published DaRUS model/data archives are not downloaded in CI. A local
smoke test with those assets is therefore the final acceptance test for exact
numerical inference with the pretrained networks.

## Next phases

After the real-model smoke test, the next implementation layers are an
extensible QoI framework and a borehole-conditioned geostatistical sampler.
Advanced propagation approaches such as JVP/first-order Gaussian propagation
remain optional comparison methods rather than baseline dependencies.
