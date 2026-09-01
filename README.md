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
  --fixed-run-id RUN_1 \
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
  --fixed-run-id RUN_1 \
  --permeability-run-ids RUN_1,RUN_2,RUN_4 \
  --device cuda \
  --output run_output/release25_empirical_mc.npz
```

See `docs/release25_darus.md` for the required DaRUS assets, directory layout,
and model-loading contract.

## LGCNN diagnostic and publication plots

Supplying `--plots-dir` to the release25 runner generates deterministic plots for
the fixed run: velocity x/y/magnitude, central and outer streamline feature
fields, and temperature. It also creates three combined figures: temperature
with central streamline contours, temperature with heat-pump markers, and a
combined temperature/streamline/heat-pump overlay.

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --device cpu \
  --output run_output/release25_single.npz \
  --plots-dir run_output/release25_RUN_1_lgcnn
```

The plotting default `--cell-size-m 5.0` matches the release25 synthetic grid
and expresses spatial axes in kilometres. `--streamline-plot-threshold` controls
which values of the actual rasterized central streamline feature are shown as
contours in the overlays. Heat-pump markers are derived from connected
`Material ID == 2` regions after center alignment to the temperature output.
For other grids, pass the correct cell size explicitly.

## Temperature validation against prepared DaRUS labels

The `subsurface_uq.validation` package compares one physical release25
prediction against a prepared temperature label for the same run. It
reverse-normalizes the stored label from its `info.yaml`, aligns the larger
reference field to the valid-convolution model output by a center crop, and
reports MAE, MSE, RMSE, maximum absolute error, mean bias, absolute-error
percentiles (p50, p90, p95, p99 and p99.9), and the spatial fractions above
0.1 °C, 0.5 °C and 1.0 °C absolute error.

After extracting a prepared `inputs_pki outputs_t` dataset to, for example,
`data/prepared_pki_temperature`, run:

```bash
subsurface-uq-validate-temperature \
  --prediction run_output/release25_single.npz \
  --reference-dir data/prepared_pki_temperature \
  --run-id RUN_1 \
  --output run_output/release25_RUN_1_validation.npz \
  --plots-dir run_output/release25_RUN_1_plots
```

If console scripts are not available in the active shell, the equivalent module
command is:

```bash
python -m subsurface_uq.validation.cli \
  --prediction run_output/release25_single.npz \
  --reference-dir data/prepared_pki_temperature \
  --run-id RUN_1 \
  --output run_output/release25_RUN_1_validation.npz \
  --plots-dir run_output/release25_RUN_1_plots
```

The output archive stores the aligned prediction/reference fields, signed and
absolute error fields, and scalar metrics. `--plots-dir` additionally creates
four PNG maps: predicted temperature, reference temperature, signed error, and
absolute error. Use `--alignment strict` when comparing two already-aligned
deterministic runtime outputs; the default `center-crop-reference` is intended
for comparison with the larger prepared DaRUS temperature label.

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
surrogate = CallableTemperatureSurrogate(lambda field: 10.0 - field / 1.0e-10)
result = MonteCarloRunner(EmpiricalPermeabilitySampler(k, batch_size=2), surrogate).run()

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
  deterministic lightweight Step-2 fixture;
- release25 diagnostic-map and combined-overlay generation;
- reverse-normalization, center-crop alignment, scalar metrics, spatial error
  percentiles/threshold fractions, and diagnostic-map generation for prepared
  temperature-reference validation.

The large published DaRUS model/data archives are not downloaded in CI. A local
smoke test with those assets remains the acceptance test for numerical inference
with the pretrained networks; the temperature-validation CLI then quantifies the
surrogate error against a stored prepared reference label.

## Next phases

After deterministic/reference validation, the next implementation layers are an
extensible QoI framework and a borehole-conditioned geostatistical sampler.
Advanced propagation approaches such as JVP/first-order Gaussian propagation
remain optional comparison methods rather than baseline dependencies.
