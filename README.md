# Master Thesis — Subsurface UQ

This repository contains the thesis-specific forward uncertainty-quantification
(UQ) framework around deterministic groundwater heat-plume surrogate models.
The current scientific baseline focuses on uncertainty in the permeability
field while pressure and heat-pump locations are held fixed.

The implementation is designed around interchangeable components:

```text
PermeabilitySampler
        -> TemperatureSurrogate
        -> MonteCarloRunner
        -> TemperatureAccumulator(s)
             |- field statistics
             |- exceedance probabilities
             `- future QoIs
```

The original `Heat-Plume-Prediction` release25 implementation remains the
external deterministic reference, and `VampireMan` / the PFLOTRAN generation
repository remain external data-generation references. Scientific data, model
checkpoints and generated outputs are intentionally not committed.

## Documentation

The mathematical definition of the current pipeline is documented in
[`docs/methodology.md`](docs/methodology.md), including the LGCNN composition,
Monte Carlo estimators, streaming Welford statistics, exceedance probabilities,
the historical Perlin transformation, release25 normalization and output
alignment.

Runtime/data details are in
[`docs/release25_darus.md`](docs/release25_darus.md), and the synthetic Perlin
baseline is documented in
[`docs/release25_perlin_uq.md`](docs/release25_perlin_uq.md).

## Installation

```bash
python -m pip install -e ".[test]"
python -m pytest
```

`noise>=1.2.2` is currently required because the historical synthetic generator
uses `noise.pnoise2`. On Windows this package may require a working MSVC/Windows
SDK build environment.

## Deterministic release25 integration

The real pretrained LGCNN execution path is

```text
physical p,k,i
      -> pretrained CNN1 / Step 1
      -> physical vx,vy
      -> original release25 Step-2 streamline solver
      -> [i,vx,vy,s,k,s_outer]
      -> pretrained CNN3 / Step 3
      -> physical temperature
```

A deterministic smoke run is:

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --device cpu \
  --output run_output/release25_single.npz
```

`--cnn2-dir` is retained as a backwards-compatible CLI name. It denotes the
**second CNN in the LGCNN, i.e. Step 3 / CNN3**.

For a one-sample deterministic diagnostic, the experiment automatically uses
`ddof=0`. For a real Monte Carlo ensemble the default is `ddof=1`; the code does
not report a one-sample sample variance as zero because that estimator is
undefined for `N=1, ddof=1`.

## Empirical Monte Carlo baseline

Existing permeability fields can be propagated while pressure and heat-pump
locations stay fixed to one prepared run:

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --permeability-run-ids RUN_1,RUN_2,RUN_4 \
  --device cpu \
  --output run_output/release25_empirical_mc.npz
```

Alternatively, `--ensemble-file` accepts an `[N,H,W]` permeability array/tensor
stored as `.npy`, `.npz`, `.pt`, or `.pth`.

## Historical Perlin forward UQ

`Release25PerlinPermeabilitySampler` reproduces the historical `perlin_v2`
spatial formula used for the released random-permeability data. For raw Perlin
field `P(x)`,

$$
U(x)=\frac{P(x)-\min P}{\max P-\min P},
$$

and the permeability is generated in base-10 logarithmic space,

$$
K(x)=10^{\log_{10}k_{\min}
+U(x)(\log_{10}k_{\max}-\log_{10}k_{\min})}.
$$

The released synthetic defaults are a `2560 x 2560` grid, `(18,18)` Perlin
frequency and permeability range
`1.0193679918450561e-11 ... 5.09683995922528e-09 m^2`.

A five-sample end-to-end smoke run is:

```bash
python -m subsurface_uq.experiments.release25_perlin \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --n-samples 5 \
  --seed 2907 \
  --device cpu \
  --output run_output/release25_perlin_mc_5.npz \
  --plots-dir run_output/release25_perlin_mc_5_plots
```

The exact initial groundwater temperature in the historical synthetic PFLOTRAN
setup is **10.6 °C** and the injection temperature is `15.6 °C`. Therefore the
Perlin baseline defines

$$
\Delta T = T-10.6\ ^\circ\mathrm C
$$

and by default accumulates the empirical spatial probabilities

$$
\widehat P_\tau(x)=\frac1N\sum_{m=1}^N
\mathbf 1[\Delta T^{(m)}(x)\ge\tau]
$$

for `tau=0.1 °C` and `1.0 °C`. Background and thresholds are configurable.
These statistics are accumulated online; `--store-all` is only needed if the
individual temperature realizations are required.

## Monte Carlo statistics and extensible QoIs

For propagated temperature fields `T^(m)(x)`, the core computes

$$
\hat\mu_T(x)=\frac1N\sum_{m=1}^N T^{(m)}(x)
$$

and

$$
\hat\sigma_T^2(x)=\frac1{N-d}\sum_{m=1}^N
(T^{(m)}(x)-\hat\mu_T(x))^2,
$$

where `d` is `ddof`. Mean, variance, standard deviation and extrema are updated
with a streaming batch-merge form of Welford's algorithm.

Additional quantities of interest use the `TemperatureAccumulator` interface.
An accumulator receives temperature batches through `update(...)` and returns
its final result through `finalize()`. This keeps future monitoring-point,
plume-length, plume-area or other QoIs out of the propagation core.

## Monte Carlo/UQ visualization

A Perlin run with `--plots-dir` creates physical-space maps of:

- mean temperature;
- standard deviation;
- sample range `max-min`;
- mean temperature with standard-deviation contours;
- mean `Delta T`;
- one exceedance-probability map per threshold.

Existing MC archives can be plotted without rerunning the surrogate:

```bash
python -m subsurface_uq.visualization.cli \
  --input run_output/release25_perlin_mc_5.npz \
  --output-dir run_output/release25_perlin_mc_5_plots
```

For older archives that do not store a background temperature, use
`--background-temperature 10.6` for the historical synthetic dataset.

## Deterministic diagnostics and temperature validation

The empirical release25 runner can additionally create velocity, streamline and
temperature diagnostic plots with `--plots-dir`.

A physical prediction can be compared against a prepared reference-temperature
label with:

```bash
python -m subsurface_uq.validation.cli \
  --prediction run_output/release25_single.npz \
  --reference-dir data/prepared_pki_temperature \
  --run-id RUN_1 \
  --output run_output/release25_RUN_1_validation.npz \
  --plots-dir run_output/release25_RUN_1_plots
```

The validation layer reverse-normalizes the reference, center-crops the larger
reference to the valid-convolution prediction shape when requested, and reports
MAE, MSE, RMSE, maximum absolute error, bias and spatial error diagnostics.
Optional overlays regenerate the release25 central streamline feature and
heat-pump locations.

## Reproducibility

New Monte Carlo results use a versioned NPZ schema and write the same run
metadata to a neighboring `.metadata.json` sidecar. Release25 experiment runs
record, where available, the realized sample count, variance `ddof`, sampler
parameters, fixed run, Git SHAs, SHA-256 hashes of the two model checkpoints,
Python/platform and dependency versions, release25 settings, and historical
Perlin generator provenance.

The historical Perlin formula is tested against an independent transcription of
the original generator. The historical generator itself did not effectively use
the DaRUS `seed_id` because its NumPy seeding line was commented out; the thesis
sampler makes the seed explicit for reproducible *new* UQ experiments and does
not claim bitwise recovery of the original training fields.

## Current validation scope

CI uses lightweight fixtures and tests sampler behavior, batch-invariant Monte
Carlo statistics, custom accumulator extensibility, historical Perlin formula,
result metadata, UQ plotting, release25 normalization/model reconstruction,
three-stage adapter execution and temperature validation utilities.

The actual released DaRUS models/data have also been exercised locally with a
real deterministic RUN_1 prediction and prepared reference comparison. This is
not yet the stronger claim of numerical equivalence to a separately executed
original release25 runtime; that remains a distinct validation step.

## Next phases

The cleaned baseline is intended to support the next scientific layers without
changing the propagation core: first QoIs and Monte Carlo convergence
diagnostics, then a borehole-conditioned geostatistical permeability sampler.
The latter will replace the synthetic/empirical `PermeabilitySampler` with draws
from a conditional spatial uncertainty model while retaining the release25
surrogate, Monte Carlo runner, accumulators and visualization interfaces.
Model uncertainty and alternative propagation methods remain later/optional
extensions.
