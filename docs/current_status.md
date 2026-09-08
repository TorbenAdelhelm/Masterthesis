# Current implementation status

This file is a concise snapshot of what the repository implements *now*. It is
intended to prevent design notes and old baseline plans from being mistaken for
current functionality.

## Implemented pipeline

```text
PermeabilitySampler
  |- EmpiricalPermeabilitySampler
  `- Release25PerlinPermeabilitySampler
            |
            v
TemperatureSurrogate
  |- CallableTemperatureSurrogate
  `- Release25Surrogate
            |
            v
MonteCarloRunner
            |
            +--> FieldStatisticsAccumulator
            +--> ExceedanceProbabilityAccumulator
            `--> arbitrary future TemperatureAccumulator
            |
            v
versioned NPZ + metadata JSON + UQ plots
```

The release25 surrogate executes the real pretrained three-stage LGCNN path
using an external release25 checkout and external model/data assets. The current
input-UQ experiments vary permeability only; pressure and heat-pump locations
are fixed by the selected prepared scenario.

## Implemented uncertainty models

### Empirical ensemble

A finite set of existing permeability fields

$$
\{K^{(1)},\ldots,K^{(N)}\}
$$

is treated as the input ensemble. This is useful for integration tests and a
simple baseline, but it is not a continuous probability model and is not
borehole-conditioned uncertainty.

### Historical synthetic Perlin prior

New permeability realizations are generated with the historical `perlin_v2`
formula used by the released synthetic data. The sampler is lazy and
reproducible for new UQ experiments. It should be described as an unconditional
synthetic prior/input distribution.

## Implemented propagation/statistics

For every sampled permeability field,

$$
T^{(m)}=F(K^{(m)};p_0,i_0)
$$

is evaluated by the deterministic surrogate. Streaming accumulators currently
provide mean, variance, standard deviation, minimum, maximum and threshold
exceedance probabilities. Individual model outputs are only retained when
`--store-all` is requested.

The propagation loop is now QoI-extensible through `TemperatureAccumulator`, so
new monitoring-point or plume-geometry quantities should be implemented as
accumulators rather than as new special-case arguments in the core runner.

## Implemented validation/visualization

The repository contains:

- deterministic velocity/streamline/temperature diagnostics;
- temperature/streamline/heat-pump overlays;
- comparison to prepared physical temperature reference labels;
- MAE, MSE, RMSE, maximum absolute error and bias diagnostics;
- Monte Carlo mean/std/range maps;
- mean-temperature/std-contour overlays;
- mean temperature-change maps;
- empirical spatial exceedance-probability maps.

For the historical synthetic data, temperature change uses the exact
`10.6 °C` initial groundwater temperature from the PFLOTRAN setup.

## Reproducibility state

New Monte Carlo result files use an explicit schema version and a JSON metadata
sidecar. Release25 experiment metadata records code/model provenance where it can
be determined from the local checkout, including Git SHAs, model checkpoint
SHA-256 hashes and software versions.

The historical Perlin generator source/commit is recorded separately from the
new UQ seed. The original generator's NumPy seeding statement was commented out,
so the repository does not claim bitwise reconstruction of the three original
training fields from `seed_id=2907`.

## Validation level

Lightweight CI validates the package without downloading the large DaRUS
assets. The real published models/data have additionally been exercised locally
through the deterministic release25 path and against a prepared RUN_1
reference-temperature field.

A direct numerical comparison against an independently executed original
release25 runtime is a stronger validation level and remains outstanding.

## Not implemented yet

The following are planned thesis layers rather than current functionality:

- monitoring-point QoIs;
- plume length/area QoIs;
- Monte Carlo convergence diagnostics;
- borehole-conditioned GRF/kriging simulation and conditioning diagnostics;
- systematic comparison of geostatistical uncertainty models;
- model/surrogate uncertainty;
- global sensitivity analysis;
- alternative propagation methods such as first-order/JVP approximations.

The next scientific implementation should add QoIs/convergence diagnostics and
then introduce a conditional geostatistical `PermeabilitySampler` without
changing the existing surrogate or propagation interfaces.
