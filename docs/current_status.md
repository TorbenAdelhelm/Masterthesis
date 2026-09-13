# Current implementation status

This file is a concise snapshot of what the repository implements *now*. It is
intended to prevent design notes and old baseline plans from being mistaken for
current functionality.

## Implemented pipeline

```text
PermeabilitySampler
  |- EmpiricalPermeabilitySampler
  |- Release25PerlinPermeabilitySampler
  |- GaussianCoordinatePermeabilitySampler
  |            |
  |            `--> StochasticPermeabilityMap: xi -> K
  |                 `- KLLogGaussianPermeabilityMap
  `- UniformCoordinatePermeabilitySampler
               |
               `--> StochasticPermeabilityMap: xi -> K
                    `- PerlinCoordinatePermeabilityMap
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
release25 input-UQ experiments vary permeability only; pressure and heat-pump
locations are fixed by the selected prepared scenario.

The stochastic-coordinate layer is implemented and tested independently. The
new coordinate maps are deliberately **not yet wired into the release25
experiment CLIs**. This keeps the input-law work separable from validation of
the published LGCNN path.

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

`Release25PerlinPermeabilitySampler` generates new permeability realizations
with the historical `perlin_v2` spatial formula used by the released synthetic
data. The sampler is lazy and reproducible for new UQ experiments. It should be
described as an unconditional synthetic prior/input distribution rather than as
an exact reconstruction of the finite training fields.

### Explicit Perlin stochastic-coordinate map

`PerlinCoordinatePermeabilityMap` exposes a low-dimensional stochastic map

$$
G_{\mathrm{Perlin}}:[-1,1]^2\to\mathbb R_+^{H\times W}.
$$

For the historical 2-D `perlin_v2` implementation, only the first two elements
of the random three-component `base_offset` affect `pnoise2`; the third element
is only relevant to the 3-D branch. The new map therefore uses two independent
standardized coordinates

$$
\xi_x,\xi_y\sim\mathcal U(-1,1)
$$

and maps them affinely to the historical random-offset span `[0,4242]` before
calling the same `historical_perlin_v2_field` transformation. An optional
`x_base_shift` represents the historical deterministic integer sample shift
without treating sample index as a stochastic coordinate.

This is an iid continuous law from the same generator family, **not** the exact
joint law of the historical finite training set: the original generator drew
one random base offset and shared it across fields while applying successive
integer x-shifts. The distinction is documented in
`docs/perlin_stochastic_coordinates.md` and should remain explicit in the
thesis.

`UniformCoordinatePermeabilitySampler` supplies iid `U(-1,1)` coordinates for
Monte Carlo use. The map itself remains independent of the experimental design,
so future PCE code can pass deterministic or randomized coordinate matrices
directly. Because the coordinates are independent uniform variables, Legendre
polynomials are the natural PCE family for this baseline.

### KL log-Gaussian stochastic-coordinate map

`KLLogGaussianPermeabilityMap` implements an explicit finite-dimensional map

$$
G_m:\mathbb R^m\to\mathbb R^{H\times W},\qquad \boldsymbol\xi\mapsto K.
$$

The log10-permeability field uses a truncated discrete Karhunen-Loève expansion
with independent standard-normal coordinates. The covariance is a separable
product of one-dimensional Matérn-3/2 correlation matrices. This permits the
2-D KL modes to be formed from two 1-D eigensystems rather than constructing a
full `(H*W) x (H*W)` covariance matrix.

The KL truncation can use either a fixed number of modes or a requested global
variance/energy fraction. `GaussianCoordinatePermeabilitySampler` samples
independent standard-normal coordinates and adapts the map to the existing
`PermeabilitySampler` interface. The map itself does not select a Monte Carlo or
PCE experimental design, which preserves the same `xi -> K` map for future PCE
training.

The current KL parameters are configuration inputs, not calibrated scientific
defaults. In particular, the Matérn length scales and log-permeability moments
still need to be estimated or selected before scientific LGCNN-UQ experiments
are run.

## Implemented propagation/statistics

For every sampled permeability field,

$$
T^{(m)}=F(K^{(m)};p_0,i_0)
$$

is evaluated by the deterministic surrogate. Streaming accumulators currently
provide mean, variance, standard deviation, minimum, maximum and threshold
exceedance probabilities. Individual model outputs are only retained when
`--store-all` is requested.

The propagation loop is QoI-extensible through `TemperatureAccumulator`, so new
monitoring-point or plume-geometry quantities should be implemented as
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
- empirical spatial exceedance-probability maps;
- unit tests that reconstruct the full small-grid covariance from the factorized
  KL basis and compare it to the direct Kronecker covariance;
- KL mode-selection, coordinate-shape, positivity and reproducibility tests;
- Perlin coordinate-to-offset tests against the historical formula;
- Perlin batch, domain-validation and uniform-sampler reproducibility tests.

For the historical synthetic data, temperature change uses the exact
`10.6 °C` initial groundwater temperature from the PFLOTRAN setup.

## Reproducibility state

New Monte Carlo result files use an explicit schema version and a JSON metadata
sidecar. Release25 experiment metadata records code/model provenance where it can
be determined from the local checkout, including Git SHAs, model checkpoint
SHA-256 hashes and software versions.

The historical Perlin generator source/commit is recorded separately from the
new UQ seed. The original generator's NumPy seeding statement was commented out,
so the repository does not claim bitwise reconstruction of the original random
base offset from `seed_id=2907`.

The Gaussian- and uniform-coordinate samplers use explicit NumPy generator seeds.
The stochastic maps expose their coordinate law and physical-field parameters in
metadata. This metadata is not yet written by a release25 experiment because
that integration is intentionally deferred.

## Validation level

Lightweight CI validates the package without downloading the large DaRUS
assets. The real published models/data have additionally been exercised locally
through the deterministic release25 path and against a prepared RUN_1
reference-temperature field.

The stochastic-coordinate infrastructure is tested independently of the
release25 runtime. A direct numerical comparison against an independently
executed original release25 runtime is a stronger validation level for the
LGCNN adapter and remains outstanding.

## Not implemented yet

The following are planned thesis layers rather than current functionality:

- PCE basis construction, experimental design and coefficient fitting;
- release25 propagation of the explicit Perlin coordinate law;
- MC-vs-PCE convergence/error comparison for Perlin QoIs;
- calibration/selection of KL/GRF hyperparameters for a scientific experiment;
- connection of the KL sampler to a release25 experiment CLI;
- monitoring-point QoIs;
- plume length/area QoIs;
- Monte Carlo convergence diagnostics;
- borehole-conditioned GRF/kriging simulation and conditioning diagnostics;
- systematic comparison of geostatistical uncertainty models;
- model/surrogate uncertainty;
- global sensitivity analysis;
- alternative propagation methods such as first-order/JVP approximations.

The next roadmap step is the Perlin MC/PCE proof of concept. It should use the
same `PerlinCoordinatePermeabilityMap` for both reference Monte Carlo and the
PCE training/evaluation design, so discrepancies measure the propagation method
rather than a change in the permeability generator.
