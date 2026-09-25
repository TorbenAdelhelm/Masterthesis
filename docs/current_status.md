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

Coordinate-aware PCE path
  xi_train / xi_validation
        |
        v
StochasticPermeabilityMap
        |
        v
TemperatureSurrogate
        |
        v
TemperatureFunctional
  `- MeanTemperatureAnomaly
        |
        v
PolynomialChaosRegressor
        |
        +--> held-out LGCNN-vs-PCE diagnostics
        `--> analytic PCE mean/variance

Field Monte Carlo path
PermeabilitySampler
        |
        v
TemperatureSurrogate
        |
        v
MonteCarloRunner
        |
        +--> FieldStatisticsAccumulator
        +--> ExceedanceProbabilityAccumulator
        `--> arbitrary future TemperatureAccumulator
```

The release25 surrogate executes the real pretrained three-stage LGCNN path
using an external release25 checkout and external model/data assets. The current
release25 input-UQ experiments vary permeability only; pressure and heat-pump
locations are fixed by the selected prepared scenario.

The finalized RQ1 experiment is implemented as
`python -m subsurface_uq.experiments.release25_rq1 --config <yaml>`. It reuses
the same frozen release25 runtime and propagation interfaces for four variants:
an iid Perlin-coordinate MC baseline, unconditional GRF/KL MC, conditional
GRF/KL MC, and repeated scrambled-Sobol randomized QMC for the conditional
GRF/KL law. RQ1 writes exact empirical temperature quantile fields, nested
convergence diagnostics, receptor/optional mean-anomaly QoIs, input
compatibility diagnostics, and MC-vs-RQMC RMSE/gain tables.

The stochastic-coordinate layer and the generic PCE proof-of-concept machinery
are implemented and independently testable. A release25-specific Perlin-PCE CLI
is also implemented, but the scientifically meaningful run still requires the
external published model/data assets and is therefore not executed by CI.

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
ordinary Monte Carlo use. The map itself remains independent of the
experimental design, so PCE training can pass its own coordinate matrix
directly.

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
`PermeabilitySampler` interface.

The current KL parameters are configuration inputs, not calibrated scientific
defaults. In particular, the Matérn length scales and log-permeability moments
still need to be estimated or selected before scientific LGCNN-UQ experiments
are run.

### Real-field-calibrated permeability generation

The repository now contains a calibration/generation layer for realistic
permeability experiments. Empirical positive permeability fields are transformed
to `log10(K)`; scalable regular-grid y/x/diagonal semivariograms are estimated;
and three candidates are fit and compared by variogram RMSE:

- separable Matérn-3/2;
- separable exponential;
- radial anisotropic exponential.

Synthetic borehole observations can be drawn reproducibly from a held-out real
field with optional margin and minimum-spacing constraints. Separable candidates
reuse `KLLogGaussianPermeabilityMap` plus the existing conditional-KL map. The
radial anisotropic exponential candidate is implemented with GSTools
`Exponential` + simple kriging + `CondSRF`, allowing scalable exact-condition
Monte Carlo fields without a dense full-grid covariance matrix.

The radial GSTools path is intentionally MC-only at present because it does not
expose the explicit finite independent Gaussian coordinates required by the
project's RQMC/Hermite-PCE design. Automatic anisotropy-angle fitting is also
not implemented; calibration currently assumes grid-aligned principal axes,
although the radial sampler accepts a configured rotation angle.

The CLI is:

```text
subsurface-uq-realistic-permeability calibrate ...
subsurface-uq-realistic-permeability generate ...
```

Held-out generation reports log-space mean RMSE/MAE, empirical 90% interval
coverage, and maximum conditioning residual.

A leave-one-out `realistic_permeability evaluate` experiment is also available:
one real field is excluded from calibration, all three covariance candidates are
fit on the remaining fields, and the same synthetic boreholes are used for each
conditional reconstruction. Covariance-family selection now uses exact
full-covariance Gaussian simple kriging for all candidates, evaluated through
the small observation covariance and chunked grid-to-observation
cross-covariances. It therefore does not depend on KL truncation or Monte Carlo
sampling. The experiment writes empirical/fitted variogram plots, fitted
directional length scales and anisotropy ratios, held-out
truth/conditional-mean/std/error figures, and CSV/JSON model-comparison tables.
Large DaRUS grids can use separate calibration and reconstruction strides while
preserving physical distances in metres.

## Implemented Perlin PCE proof-of-concept machinery

`PolynomialChaosRegressor` currently implements scalar non-intrusive PCE for
independent `U(-1,1)` coordinates. The basis is an orthonormal tensor-product
Legendre basis with total-degree truncation.

For degree `p` and dimension `m=2`, the basis size is

$$
P=\binom{m+p}{p}.
$$

Training uses a randomized Latin-hypercube design and ordinary least squares.
Underdetermined and rank-deficient designs are rejected. The fitted model stores
its basis rank, singular values, design condition number and training RMSE.

Because the basis is orthonormal,

$$
\mathbb E[\widehat Q]=c_{\mathbf 0},
$$

and

$$
\operatorname{Var}[\widehat Q]
=\sum_{\boldsymbol\alpha\neq\mathbf0}c_{\boldsymbol\alpha}^2.
$$

`CoordinateQoIEvaluator` preserves paired `(xi,Q)` data without changing
`MonteCarloRunner`. The first implemented continuous target is
`MeanTemperatureAnomaly`, i.e. the spatial mean of `T-T_bg`.

`run_uniform_pce_proof_of_concept` fits the PCE on the Latin-hypercube training
set and validates it on an independent iid `U(-1,1)^2` ensemble. That validation
ensemble is evaluated by the expensive temperature surrogate and therefore acts
as a scalar Monte Carlo reference under exactly the same Perlin coordinate law.
The current diagnostics are RMSE, MAE, maximum absolute error, relative L2 error,
`Q2`, and comparison of Monte Carlo versus analytic-PCE mean and variance.

The module command

```text
python -m subsurface_uq.experiments.release25_perlin_pce
```

constructs the real release25 LGCNN, the Perlin coordinate map and the mean
Delta-T QoI, then runs this train/validation workflow and stores coordinates,
QoIs, coefficients, predictions and metadata in a versioned NPZ archive plus
JSON sidecar. The installed alias is `subsurface-uq-release25-perlin-pce`. The
code path exists, but a real scientific result requires running it with the
external release25 model/data assets.

The detailed formulation is documented in
`docs/perlin_pce_proof_of_concept.md`.

## Implemented propagation/statistics

For every sampled permeability field,

$$
T^{(m)}=F(K^{(m)};p_0,i_0)
$$

is evaluated by the deterministic surrogate. Streaming accumulators currently
provide mean, variance, standard deviation, minimum, maximum and threshold
exceedance probabilities. Individual model outputs are only retained when
`--store-all` is requested.

The propagation loop is QoI-extensible through `TemperatureAccumulator`, while
the PCE path uses per-sample `TemperatureFunctional` objects because regression
must retain one QoI value for each stochastic coordinate vector.

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
- Perlin batch, domain-validation and uniform-sampler reproducibility tests;
- exact-polynomial recovery tests for the orthonormal Legendre regression;
- an end-to-end coordinate-map -> deterministic surrogate -> QoI -> PCE unit test;
- PCE archive/metadata serialization tests.

For the historical synthetic data, temperature change uses the exact
`10.6 °C` initial groundwater temperature from the PFLOTRAN setup.

## Reproducibility state

New Monte Carlo result files use an explicit schema version and a JSON metadata
sidecar. PCE proof-of-concept archives likewise use a versioned schema and store
training/validation coordinates, scalar targets, PCE predictions, multi-indices,
coefficients and JSON metadata.

The historical Perlin generator source/commit is recorded separately from the
new UQ seeds. The original generator's NumPy seeding statement was commented out,
so the repository does not claim bitwise reconstruction of the original random
base offset from `seed_id=2907`.

The Gaussian- and uniform-coordinate samplers and the PCE training/validation
designs use explicit seeds. The stochastic maps expose their coordinate law and
physical-field parameters in metadata.

## Validation level

Lightweight CI validates the package without downloading the large DaRUS
assets. The real published models/data have additionally been exercised locally
through the deterministic release25 path and against a prepared RUN_1
reference-temperature field.

The stochastic-coordinate and PCE mathematics are tested independently of the
release25 runtime. The release25 Perlin-PCE CLI is implemented but has not been
executed in CI because the published model/data assets are external. A direct
numerical comparison against an independently executed original release25
runtime also remains outstanding for the LGCNN adapter.

## Not implemented yet

The following remain planned thesis layers or scientific experiments:

- execute the release25 Perlin-PCE experiment with real model/data assets and
  perform degree/training-budget convergence studies;
- add additional smooth scalar QoIs such as monitoring-point temperatures;
- calibrate/select the final GRF covariance model and hyperparameters on the
  real DaRUS permeability fields, including held-out reconstruction;
- connect the KL/GRF coordinate law to the PCE workflow (Hermite rather than
  Legendre basis);
- quantify Perlin-to-GRF distribution shift before interpreting LGCNN output;
- plume length/area QoIs;
- systematic comparison of geostatistical uncertainty models;
- model/surrogate uncertainty;
- global sensitivity analysis;
- alternative propagation methods such as first-order/JVP approximations.

The immediate scientific step is to run the RQ1 pilot/full experiment with the
selected thesis GRF/KL parameters, conditioning observations, receptors and
optional ROI, then check whether the largest conditional-MC budget is adequate
as an empirical reference. PCE remains a later surrogate layer and is excluded
from the finalized RQ1 evaluation itself.
