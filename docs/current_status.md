# Current implementation status

This file is a concise snapshot of what the repository implements *now*. It is
intended to prevent design notes and old baseline plans from being mistaken for
current functionality.

## Complete spatial density study at 32–512 observations (2026-10-26)

The [spatial study and atlas index](conditional_density_visual_results.md)
contains eight native permeability inputs and eight LGCNN outputs at each of
32, 64, 128, 256 and 512 nested exact synthetic observations. All 40 conditional
queries and one hidden-reference query completed under the fixed RUN 2
structural law and model context. Matched Gaussian driving draws pair the
counts. Blocked exact conditioning avoids a 25 GiB geometry cache; the
512-observation matrix is numerically stable without a nugget.

The offline gallery includes all fields, reference errors, deviations from
the empirical eight-draw means, common/detail scales, observation overlays
and native-array links. PNG atlases and a static report provide the same
all-sample coverage. From 32 to 512 observations, temperature SD RMS falls
from 0.0544 to 0.0244 C and ensemble-mean temperature RMSE from 0.0501 to
0.0301 C in this single-truth case; the mean error has a small increase at 64.

The 654-test full suite and independent native equations/provenance audit pass.
Offline gallery code checks exercise controls and verify image/array links;
browser rendering was not automated because local-file URLs are blocked in
the in-app browser. Source references and earlier studies remain unchanged.
This is a spatial illustration, not a calibration or geological validation.

## Map-informed structural profiles and sensitivity (2026-10-26)

The primary input laws are now the [RUN 1/2/3 structural profiles](structural_profiles.md),
with native directional-variogram fits, anisotropic Matérn-3/2 correlation,
observed log means and finite-map-corrected total variance. Parameters, source
hashes, fitting criteria, roughness/connectivity departures and all 48 retained
diagnostic draws are documented. These are synthetic processed-map priors;
they have no calibrated geological probability weights.

The [completed sensitivity pilot](structural_profile_results.md) contains 204
actual LGCNN evaluations: three profiles, four paired independent trials,
eight draws per case, and zero versus 32 balanced exact readings. RUN 3's prior
temperature variance is about 35% below RUN 2's. Analytical full-input variance
reductions range from 33% to 77%, while estimated temperature variance reductions
are 22–29%. Local receptor uncertainty does not follow a universal profile
ranking. Four-versus-eight draw differences and outer-trial errors remain
material, so precise tail-risk and conditional-profile rankings are not established.

Independent audits passed 156 source/model hash checks, 648 artifact checks,
conditioning/statistics calculations and source-map variance identities. The
full suite passed 649 tests; the final 24-test focused run also verifies portable
catalog paths and unambiguous Gaussian driving-field metadata. The original
hashed execution sources remain archived. The 7728-evaluation profile matrix
is freshly planned/densely validated with final code; its model propagation
has not run. Earlier illustrative-law matrices below are historical records.

## Measurement-derived model assessment (2026-10-26)

The [measurement-only validation](measurement_model_validation.md) retains 524
real conductivity records at 523 locations. It freezes 12 occupied 8 km test
blocks and a 1 km buffer, leaving 274 development records, 196 test records and
54 buffer exclusions. Buffered development CV compares the existing variogram
families, constant/linear-mean REML alternatives and a non-spatial baseline.
Model choices are persisted before the outer test is scored.

The selected radial-exponential model does not improve regional test scores over
the non-spatial baseline: equal-block CRPS is 2.4% worse, and nominal 90%
observation coverage is 88.3% by points but 69.3% by equally weighted blocks.
It fails the predeclared exploratory observation checks. A separately labelled
post-primary 2 km CV diagnostic suggests modest gains within 500 m of data;
this does not replace the frozen assessment or validate a joint geological law.
Observation and latent uncertainty remain separate, and the residual nugget is
not identified as instrument noise. No native fields or LGCNN queries were made.
Independent covariance/score checks passed; the local full suite passes 640 tests.

## Main controlled experiment and separate realism challenge (2026-10-26)

The main [controlled conditional-field benchmark](controlled_field_benchmark.md)
uses a proper known-mean Gaussian prior, independent hidden truths and exact
synthetic readings. Conditional fields use matched native FFT draws and
Matheron's update. The workflow includes an unconditioned baseline, nested
counts, broad/clustered layouts, independent dense marginal checks, resumable
frozen LGCNN queries, finite-ensemble calibration ranks with randomized ties,
CRPS, risk maps and MC checkpoints. Scenario and layout identities remain separate.
The pilot and larger main matrix have separate configurations; completing the
pilot does not establish convergence or completion of the larger matrix.

The RUN-based validation below remains the empirical realism challenge. Success
under a synthetic known law does not revise its failed adequacy assessment.
Known geological parameters, frozen model weights and simulator discrepancy are
separate uncertainty components. Only conditional permeability uncertainty is
propagated in this first controlled experiment.

The [completed pilot](controlled_field_results.md) contains 82 actual LGCNN
evaluations, independent conditioning/statistics checks and archived execution
sources. The final local suite passes 630 tests; one actual-model regression
reproduces the pilot truth temperature bit for bit. The main 28,736-evaluation
matrix has completed planning/dense validation, with model propagation pending.

## Sparse-reading reconstruction validation (2026-10-25)

The RUN realism [measurement-density study](measurement_density.md) now reserves
spatial test blocks before drawing observations, selects a declared covariance
using spatial CV of only the smallest reading prefix, and freezes it across
larger counts. Withheld error, coverage and standardized residuals are evaluated
before native field generation or model loading. Failed exploratory adequacy
checks produce a report and stop the validation workflow. Repeated layouts,
per-count native recovery and verified prediction recovery are implemented.
The earlier texture-fit screen remains a labelled reference-informed benchmark.
Reference-centered perturbations, measurement-derived site models and sparse
synthetic reconstruction answer different information-state questions; older
production-default descriptions below retain historical context.
The [completed withheld assessment](measurement_density_results.md) passes its
declared exploratory checks in 15 of 54 cases. The study records
`validation_failed`, so no native ensembles or temperature predictions were
generated from its selected laws.

## Conditional temperature studies (2026-10-24)

The [measurement-density workflow](conditioning_temperature.md) now connects the
native conditional FFT law directly to the frozen LGCNN. It uses nested frozen
measurement prefixes, an unconditioned baseline, repeated designs and field-
sample checkpoints. Outputs include per-design temperature uncertainty,
reference/input diagnostics and initial warming exceedance/severity/area and
receptor-risk products. The original conditioning command remains input-only.
Direct MC requires no KL bridge. A matched reduced representation, Sobol mode
attribution, source-connected plume geometry and temperature-based RQ5 EVR
remain separate work. The existing Hermite PCE CLI already accepts serialized
Gaussian/KL artifacts; this does not establish that it represents the new native
radial FFT law. Model-form parameters remain assumptions, and RUNs are separate.
The [completed native pilot](conditioning_temperature_results.md) evaluated 72
generated fields plus three references through the actual frozen LGCNN. It
demonstrates execution and saved-output consistency, not statistical convergence.

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

The earlier RQ1 experiment remains implemented as
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

For the earlier KL-based RQ1 path, KL parameters do not need to be copied into
the RQ1 configuration manually. The realistic-permeability workflow calibrates
the prior from exactly the three complete fields in the DaRUS-5065
real-permeability training dataset that were used for the DaRUS-5082 real-K
LGCNN, and writes a reusable `stochastic_input_model.yaml` artifact.
The empirical between-field mean variation is recorded, but the corresponding
field-wide Gaussian mode is disabled by default because only three complete
training fields support that estimate. It is available only as an explicit
sensitivity option. Manual KL parameters remain available as a synthetic/debug
fallback.

### Exact Munich-to-LGCNN geospatial mapping

The realistic-permeability branch now contains a georeferencing step for the
release25/DaRUS real-permeability domains. The public raw dataset defines the
standard 12.8 km x 12.8 km, 2560 x 2560, 5 m geometry but does not expose the
projected Munich crop origin in the PFLOTRAN HDF5 arrays. The
`georeference-domain` experiment therefore matches a run's raw permeability
fingerprint to the Munich 100 m reference field, searches raw-array axis/flip
conventions, refines the crop at 5 m resolution, and stores an explicit projected
cell-centre/domain-edge manifest with match diagnostics. Weak matches are rejected
by default instead of being silently accepted.

### Multi-run georeference sweep

The branch now provides `georeference-sweep`, which evaluates all nine
`K_P10/K_P50/K_P90 x top/bottom/log_geomean` reference representations across
multiple DaRUS `RUN_n` fields. It writes a single comparison CSV and a YAML
decision summary. An exact Munich mapping is flagged as defensible only when one
representation passes the single-run validation thresholds for every requested
run, uses one common raw-array transform, has a common unit-shift interpretation
and a small cross-run log-unit-shift spread. Otherwise the workflow explicitly
records that no exact geographic mapping is supported by the available
reference table.

### Training-informed, measurement-conditioned new LGCNN domain

The earlier measurement-derived workflow distinguishes geostatistical replication from the
CNN training sample population. The real-permeability training fields come from
DaRUS-5065; the pretrained model artifacts are published separately as
DaRUS-5082. Three of the four standard 12.8 km fields are used for training and
one for validation. During
training, release25 `SimulationDatasetCuts` presents overlapping patches to the
CNN; the published best settings for both Step 1 and Step 3 are a 1280-cell box
and skip 8.

That workflow uses **measurement-derived kriging represented through
a finite KL basis**. It is a separate earlier experiment, not the current main
controlled benchmark or evidence of an empirically calibrated site posterior.
The real Munich observations determine the mean,
structured variance, nugget and directional correlation lengths through
irregular-point variograms plus spatial block cross-validation. After converting
hydraulic conductivity to intrinsic permeability, those parameters define the
site-specific `log10(k)` prior and the same observations condition it at their
continuous coordinates. The posterior remains parameterized by iid Gaussian
coordinates for MC/RQMC/Hermite PCE.

The three DaRUS-5065 training fields no longer define the production geological
prior. Their overlapping patches are not counted as independent geological
realizations. Instead, they define the **primary surrogate-support/fidelity
reference**: patch descriptors, full-field statistics, marginal distances and
directional variogram mismatch are reported for every generated ensemble.

The production command requires the exact three `RUN_*` training names when a
raw release25 dataset directory is used. It verifies that the field-based prior
calibration was fitted to the same run subset. The public paper establishes the
3/1 split but not the run names in its text, so directory order is deliberately
not used as an implicit split.

Generated fields and patches are never filtered by the support diagnostics.
Thus the conditional map retains iid Gaussian posterior coordinates for
MC/RQMC/Hermite PCE. The stochastic-input artifact now records the primary patch
reference, secondary full-field reference, optional release25 permeability
normalization metadata, and whether the between-field mean sensitivity mode was
enabled.

With `--save-samples`, every generated realization keeps its float32 NPY
source plus a robust shared field PNG whose scale spans the training and
measurement q01--q99 envelopes, a dedicated measurement-overlay PNG and
a four-panel comparison PNG. The overlay draws the original measured intrinsic
permeability values on top of the generated raster with the **same color
normalization**, so local agreement is visually inspectable. The comparison also
shows the measurement-derived kriging reference, closest DaRUS-5065 training
field and generated-minus-reference residual.

### Real Munich measurement calibration

The realistic-permeability branch now contains a second, preferred calibration
path based on the actual Munich hydraulic-conductivity observations. The Excel
loader retains positive Quaternary/unconfined measurements, intersects them with
the active XY footprint recovered from the 3-D reference field, and preserves
the original continuous coordinates while recording nearest 100 m grid cells
for diagnostics.

Covariance calibration can be performed directly on irregular
`log10(K_h)` point pairs using directional x/y/diagonal semivariograms. The
measurement workflow now jointly fits structured variance, nugget variance and
directional length scales with pair-count-weighted variogram residuals. Model
comparison uses spatial block cross-validation with fold-wise recalibration and
exact full-covariance simple-kriging predictions at held-out real measurement
locations. Reported diagnostics include RMSE/MAE, 90% posterior coverage,
standardized residuals, Gaussian NLPD, fold-wise length scales, nugget fractions
and weighted/unweighted variogram RMSE. A default robustness check repeats the
fit after excluding only values above 5e-2 m/s while retaining all measurements
in the baseline calibration.

After model selection, `measurement-condition` evaluates the analytical
posterior on the active 100 m reference grid from all accepted continuous
measurements and records both hydraulic-conductivity quantities and an explicit
conversion to intrinsic permeability using documented fluid-property constants.

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
- validate a covariance-matched reduced representation of the selected native
  radial law before applying the existing Hermite PCE workflow;
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
