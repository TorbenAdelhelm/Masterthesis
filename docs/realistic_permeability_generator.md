# Realistic permeability generator

## Purpose

The realistic-permeability workflow turns empirical permeability fields into a
calibrated stochastic input model that can be conditioned on sparse synthetic
boreholes. It is designed for the DaRUS real-permeability data but accepts any
positive `[N,H,W]` ensemble supported by `load_empirical_fields`.

The modeled Gaussian field is

```text
Y(x) = log10 K(x)
```

and permeability is recovered with `K=10**Y`.

## Candidate covariance models

The calibration compares three stationary, grid-aligned candidates:

- `matern32`: separable Matérn-3/2, retained for the existing factorized KL path;
- `exponential`: separable exponential / Matérn-1/2, giving rougher fields while
  preserving the factorized KL structure;
- `radial_exponential`: radial anisotropic exponential covariance

  `rho(dx,dy)=exp(-sqrt((dx/lx)^2+(dy/ly)^2))`.

The radial model is generated with GSTools and may additionally be rotated by a
configured angle. Automatic angle fitting is intentionally not implemented yet;
the calibration currently assumes the principal anisotropy axes are aligned with
the permeability grid. A non-zero angle can still be supplied programmatically
to `RadialExponentialPermeabilitySampler` when justified externally.

The separable exponential model is different from the radial anisotropic model:

```text
rho_sep(dx,dy) = exp(-|dx|/lx - |dy|/ly)
rho_rad(dx,dy) = exp(-sqrt((dx/lx)^2 + (dy/ly)^2))
```

Both have identical one-dimensional axis correlations. The additional main-
diagonal empirical variogram is therefore included in calibration to distinguish
their two-dimensional structure.

## DaRUS-5065 raw-data support

The real-permeability DaRUS dataset is distributed as PFLOTRAN HDF5 data. The
loader mirrors the release25 preprocessing convention:

- read grid dimensions from `settings.yaml -> grid -> size [m]`;
- find `RUN_*/pflotran.h5` in numerical run order;
- read initial-time group `   0 Time  0.00000E+00 y`;
- extract `Permeability X [m^2]`;
- reshape with the release25 grid dimensions and remove the singleton vertical
  dimension.

Therefore the calibration CLI may point `--fields` directly at an unpacked
DaRUS-5065 raw dataset root; conversion to an intermediate NumPy file is not
required.

## Calibration from real fields

`calibrate_covariance_candidates` estimates the global `log10(K)` mean and
standard deviation and computes y-, x- and main-diagonal empirical
semivariograms on the regular grid. Candidate length scales are fitted jointly
by nonlinear least squares. Results are sorted by the combined variogram RMSE.

This ranking is a model-screening diagnostic, not a proof that the first model
is the unique geological truth. Held-out reconstruction, visual morphology and
application-level compatibility with the real-permeability LGCNN remain part of
the scientific validation.

For large 2560x2560 fields, `spatial_stride` can reduce calibration cost while
preserving physical lag distances. The original fields are still used for the
marginal mean and variance.

Example:

```bash
subsurface-uq-realistic-permeability calibrate \
  --fields data/dataset_100hp_giant_real_fixP0_0025 \
  --cell-size-m 5 \
  --max-lag-cells 96 \
  --spatial-stride 4 \
  --output run_output/realistic_k/calibration.yaml
```

## Exact Munich-to-LGCNN domain georeferencing

The release25 real-permeability runs contain local PFLOTRAN arrays, but the raw
HDF5 files do not encode the projected Munich origin of each 12.8 km cutout.
Before measurement-conditioned stochastic fields are evaluated on a 5 m LGCNN
grid, the local array must therefore be tied back to the projected Munich
reference model.

The `georeference-domain` command infers this mapping from the spatial
permeability fingerprint rather than hard-coding an undocumented crop origin.
It:

- loads one DaRUS/release25 `RUN_*/pflotran.h5` permeability field;
- reads a horizontal `K_P10`, `K_P50` or `K_P90` surface from the
  headerless Munich reference table;
- tests all axis-swap / x-flip / y-flip combinations because the release25 raw
  loader reshapes PFLOTRAN arrays in settings-file dimension order whereas the
  geostatistical code uses geographic `[y,x]` indexing;
- coarsens the 5 m raw field to a 100 m log-space fingerprint and searches the
  reference grid for candidate cutouts;
- refines the winning crop at 5 m resolution by bilinear interpolation of the
  reference surface;
- fits one additive log10 offset during matching, so an unknown constant
  hydraulic-conductivity-to-intrinsic-permeability conversion does not alter the
  spatial score;
- records the first 5 m cell centre, domain edges, array transform, match
  correlation/RMSE, unit-shift diagnostic and optional overlap with the real
  measurement locations.

For a standard DARUS-5065 real run, the target geometry is 12.8 km x 12.8 km,
2560 x 2560 cells at 5 m resolution. The public dataset metadata confirms this
geometry, but not the projected crop origin; the manifest produced here supplies
that missing information.

Example for the fixed RQ1 run:

```bash
python -m subsurface_uq.experiments.realistic_permeability georeference-domain \
  --raw-dataset data/dataset_100hp_giant_real_fixP0_0025 \
  --run RUN_1 \
  --reference-grid "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf-Werte-3D Modell/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --reference-column K_P50 \
  --reference-z-mode top \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --raw-cell-size-m 5 \
  --require-validated \
  --plots-dir run_output/realistic_k/georeference_plots \
  --output run_output/realistic_k/georeference_RUN_1.yaml
```

The command fails when `--require-validated` is active and the best match does
not meet the default minimum correlation, maximum centred log-RMSE and reference
coverage. In that case the reference surface must be inspected rather than
silently accepting an uncertain origin. In particular, if the 3-D reference
model varies materially with depth, rerun with `--reference-z-mode bottom`,
`log_geomean`, or `nearest --reference-z-m <z>` and compare the validation
scores.

The resulting YAML manifest is the authoritative mapping for the subsequent
stochastic generator. It stores geographic first-cell centres and cell-edge
bounds explicitly, avoiding ambiguity between PFLOTRAN array axes, image rows,
cell centres and domain edges. When `--plots-dir` is supplied, a validation
figure shows the full Munich reference surface with the inferred crop rectangle
beside the oriented, unit-shift-corrected raw field using common color limits.

### Automatic multi-run georeference sweep

A failed single `K_P50/top` match is not resolved by weakening the validation
thresholds. Use `georeference-sweep` to test the complete predefined diagnostic
set

```text
{K_P10, K_P50, K_P90} x {top, bottom, log_geomean}
```

across at least two independent `RUN_n` fields. The command always evaluates
all nine representations and writes one row per
`run x reference-column x vertical-representation` combination.

Example:

```bash
python -m subsurface_uq.experiments.realistic_permeability georeference-sweep \
  --raw-dataset data/dataset_100hp_giant_real_fixP0_0025 \
  --runs RUN_1 RUN_2 RUN_3 \
  --reference-grid "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf-Werte-3D Modell/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --raw-cell-size-m 5 \
  --output-dir run_output/realistic_k/georeference_sweep
```

Optional `--plots-dir <dir>` writes the ordinary alignment figure for every
successful run/representation pair. This can produce many large figures, so it
is intentionally not required for the first sweep.

The main table is

```text
georeference_sweep.csv
```

and contains the individual match diagnostics plus cross-run representation
diagnostics. In particular, each row records whether its reference
representation:

- is present and individually validated for every requested run;
- selects one common raw-array transform across all runs;
- has a consistent unit-shift interpretation;
- keeps the max-min fitted log10 unit-shift spread below the configured
  tolerance (default 0.15);
- therefore satisfies the explicit `representation_defensible` flag.

The CSV also contains the global
`consistent_defensible_mapping_exists` flag, the number of defensible
representations, the `unique_defensible_representation` flag, and the selected
representation, if one exists. This avoids treating several equally defensible
reference representations as a uniquely identified parent field. The decision
is additionally written in `georeference_sweep_summary.yaml`.

The crop origin itself is not required to be identical across runs because
different DaRUS runs may be different cutouts. Consistency concerns the parent
reference representation, array orientation, validation quality and unit
relationship.

A representation is called defensible only if **all** requested runs pass the
single-run thresholds

```text
correlation >= 0.90
centered log10 RMSE <= 0.15
reference coverage >= 0.80
```

and the cross-run transform/unit-shift criteria above. If no representation
passes, the workflow reports

```text
consistent_defensible_mapping_exists: false
```

and no exact Munich crop should be assigned to the DaRUS LGCNN runs from this
reference table. This negative result is scientifically usable: it separates
measurement-based covariance calibration, which remains valid, from an
unsupported claim of exact geographic correspondence.

Use `--require-defensible` only when a downstream script should fail after
writing the diagnostic outputs if no defensible mapping exists.

## Training-informed, measurement-conditioned new LGCNN domain

The realistic-`k` LGCNN was not trained on four complete permeability images as
four independent optimization samples. The publication and release25 code show a
two-level structure:

```text
complete permeability fields -> overlapping SimulationDatasetCuts patches -> CNN
```

For the real-permeability experiment, three of the four standard 12.8 km fields
are used for training and one for validation; a separate larger field is used
for scaling. The selected hyperparameters for both LGCNN Step 1 and Step 3 are

```text
box_length = 1280 cells
skip_per_dir = 8 cells
batch_size = 8
```

with 5 m cells. The real-permeability training data are published as
[DaRUS-5065](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5065).
The separate DaRUS-5082 dataset contains the pretrained real-`k` model
artifacts.

This matters for uncertainty modeling. The three complete training fields are
still only three geostatistical realizations. Their tens of thousands of
overlapping cutouts are strongly dependent and must **not** be interpreted as
independent samples from the geological random field. Conversely, patch-scale
statistics are the more relevant reference for whether a new permeability input
falls inside the spatial support seen by the convolutional surrogate.

The production architecture is therefore

```text
real Munich measurements
    -> directional irregular-point variograms
    -> spatial block CV across covariance families
    -> selected mean / structured variance / nugget / ell_x / ell_y
    -> K_h -> intrinsic k (constant log shift)
    -> 5 m KL prior in log10(k)
    -> continuous-point conditioning on the same measurements

eta ~ N(0,I)
    -> conditional measurement-derived KL field
    -> generated intrinsic-permeability field

three complete DaRUS-5065 training fields
    -> exact release25 cutout lattice
    -> LGCNN support/fidelity diagnostics only
```

Thus the site-specific geological law is measurement-derived, while the
DaRUS-5065 fields answer the separate question of whether the generated inputs
remain compatible with what the frozen LGCNN saw during training. No generated
field or patch is rejected, so the posterior coordinates remain iid standard
Gaussian for MC, randomized QMC and Hermite PCE.

### Step 1: identify the three LGCNN training runs

The public paper establishes the 3/1 training-validation split but does not
identify the exact `RUN_*` names in the text. The implementation therefore
requires the three training run names to be supplied explicitly rather than
assuming that the first three directory entries are the training split. Obtain
those names from the DaRUS-5082 model package / command-line training metadata.

Calibrate these fields for **surrogate-support/fidelity comparison and
sensitivity alternatives**:

```bash
python -m subsurface_uq.experiments.realistic_permeability calibrate \
  --fields data/dataset_100hp_giant_real_fixP0_0025 \
  --runs <TRAIN_RUN_A> <TRAIN_RUN_B> <TRAIN_RUN_C> \
  --cell-size-m 5 \
  --spatial-stride 4 \
  --output run_output/realistic_k/training_calibration.yaml
```

The loader accepts the unpacked release25 dataset root and reads each
`RUN_*/pflotran.h5`. A single `RUN_n/pflotran.h5` path is also accepted for
inspection/debugging, but it is not sufficient to calibrate the production
training prior.

The training calibration stores ordinary physical-`log10(K)` covariance fits,
the empirical normal-score sensitivity model and the patch/full-field support
statistics. None of those training-derived covariance fits defines the
production geological prior anymore. The field-wide training-mean coordinate is
also unavailable in the measurement-kriging baseline; it remains an explicit
sensitivity assumption only for training-derived laws.

### Step 2: reproduce the network's patch support

The release25 implementation `SimulationDatasetCuts` defines

```text
n_patch_per_field = (H-B)(W-B) / skip^2
```

for the published geometry, with integer indexing equivalent to the original
`idx_to_pos` implementation. For `H=W=2560`, `B=1280` and `skip=8`, this is

```text
25,600 overlapping patches per complete training field
76,800 patches across the three training fields
```

These 76,800 patches are highly correlated. The diagnostics therefore reproduce
the exact patch lattice but use a deterministic, configurable subsample for
feature calculation. The default is 192 training patches total and 32 inspected
patches per generated field. This controls computational cost without changing
the stochastic prior.

Patch descriptors are computed in physical `log10(K [m^2])` and include

- mean, standard deviation and q05/q50/q95;
- x/y RMS gradients per metre;
- x/y lag-one correlation.

The same descriptor set is retained at full-field scale as a **secondary**
diagnostic. The output explicitly labels

```text
training_patch_compatibility      -> primary surrogate-support diagnostic
training_full_field_compatibility -> secondary field-scale diagnostic
```

Both have `decision_rule: null`.

If the `info.yaml` from the DaRUS-5082 model/prepared dataset is supplied via
`--training-info-yaml`, the exact permeability input normalization metadata is
also stored. release25 normalizes the prepared full inputs before patch
extraction, so this records the network-input context without pretending that
the patch population is statistically independent.

### Step 3: calibrate the production measurement-kriging model

Run `measurement-evaluate` on the Munich hydraulic-conductivity observations.
This is now the **production geological calibration**. The workflow estimates
directional point variograms, structured variance, nugget and x/y correlation
lengths, and compares covariance families with spatial block cross-validation.

When `new-domain-generate` is called without `--model`, it chooses the
lowest-CV-RMSE candidate among the KL-supported Matérn-3/2 and exponential
families. Hydraulic-conductivity-to-intrinsic-permeability conversion adds only
a constant in log space, so the fitted variogram shape, variance and length
scales remain valid for `log10(k)`; only the mean is shifted.

### Step 4: generate the conditional law

Example:

```bash
python -m subsurface_uq.experiments.realistic_permeability new-domain-generate \
  --measurements "C:/path/to/kf_werte_190201.xlsx" \
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --training-fields data/dataset_100hp_giant_real_fixP0_0025 \
  --training-runs <TRAIN_RUN_A> <TRAIN_RUN_B> <TRAIN_RUN_C> \
  --training-calibration run_output/realistic_k/training_calibration.yaml \
  --measurement-calibration run_output/realistic_k/measurements/measurement_calibration.yaml \
  --training-info-yaml <PATH_TO_DARUS_5082_INFO_YAML> \
  --input-law measurement-kriging \
  --domain-size-m 12800 \
  --cell-size-m 5 \
  --energy-threshold 0.95 \
  --n-samples 8 \
  --batch-size 1 \
  --seed 4901 \
  --output-dir run_output/realistic_k/new_domain
```

The command verifies that the training calibration and support diagnostics use
the same `RUN_*` subset. It also requires exactly three full training fields for
the DaRUS-5082 production profile.

For each real measurement inside the selected domain, hydraulic conductivity is
converted to intrinsic permeability with the documented constant fluid-property
factor. The selected measurement-derived covariance defines the 5 m
`log10(k)` KL prior. The fitted nugget plus optional additional measurement
error defines observation uncertainty, and the KL basis is evaluated at the
original continuous measurement coordinates.

The deterministic zero-coordinate field is therefore the truncated-KL
representation of the measurement-derived simple-kriging posterior mean. The
generator reports its RMSE/MAE and predictive coverage at the measurement
locations, together with the KL covariance-approximation error.

### Outputs and reuse in RQ1/PCE

The output directory contains

```text
conditioning_measurements.csv
empirical_mean_log10_permeability_m2.npy
empirical_std_log10_permeability_m2.npy
new_domain_generator.yaml
new_domain_generator.json
stochastic_input_model.yaml
new_domain_mean_std.png
conditioned_reference_log10_permeability_m2.npy
conditioned_reference_permeability_m2.npy
conditioned_reference_permeability_m2.png
conditioned_reference_measurement_overlay.png
samples/
  sample_0001_permeability_m2.npy
  sample_0001_permeability_m2.png
  comparisons/
    sample_0001_measurement_overlay.png
    sample_0001_comparison.png
  ...
```

`new_domain_generator.yaml/json` stores the primary patch-level support
comparison, secondary full-field comparison, prior-predictive measurement
compatibility and whether the optional between-field mean mode was enabled.

`stochastic_input_model.yaml` remains the authoritative reusable probability
law. It stores the prior/conditioning map plus the patch reference, full-field
reference and optional release25 normalization context. RQ1 consumes this file
through

```yaml
grf:
  input_model: run_output/realistic_k/new_domain/stochastic_input_model.yaml
```

without retyping stochastic parameters.

At 2560 x 2560 resolution a float32 field is about 26 MB. With the default
`--save-samples`, the NPY remains the numerical source of truth. The simple PNG
uses one robust shared `log10(k)` scale spanning the training and measurement
q01--q99 envelopes. More importantly, every
`*_measurement_overlay.png` draws the original measured intrinsic-permeability
values on the generated raster using **the identical color normalization**.
This is the direct visual check requested for local measurement consistency.

The companion comparison PNG displays the generated field, the
measurement-derived kriging reference, the closest full DaRUS-5065 training
field according to standardized marginal/gradient/correlation descriptors, and
the generated-minus-reference residual. The generator additionally stores
measurement conditioning RMSE/MAE/coverage and `training_data_fidelity`
(log-space Wasserstein distance, quantile errors and x/y/diagonal semivariogram
curve mismatch). These diagnostics never reject samples.

The new projected domain defines only the permeability field. Reusing the
release25 pressure/material-ID/heat-pump inputs still means those quantities are
domain-relative templates rather than measured co-located Munich site inputs.

## Real Munich measurement workflow

The preferred calibration path now uses the actual hydraulic-conductivity
measurements rather than treating high values in a derived raster as measurement
locations.

Expected source files:

- Excel workbook `kf_werte_190201.xlsx`, sheet `kf_werte_180223`;
- headerless, whitespace-delimited 3-D Munich reference grid
  `3D_K_Field_Munich_K_P10_P50_P90.csv`.

The Excel loader retains finite positive `KF Wert` observations with
`Strategrap == "q"` and `GW_Zustand == "ungespannt"`, normalizes text
fields, generates stable identifiers for missing object IDs, and intersects the
continuous point coordinates with the active XY footprint of the 3-D reference
grid. The original coordinates are retained for kriging. Nearest 100 m grid
indices are recorded only for diagnostics / downstream methods that explicitly
require one grid cell per observation.

The coordinate reference system is recorded as the working assumption
`DHDN / Gauss-Krüger zone 4 (EPSG:31468)`; the workbook itself does not encode
a CRS, so this remains an assumption until externally confirmed.

### Irregular-point variogram calibration

The command `measurement-evaluate` estimates directional empirical
semivariograms directly from the real measurement pairs in
`log10(K_h)`, where hydraulic conductivity `K_h` is measured in m/s.
The x, y, and combined diagonal directions use configurable angular and lag-bin
tolerances. Matérn-3/2, separable exponential, and radial anisotropic
exponential candidates are then fit to those point variograms.

The measurement fit now separates structured spatial variance and a nugget:
```text
gamma(h) = tau^2 + sigma_s^2 * (1 - rho(h)),  h > 0
gamma(0) = 0
```
where `tau^2` is the nugget variance and `sigma_s^2` is the spatially
correlated variance. The optimizer jointly fits the total sill, nugget fraction
and directional length scales. Empirical variogram bins are weighted by their
number of contributing point pairs by default, so a sparsely supported bin does
not have the same influence as a bin supported by hundreds of pairs.

The output reports the nugget fraction
```text
tau^2 / (tau^2 + sigma_s^2)
```
for each covariance family. Sensitivity flags allow the nugget or pair-count
weighting to be disabled, but the default thesis workflow uses both.

The fitted nugget must be interpreted as an effective unresolved short-scale
component. The available workbook does not provide repeated co-located
measurement uncertainty or quality metadata, so `tau^2` cannot be uniquely
decomposed into measurement error, sub-bin spatial variability and true
microscale heterogeneity.

Because the raw observations are used directly, the covariance fit no longer
inherits the smoothing properties of an already interpolated P50 raster.

### Spatial block cross-validation

Kernel comparison uses spatial block cross-validation rather than a random
point split. All measurements within the same rectangular block are held out
together, and each fold recalibrates the variogram parameters on the remaining
measurements before predicting the held-out real observations by exact
full-covariance simple kriging.

Reported diagnostics include:

- RMSE and MAE in `log10(K_h)`;
- nominal 90% Gaussian posterior coverage;
- standardized residual mean and standard deviation;
- Gaussian negative log predictive density (NLPD);
- fold-wise fitted length scales, nugget fractions and variogram RMSE;
- pair-count-weighted variogram RMSE.

The workflow records the lowest spatial-CV RMSE candidate as a diagnostic
selection, but the output explicitly notes that this is not proof of a unique
geological covariance law.

Example:

```bash
python -m subsurface_uq.experiments.realistic_permeability measurement-evaluate \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --reference-grid "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf-Werte-3D Modell/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --models matern32 exponential radial_exponential \
  --lag-bin-m 250 \
  --max-lag-m 3000 \
  --angle-tolerance-deg 22.5 \
  --min-pairs-per-bin 8 \
  --cv-folds 5 \
  --cv-block-size-m 2000 \
  --cv-seed 2907 \
  --observation-std-log10-k 0 \
  --robustness-upper-k-m-s 5e-2 \
  --output-dir run_output/realistic_k/measurements
```

The output includes filtered continuous measurements, a one-value-per-cell
geometric-mean diagnostic table, empirical point variograms, fitted covariance
parameters, nugget fractions, spatial-CV summaries, and per-fold diagnostics.
The figures additionally include `measurement_nugget_fraction.png`.

The baseline calibration always retains all accepted measurements. By default,
the command also repeats calibration and spatial CV after excluding only values
above `5e-2 m/s`, corresponding to the nominal upper range used in the
accompanying parameter table. This is a robustness diagnostic only: the high
measurements are not clipped or removed from the baseline model. The retained
measurements keep the same spatial fold assignments as in the baseline CV, so
changes in the robustness table are not caused by a new fold partition. Results
are written to `measurement_upper_tail_robustness.csv`. Set
`--robustness-upper-k-m-s 0` to disable this check.

### Conditioning on all real measurements

After inspecting the spatial-CV results, the selected covariance family can be
conditioned on all accepted real measurements:

```bash
python -m subsurface_uq.experiments.realistic_permeability measurement-condition \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --reference-grid "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf-Werte-3D Modell/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --calibration run_output/realistic_k/measurements/measurement_calibration.yaml \
  --observation-std-log10-k 0 \
  --output run_output/realistic_k/measurements/conditioned_real_measurements.npz
```

If `--model` is omitted, the covariance family with the lowest spatial-CV
RMSE from the calibration file is used. The analytical posterior is evaluated
on the active 100 m XY reference grid from the original continuous measurement
coordinates.

With a fitted nugget, the gridded output distinguishes the posterior of the
spatially correlated field from measurement/microscale predictive uncertainty.
The structured posterior variance excludes an independent nugget realization;
a separate predictive variance adds the fitted nugget back. The nugget is not
silently injected as independent cell-wise noise into an LGCNN input field,
because that would require an explicit scale/modeling decision.

The measurement quantity is hydraulic conductivity `K_h [m/s]`, not intrinsic
permeability `k [m^2]`. The conditioning output therefore stores the posterior
in `log10(K_h)` and additionally converts physical posterior median/mean maps
to intrinsic permeability using

```text
k = K_h * mu / (rho * g).
```

Default conversion constants are documented in the JSON metadata and correspond
approximately to liquid water near 20 degC. For constant fluid properties this
conversion is a constant shift in log space and therefore does not change the
variogram shape or fitted correlation length scales.

## Leave-one-out DaRUS calibration inspection

The `evaluate` subcommand is the recommended model-selection experiment before
RQ2. It keeps one real permeability field hidden, calibrates all covariance
candidates on the remaining real fields, uses the same synthetic boreholes for
all models, and compares the resulting conditional reconstructions.

For covariance-family selection, all three candidates are now evaluated with
the same exact full-covariance Gaussian simple-kriging equations. No KL
truncation and no conditional Monte Carlo sampling are used in this comparison.
Only the small observation covariance `K_DD` and chunked grid-to-observation
cross-covariances `K_xD` are formed; the dense full-grid covariance is never
materialized. This removes the previous methodological confound in which the
radial model was full-rank while the separable models were represented by a
finite KL truncation.

For the large 2560x2560 DaRUS fields, two independent strides are available:

- `--spatial-stride` accelerates empirical-variogram estimation while retaining
  lag distances in metres;
- `--evaluation-stride` down-samples only the held-out reconstruction grid,
  again increasing the effective cell size so physical distances remain
  consistent.

A practical first run is:

```bash
subsurface-uq-realistic-permeability evaluate \
  --fields data/dataset_100hp_giant_real_fixP0_0025 \
  --cell-size-m 5 \
  --truth-index 0 \
  --models matern32 exponential radial_exponential \
  --max-lag-cells 96 \
  --spatial-stride 4 \
  --evaluation-stride 4 \
  --n-boreholes 30 \
  --borehole-seed 2907 \
  --margin-cells 10 \
  --min-spacing-cells 20 \
  --observation-std-log10-k 0 \
  --kriging-chunk-rows 64 \
  --output-dir run_output/realistic_k/loo_truth0
```

The output directory contains:

```text
calibration_leave_one_out.yaml
empirical_variograms.npz
model_comparison.csv
model_comparison.json
figures/
  variogram_fits.png
  length_scales.png
  heldout_matern32.png
  heldout_exponential.png
  heldout_radial_exponential.png
  heldout_metric_comparison.png
arrays/
  <model>/
    posterior_mean_log10_k.npy
    posterior_std_log10_k.npy
    posterior_variance_log10_k.npy
```

`model_comparison.csv` reports fitted `ell_x`, `ell_y`,
`ell_x/ell_y`, total/directional variogram RMSE, held-out conditional-mean
RMSE/MAE, Gaussian 90% posterior coverage, and conditioning residuals.

The 90% coverage in this inspection experiment uses the analytical Gaussian
posterior interval
`mean +/- 1.64485 * std` in `log10(K)`. The mean and variance are
deterministic for a fixed covariance model and borehole layout, so the model
comparison is independent of random seeds, sample count and KL truncation. The
separate `generate` command still produces stochastic conditional ensembles
after a covariance model has been selected.

The hidden truth is excluded from covariance calibration, so this is a genuine
leave-one-out reconstruction diagnostic rather than an in-sample fit. No
automatic geological winner is declared: variogram fit and held-out
reconstruction are reported side by side.

## Synthetic borehole experiment

`sample_borehole_observations` samples fixed measurements from one hidden
reference field. Optional margins and minimum grid-cell spacing support
controlled borehole layouts.

The generator CLI then treats the chosen empirical field as hidden truth, keeps
only the synthetic borehole values for conditioning, and produces a conditional
ensemble:

```bash
subsurface-uq-realistic-permeability generate \
  --fields data/dataset_100hp_giant_real_fixP0_0025 \
  --calibration run_output/realistic_k/calibration.yaml \
  --truth-index 0 \
  --n-boreholes 30 \
  --min-spacing-cells 20 \
  --n-samples 64 \
  --output run_output/realistic_k/conditional_fields.npz
```

The output archive stores the generated permeability fields and borehole
observations. A JSON sidecar stores the calibration, sampler metadata and
held-out validation metrics:

- RMSE and MAE of the conditional ensemble mean in `log10(K)`;
- empirical 90% pointwise interval coverage of the hidden truth;
- maximum conditioning residual at borehole cells.

## Separable KL generation

For `matern32` and `exponential`, the existing `KLLogGaussianPermeabilityMap`
is reused. The covariance remains

```text
sigma_Y^2 * (C_y kron C_x)
```

which avoids constructing the full 2-D dense covariance. The existing
`ConditionalKLLogGaussianPermeabilityMap` then conditions the retained KL model
on boreholes and exposes independent posterior Gaussian coordinates. This path
remains suitable for MC, randomized QMC and future Hermite PCE.

Exact conditioning is exact only relative to the retained KL subspace. If sparse
KL truncation cannot represent the exact borehole values, increase `n_modes` or
use a justified observation uncertainty.

## Radial anisotropic exponential generation

`RadialExponentialPermeabilitySampler` uses `gstools.Exponential` and
`gstools.CondSRF` with known-mean simple kriging. This provides scalable
conditioned MC fields with the radial anisotropic exponential covariance and
avoids constructing a dense full-grid covariance matrix.

The baseline borehole measurements are treated as exact. The radial sampler is
currently an MC sampler only: GSTools' random-field generator does not expose
the explicit finite independent Gaussian coordinate vector required by the
project's RQMC/Hermite-PCE path. Therefore radial exponential fields are not
silently substituted into the KL-based RQMC workflow.

Install the optional geostatistical dependency with:

```bash
python -m pip install -e ".[geostat]"
```

This extra does not install the legacy `noise` C extension. The latter is only
needed for the historical Perlin experiments and is isolated in the separate
`perlin` extra, which avoids unnecessary Windows compiler/SDK failures during
DaRUS calibration.

## Integration with RQ1 and later RQs

`RQ1Config.grf.covariance_model` now accepts `matern32` and `exponential` for
the factorized KL path. The radial exponential generator remains a separate
realism/MC path until an explicit finite-coordinate representation is validated.

The intended scientific sequence is:

```text
real DaRUS K fields
  -> log10(K) spatial-statistics calibration
  -> compare Matérn-3/2 / separable exponential / radial exponential
  -> synthetic boreholes from a held-out real field
  -> conditional realizations
  -> held-out realism validation
  -> real-permeability LGCNN
  -> RQ1/RQ2/... QoIs
```

Kernel choice should therefore be locked only after the real fields have been
processed and the held-out reconstruction diagnostics have been inspected.
