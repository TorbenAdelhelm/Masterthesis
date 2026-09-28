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
measurements are not clipped or removed from the baseline model. Results are
written to `measurement_upper_tail_robustness.csv`. Set
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
