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

## Leave-one-out DaRUS calibration inspection

The `evaluate` subcommand is the recommended model-selection experiment before
RQ2. It keeps one real permeability field hidden, calibrates all covariance
candidates on the remaining real fields, uses the same synthetic boreholes for
all models, and compares the resulting conditional reconstructions.

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
  --n-samples 32 \
  --batch-size 1 \
  --seed 3901 \
  --n-modes 128 \
  --observation-std-log10-k 0 \
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
    sample_001_log10_k.npy
```

`model_comparison.csv` reports fitted `ell_x`, `ell_y`,
`ell_x/ell_y`, total/directional variogram RMSE, held-out conditional-mean
RMSE/MAE, Gaussian 90% posterior coverage, and conditioning residuals.

The 90% coverage in this inspection experiment uses
`mean +/- 1.64485 * std` in `log10(K)`. This avoids storing the full
`[N,H,W]` ensemble during model comparison and is appropriate for the
conditional Gaussian models being compared. The separate `generate` command
still computes empirical ensemble quantiles when full realizations are stored.

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
