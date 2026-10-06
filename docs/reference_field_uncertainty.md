# Reference-centered lognormal input uncertainty

This primary computational scenario family describes local input uncertainty
around a realistic reference field. The original measurement-kriging CLI defaults
remain available as a legacy/comparison baseline for reproducibility. It does not
claim to recover the unique stochastic geological process or calibrate the error
of the interpolation merely by fitting spatial texture.

## Model and source of each component

For one selected reference/RUN scenario:

$$Y(x;\eta)=m_{\rm ref}(x)+B_r(x)\eta,\quad\eta\sim N(0,I_r),\quad k=10^Y.$$

* `m_ref = log10(k_ref)` comes from one physical permeability input field.
  Different RUNs remain different reference scenarios, not aligned repeated
  observations for pixelwise covariance/PCA.
* `B_r` is a KL factor of an explicit zero-mean Gaussian residual covariance.
  The existing separable Matérn-3/2 and exponential kernels are supported.
  Residual standard deviation is specified separately from reference variance.
* Reference-texture variogram fits can provide a covariance-shape proxy. They
  do not identify interpolation error covariance or its amplitude. No spatial
  trend is removed in that descriptive fit.

Adding variation to an already textured field can inflate its heterogeneity.
Residual amplitude therefore remains an uncertainty assumption/sensitivity
parameter. Resimulating texture around a smooth trend is a different candidate.

The default reference is the unconditional pointwise median/geometric mean.
With `--center arithmetic-mean`, the deterministic mean instead is

$$m_{\rm ref}(x)=\log_{10}k_{\rm ref}(x)-\tfrac12\ln(10)v_r(x),\qquad
v_r(x)=(B_rB_r^T)_{xx}.$$

This uses actual truncated variance and preserves the unconditional arithmetic
mean. Conditioning changes either center. Fixed-location marginals are
lognormal; pooling over a spatially varying reference produces a mixture whose
histogram need not be lognormal.

## Conditioning and data conventions

At observation points, A evaluates the same retained residual KL basis. Use

$$d_R=d-m_{\rm ref}(x_d),\qquad
\mu_\xi=A^T(AA^T+N)^{-1}d_R,\qquad
S_\xi=I-A^T(AA^T+N)^{-1}A.$$

Then `xi=mu_xi+L eta`, `LL^T=S_xi`. Continuous conditioning requires positive
noise standard deviations and does not claim exact interpolation. Rebuild the
posterior for each truncation. The deterministic log reference is interpolated
bilinearly between grid cell centers, with constant edge extension within the
half-cell margin. Observation coordinates must be local `(y,x)` metres inside
the generated domain. No geographic alignment is guessed.

Historical training inputs use `k=K_h/7.5e6`. Measurement conversion defaults use
`k=K_h*mu/(rho*g)`, giving ratio 0.76769859 at the default properties. The converter
now accepts `permeability_convention="historical-training"` explicitly; its
existing physical default is retained. Reference/observation conventions must
match. Original input HDF5 loading honors one-based `Cell Ids` and includes
input-only RUN_1. The actual training split must still be selected explicitly.

The exact TIFF identity and historical extraction conventions are recovered.
That does not prove successful numerical reconstruction of every RUN from this
TIFF: the strict reconstruction currently rejects invalid source-window cells
for some RUNs. The validation report records those failures and any computable
mismatch. It does not fill missing values to force agreement. The earlier
measurement-to-surface interpolation recipe remains unproven. Reusing the same
measurements as an independent update to their own interpolated reference needs
a dependence argument and cannot count as independent validation.

## Generate and compare scenarios

Install the package in the experiment environment with `pip install -e
".[test,perlin]"`. In Git Bash on Windows, for example:

```bash
python -m subsurface_uq.experiments.reference_field_permeability \
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" \
  --reference-run RUN_1 --permeability-convention historical-training \
  --roi 640 1920 640 1920 --coarsen-factor 8 \
  --residual-std-log10-k 0.05 \
  --length-scale-y-m 500 --length-scale-x-m 800 \
  --energy-thresholds 0.95 0.99 0.999 --n-samples 64 \
  --output-dir run_output/reference_field_example
```

These example scales/amplitude are sensitivity choices. Geometric block means
put the coarse reference at block centers; they differ from CNN stride
decimation. This ROI/factor studies a 6.4 km square at 40 m resolution, not a
full-resolution LGCNN scenario. Omit ROI/coarsening to produce the original
5 m grid, using `--diagnostic-factor 8` for coarser matched-support comparisons.
Samples stream into disk-backed `generated_fields.npy`.

Each energy directory contains checksum-verified schema-3 input law/reference
artifacts, generated physical fields, and `fidelity.json`. Diagnostics include
log-marginal quantiles/Wasserstein distance, x/y and both diagonal variograms,
and Hann-tapered 2-D/radial/angular spectra. Summary output records dimension,
retained energy/local variance and actual perturbation RMS. Axis power is a
descriptive indicator, not a universal rejection threshold; samples are not
filtered or clipped to force compatibility.

Optional `--observations` accepts a YAML mapping:

```yaml
permeability_convention: historical-training
yx_m: [[100, 200], [300, 450]]
log10_permeability_m2: [-9.5, -9.3]
std_log10: [0.08, 0.08]
```

The observation coordinates refer to the selected ROI domain. Values must
already be converted and aligned.

## Reproduce the local-data validation

```bash
python -m subsurface_uq.experiments.reference_field_validation \
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" \
  --reference-runs RUN_1 RUN_2 RUN_3 \
  --parent-tif "C:/Users/Torbe/Desktop/MT/Daten/tiffs/originals/Hydraulic_conductivity_20m_resolution.tif" \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --residual-stds 0.05 0.1 --n-samples 64 \
  --output-dir run_output/reference_field_validation
```

The workflow checks parent reconstruction, fits reference-texture covariance
candidates and evaluates three truncations per RUN/amplitude. Parent/measurement
discrepancies are a descriptive audit, not held-out uncertainty calibration.
Parent and RUN references are dependent. See the recorded results in
`reference_field_validation_results.md` for practical limits of this study.

## MC, RQMC and Hermite response PCE

Set `grf.input_model` to a schema-3 YAML artifact in the existing RQ1 config and
remove manual GRF parameters. Field shape and physical grid spacing must match
the pretrained-model scenario. MC and scrambled-Sobol Gaussian RQMC share this
map. RQMC sample vectors are dependent, while the target measure is a product
Gaussian distribution.

Gaussian response PCE uses normalized probabilists' Hermite products:

$$Q(\eta)\approx\sum_{|\alpha|\le p}c_\alpha
\prod_j\mathrm{He}_{\alpha_j}(\eta_j)/\sqrt{\alpha_j!}.$$

The Gaussian LHS training design has separate iid Gaussian validation data.
Mean is `c_0`; variance is the sum of squared nonconstant coefficients. Use
`python -m subsurface_uq.experiments.release25_gaussian_pce --help` for the
pretrained LGCNN CLI. It requires the serialized input law, release25/model and
fixed-scenario paths, sample budgets and actual background temperature. Real-K
models are the default (`--no-random-k`); synthetic model use is explicit.

The original Legendre proof of concept remains available. Empirical normal-score
laws use their direct inverse transform; PCE approximates a temperature response,
not that inverse CDF. The RQ1 normal-score unconditional branch now also returns
properly inverse-transformed physical permeability.

Index enumeration generates only admissible total-degree terms. Still,
`comb(r+p,p)` grows rapidly: the CLI rejects undersampled designs before loading
checkpoints. Large-dimensional PCE needs QoI/dimension sensitivity or sparse and
adaptive methods, rather than inventing geological degrees of freedom. Energy
0.95/0.99/0.999 is a trace target, not sufficient downstream QoI convergence.

Tests cover retained covariance, arithmetic-mean correction, noisy posterior
moments, serialization, input-only loading, spectral normalization and Hermite
response recovery. CI runs the full suite on this branch. The architecture is
primary; no particular residual amplitude or covariance is promoted as calibrated truth.

## Parameterized scenario matrix and expert marginals

`ReferenceScenario` declares RUN, lognormal marginal, centering, `sigma_R` in
log10 units, separable covariance family, `ell_x`, `ell_y` in metres, and exactly
one truncation choice. New artifacts use `reference-centered-lognormal`; old
schema-3 `reference-centered-lognormal-candidate` artifacts still load.

Optionally specify either `physical_mean` and `physical_variance` (m2 and m4),
or `lb`, `ub`, `alpha`, together with `marginal_reference_k` (m2). These scalar
specifications describe the **untruncated pointwise law at that reference value**,
not a pooled spatial histogram. They must agree with the chosen reference center:
the anchor equals the median for median centering or the physical mean for
arithmetic-mean centering. An omitted `sigma_R` is derived; a supplied value must
agree. Conflicting moments/bounds, amplitude, center or truncations are rejected.

Using natural-log parameters, conversions are

$$\sigma^2=\ln(1+v/m^2),\quad\mu=\ln m-\sigma^2/2,$$
$$m=e^{\mu+\sigma^2/2},\quad v=m^2(e^{\sigma^2}-1).$$

For equal-tail bounds, $z=\Phi^{-1}(1-\alpha/2)$,
$\mu=(\ln lb+\ln ub)/2$, $\sigma=(\ln ub-\ln lb)/(2z)$ and
$\sigma_R=\sigma/\ln10$. Thus $P(lb\le K\le ub)=1-\alpha$ at the declared
anchor under the untruncated law. Bounds are never hard support and draws outside
them are retained. A bounded/truncated lognormal mode is not implemented.

For other locations, the reference supplies their center and the residual supplies
relative dispersion. KL truncation changes pointwise dispersion; the mean correction
uses actual retained variance. Bounds/variance are nominal **before truncation**,
not guarantees for each represented marginal. Manifests report actual min/mean/max
retained log10 variance. Conditional artifacts also change marginal moments.

Edit `configs/reference_scenarios.example.yaml` to specify physical reference arrays
and the Cartesian product of RUN, amplitude, covariance, lengths and truncations.
Git Bash:

```bash
python -m subsurface_uq.experiments.reference_scenarios \
  --config configs/reference_scenarios.example.yaml \
  --output-dir run_output/reference_scenarios
```

Add `--rq1-config configs/rq1.example.yaml` after configuring the real-K model,
fixed source scenario and temperature QoIs. This invokes existing RQ1 per scenario;
reference RUN and fixed-source RUN are distinct choices. The same field map also
drives scrambled Sobol Gaussian RQMC and normalized Hermite response PCE. No support
metric alters the target law or rejects a realization.

IDs are hashes of scientific configuration, independent of axis order. Manifests
also record reference/input-artifact checksums, cell size, seed and sample count;
the configuration ID alone does not identify different reference bytes or grids.
Results remain separate. Optional `scenario_weights: equal` or a complete
ID-to-weight mapping is recorded explicitly as an assumption. The runner does not
aggregate results even when weights are supplied; it never infers RUN probabilities.

Optional `training_support` references enable existing exact patch-lattice
diagnostics at the generated grid resolution. The report marks observed descriptor
deviations, or membership within the sampled descriptor envelopes. This is descriptive
and does not prove in-distribution behavior. Without training references, compatibility
is explicitly unassessed. Marginals, directional variograms and 2-D/radial/angular
spectra are always reported on matched diagnostic support. Coarsened diagnostics
do not establish compatibility at the full LGCNN patch resolution.

## Temperature-oriented KL representation sensitivity

The single-reference CLI also accepts `--n-modes 10 20 40` instead of energy
thresholds. Compare unconditioned artifacts differing **only in truncation**:

```bash
python -m subsurface_uq.experiments.reference_kl_sensitivity \
  --input-models run_output/ref/energy_0.95/stochastic_input_model.yaml \
    run_output/ref/energy_0.99/stochastic_input_model.yaml \
    run_output/ref/energy_0.999/stochastic_input_model.yaml \
  --rq1-config configs/rq1.example.yaml \
  --n-samples 64 --method RQMC --seed 4901 \
  --output-dir run_output/temperature_kl_sensitivity
```

This queries the pretrained model with common Gaussian coordinates and reports
receptor temperatures and optional ROI mean anomaly: means, standard deviations,
5/50/95% quantiles, paired RMSE and mean/std differences relative to the largest
dimension. Paired samples are saved. That largest dimension is a comparison reference,
not truth. Repeat budgets and seeds to assess integration error, and choose QoI-specific
tolerances explicitly. No automatic universal convergence criterion is assumed.
The 0.95/0.99/0.999 energy targets remain **representation sensitivities**.
The paired study currently rejects conditioned artifacts; each conditioned truncation
needs its own update and a justified coupling before extending this analysis.

The older `reference_field_validation` study retains reference-texture covariance fits
for reproducibility. Its auto-selected texture proxy is a sensitivity assumption,
not statistical identification of residual covariance. Use the explicit matrix for
the primary model-form comparison.
