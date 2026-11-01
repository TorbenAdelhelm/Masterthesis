# Three-kernel conditional-field study

For roughly **30–35 GB per reference**, use the new
[compact production workflow](compact_kernel_study.md) and
`configs/controlled_field.kernels.compact.500.yaml`. It preserves the scientific
study and full native temperature analysis, with selected input retention.

The full-archive preset described below is `configs/controlled_field.kernels.500.yaml`.
It selects one reference RUN, uses its matching pressure/material/heat-pump
inputs, and generates **500 draws for each of Gaussian, Matérn-3/2 and
exponential covariance, at 64, 128 and 256 nested synthetic readings**.
The four complete source cases are RUN_0, RUN_2, RUN_3 and RUN_4.
The earlier `controlled_field.references.*` presets remain historical Matérn
baselines. They are not the three-kernel study.

## Structural distribution and fitting decision

For each source map, let Y=log10(k/[m²]). Its full-map mean and population spatial
variance define the target mean μ and finite-map heterogeneity v. The source
map informs these parameters; it is not multiplied into each generated field.
The main experiment draws an independent hidden truth from the declared prior,
reveals only the selected cells, and generates conditional draws. These are
synthetic exact readings, not recovered historical borehole measurements.

The independent directional-variogram fits of all three families are preserved
under `independent_family_fit` in `structural_profiles.kernels.json`. They use
the same 40 direction/lag combinations, log-semivariance criterion and bounds
as the earlier structural fit. All exponential fits hit the 12,800 m upper
length bound; their log-variogram RMSE is 1.53–2.61. That poor fit must not be
interpreted as a well estimated geological exponential range. At those lengths,
the common 8192² FFT extension also fails its covariance check.

The production experiment therefore uses a **controlled roughness comparison**:

1. Fit the Gaussian family independently to each processed source map, using
   the recorded directions, lags and log-semivariance criterion. The smooth
   Gaussian anchor is a structural description of the interpolated maps.
2. Keep its two anisotropic lengths and orientation fixed across the three
   kernel families. Under the implemented conventions, **all three have
   integral correlation area 2πℓ_rowℓ_col**. Thus this preserves correlation
   area and anisotropy, not pointwise correlations or practical ranges.
3. For each family compute q=(1/M²)Σᵢⱼρ(xᵢ−xⱼ) on the exact native grid and
   set σ²=v/(1−q). This holds expected finite-map spatial variance fixed.
   It does not normalize individual realizations.

With r the anisotropic rotated separation, the correlations are
exp(−r²/2), (1+√3r)exp(−√3r), and exp(−r), respectively. All parameter
values, source/provenance hashes, independent fits and achieved variograms
are in the versioned catalog. The exponential matched profile remains a
roughness sensitivity assumption; its short-lag mismatch is explicitly retained.
This is not three equally good geological calibrations or a probability mixture.
Historical fitted Matérn catalogs remain unchanged.

The resulting finite-grid law is Y~N(μ1,σ²R), k=10^Y. Given exact data d at B,
Y|d is Gaussian with mean μ1+R_XB R_BB⁻¹(d−μ1) and covariance
σ²(R_XX−R_XB R_BB⁻¹R_BX). Samples use the Matheron update
Y*=μ+σZ+R_XB R_BB⁻¹(d−μ−σZ_B), with an independent Gaussian FFT draw Z.
There is no nugget, clipping, draw filtering, fitted measurement noise or KL
truncation. The FFT extension is checked numerically on the native grid.
Exact-observation matrices that lose numerical rank cause a failure.

Independent FFT seeds provide ordinary Monte Carlo samples within each case.
Common driving modes and nested observation locations pair comparisons across
counts, kernels and separately selected RUNs; these comparisons are dependent
by design. Each family has
its own law-consistent hidden truth and readings. They do not condition three
different kernels on one fixed historical dataset. No kernel probabilities are
assigned. The RUN reconstruction challenge remains a separate realism test.

## Run on Linux

Use the current feature branch, install the package, and extract **both** asset
bundles into the repository root: the original Linux assets and the matching
reference-case supplement. See [reference selection](reference_selection.md).
The model template is `configs/frozen_temperature.by_reference.example.yaml`;
its paths resolve from the repository root. Keep the same code, dependencies,
configuration and assets while a run is active or being resumed.

For a fresh production output:

```bash
subsurface-uq-controlled-fields \
  --config configs/controlled_field.kernels.500.yaml \
  --reference-id RUN_4 \
  --output-dir run_output/kernel_study \
  --stage all
```

This executes validation, input generation, LGCNN propagation and reporting.
Use `controlled_field.kernels.pilot.yaml` and a different output parent for the
eight-draw test first. Its 99 model calls exercise all three kernels and counts.
The pilot shares the production geometry and random-stream prefixes.

To continue an interrupted or plan-only run, repeat the **same arguments** with
`--stage resume`. This verifies the saved context before continuing. A completed
run returns its status. `--stage plan` writes only a manifest. `all` requires a
fresh reference output directory and never overwrites an existing experiment.
An older plan made with changed code/configuration needs a fresh output parent.

The production preset also includes a 500-draw zero-observation baseline for
each kernel: **4,500 requested conditional calls + 1,500 baseline calls + 3
hidden-truth calls = 6,003 LGCNN calls per selected reference**. The preset has
one hidden truth and one balanced geometry. Five hundred inner samples improve
conditional Monte Carlo precision; they do not provide 500 independent truths
for coverage calibration. More outer truths are a separate study decision.

## Storage, recovery and outputs

All native input and output samples are retained. Native permeability alone
uses 157.3 GB (146.5 GiB), and 1028² float32 temperatures use 25.4 GB (23.6 GiB).
The temporary float64 prior scratch adds 26.2 GB (24.4 GiB), for one kernel/truth
at a time, and is removed after its inputs are committed. Allow about **230 GB
free disk space** including summaries, browser images and filesystem overhead.
Repeated reference studies each need their own retained storage.

The production path whitens spatial blocks once and applies all prefixes/draws
together. It avoids the older 12.5 GiB grid-by-observation cache. FFT draws are
generated sequentially into scratch; a restart verifies and continues the
recorded sample indices. An interrupted conditioning pass restarts that outer
trial. Completed inputs are hashed. LGCNN predictions resume per sample without
regenerating permeability. This remains a sequential model runner, not a Slurm
or multi-GPU scheduler. Runtime depends on the machine and streamline costs.

Results are isolated under `run_output/kernel_study/RUN_4/`:

- `manifest.json`: effective laws, selected physical context, hashes, seeds,
  estimated input/scratch storage and planned/finished call counts.
- `dense_validation/summary.json`: native FFT representation, conditioning
  diagnostics and independent dense-Gaussian checks. Dense validation uses 32
  draws in 256 independent trials; those draws are separate from production.
- `<profile>/truth-0000/<design>/n-0064/` (and 0128/0256):
  `permeability_samples.npy`, `temperature_ensemble.npy`, per-draw records,
  analytic log-space moments, MC temperature moments/quantiles, risk maps,
  QoIs, sample-budget diagnostics and spatial `preview.png`.
- `ensemble_browser.html` plus `browser/`: offline selection of **every** sample
  in every kernel/count, including inputs, outputs, errors against hidden truth
  and deviations from conditional means. Display uses point subsampling and
  common colour scales; native arrays retain full resolution. LGCNN outputs
  cover a central crop, so input and output panels have different extents.
- `report.md`, `summary.json`, `profile_sensitivity.json` and figures:
  conditional spread and paired structural-profile contrasts, without pooling
  kernels or treating correlated pixels as independent validation cases.

The reported uncertainty concerns the assumed permeability distribution through
the frozen LGCNN. It excludes uncertain weights and discrepancy against PFLOTRAN.
It is not a calibrated site posterior or proof of reconstruction-error coverage.

## Native implementation checks (2026-11-01)

All four production configurations were planned and validated at the actual
2560² grid with all three kernels and all three positive observation prefixes.
The common 8192² FFT embedding passed every covariance check. Independent dense
Gaussian equations agreed with production conditioning as follows:

| Reference | Largest dense mean/variance discrepancy | Largest requested-lag FFT covariance discrepancy |
|---|---:|---:|
| RUN_0 | 3.25e-12 | 3.87e-14 |
| RUN_2 | 7.64e-14 | 3.07e-14 |
| RUN_3 | 1.78e-15 | 1.51e-14 |
| RUN_4 | 1.78e-15 | 1.78e-14 |

Gaussian RUN_0 at 256 observations is the most demanding matrix, with condition
number about 7.49e9. It passes the existing 1e-12 relative-rank guard and the
explicit interpolation/variance checks. The guard remains enabled for changed
geometries and counts; no nugget or shorter replacement length was introduced.
These are numerical/model-consistency checks, not independent geological
validation or execution of the full 500-draw LGCNN study.

A separate native integration check retained four input draws per kernel/count
for RUN_4 and performed **12 actual frozen-LGCNN queries**: the three independent
hidden truths plus the first conditional draw at 64/128/256 observations for
each kernel. All outputs were finite 1028² fields. Per-query hashes, timing and
normalization diagnostics are in
[the forward-check record](kernel_density_native_forward_checks.json).
This verifies the real model interface; it is not a completed 500-draw analysis.
