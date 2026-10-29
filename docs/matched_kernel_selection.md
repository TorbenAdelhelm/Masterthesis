# Matched kernel selection for separate reference fields

The matched study compares **radial Gaussian, Matérn-3/2 and exponential**
covariances fitted with GSTools separately to each reference. Each kernel has
fits at 40–1280 m and 40–2560 m: 18 configurations across three RUNs. Reference
spatial covariance is texture guidance; transferring its correlation shape to
the residual remains a model-form assumption. Residual variance is not fitted
from the reference's sill. RUNs are never pooled into an inferred posterior.

The law is `K = K_ref * 10**R`, with zero-mean Gaussian `R`, median centering
and assumed `sigma_R=0.05` in log10 units. No hard support, clipping, field
rejection or output-based sample replacement is applied. The heterogeneous
spatial histogram is distinct from each pointwise lognormal law.

## Common native-grid sampler

Direct GSTools `SRF(RandMeth)` is a useful benchmark, but many random-frequency
modes can be costly on 2560² cells and insufficient modes can distort native
roughness. `sampling/gstools_reference.py` evaluates the exact closed forms of
these fitted radial covariances and checks them against GSTools, including
off-axis lags and explicit length rescaling. It samples a common FFT circulant
embedding, without KL or random-frequency truncation.

The requested crop is not made periodic. A C4 window modifies covariance only
outside every row/column lag present in the crop. Every embedding is checked for
positive semidefiniteness. If negative spectral mass exceeds the declared
numerical covariance tolerance (default 1e-11 for unit correlation), the
embedding is enlarged; a materially indefinite embedding is never silently
clipped. The tiny roundoff correction, native-increment representation error,
requested-lag covariance error and embedding size are recorded. This numerical
accuracy tolerance is not a physical acceptance cutoff. An independent Gaussian
spectral draw is made for each recorded seed; realizations are not rescaled to
force spatial means or variances.

This method follows the circulant-embedding construction in
[Dietrich and Newsam (1997)](https://epubs.siam.org/doi/10.1137/S1064827592240555).
The source covariance includes GSTools' fitted radial geometry; it is not
silently replaced by the old separable exponential/Matérn KL covariance.

## Execution in Git Bash

Use the repository environment containing `subsurface-uq[geostat]`. The fitted
JSON is the output of the recorded reference-texture fit, including each native
reference path, SHA-256, fitted lengths, family and fit-window label. The local
study uses the existing real-K model configuration and prepared 5 m inputs.
Reference data, models and study results remain local and are not committed.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# Optional: reproduce the separate GSTools reference fits first.
python examples/reference_pilot/fit_reference_kernel_models.py \
  --reference-config configs/reference_scenarios.kernels.example.yaml \
  --output run_output/matched_fits.json

python examples/reference_pilot/run_matched_kernel_study.py \
  --stage prepare --fits run_output/matched_fits.json \
  --output-dir run_output/matched_kernel_selection_20261019

python examples/reference_pilot/run_matched_kernel_study.py \
  --stage inputs --output-dir run_output/matched_kernel_selection_20261019

python examples/reference_pilot/run_matched_kernel_study.py \
  --stage select --output-dir run_output/matched_kernel_selection_20261019

python examples/reference_pilot/run_matched_kernel_study.py \
  --stage temperatures --rq1-config run_output/reference_pilot/rq1.local.yaml \
  --output-dir run_output/matched_kernel_selection_20261019
```

The prepare stage records all source/reference hashes, two independent seed
batches, runtime versions and fixed budgets (16 input and 16 temperature draws
per case by default). Scientific configuration or code changes require a fresh
study directory. Interrupted input draws and model predictions are resumable;
completed outputs and cached predictions are checksum checked. Full native
input arrays are retained only for fixed picture indices, with their hashes.

## Selection and interpretation

Input diagnostics reuse existing native-gradient, marginal, four-direction
variogram, correlation, radial/angular/2-D spectral and reference-agreement
utilities. Five fixed 1280-cell patches provide nominal structural descriptors;
their training membership is unverified. Native probe moments are checked
against the full radial Gaussian law, never the previous retained KL law.

Within each RUN and kernel family, fit windows are ordered by a declared
lexicographic preference: maximum across directions of mean absolute per-field
native-gradient deviation, maximum
directional variogram relative RMSE, angular spectral error, radial spectral
error. Both seed-batch results and all tradeoffs must be retained. This is an
explicit structural-preservation preference, not a calibrated objective or
inferred model probability. The best window from **each** family receives an
actual frozen-LGCNN screen: nine cases, 144 generated-field evaluations plus
three nominal references. This preserves the fitted exponential comparison
even when its structural scores are worse.

The model path reuses the verified release25 surrogate, real-K streamline
semantics, fixed inputs and temperature/flow descriptors from the earlier
pilot. Every case uses the same declared pressure/material/source context.
Mean map RMSE is accompanied by local receptor mean shifts, standard deviations,
independent seed-batch estimates, plume/flow descriptors and checkpoint range
excursions. Patch envelopes are nominal-reference evidence, not proof of
training membership. Small output variation does not establish predictive
accuracy. Final recommendations prioritize structural preservation and disclose
local temperature sensitivity; no automatic physical pass/fail cutoff or
minimum-output-variance objective is used.

Only after kernel selection should the chosen full covariance be approximated
by a reduced Gaussian-coordinate map for RQMC/PCE and checked for temperature-QoI
convergence. A matched 0.95 energy threshold across kernels alone is insufficient
to establish equal representation accuracy, especially for exponential fields.
