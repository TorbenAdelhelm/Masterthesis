# Conditional permeability → frozen LGCNN: measurement-density study

This study connects the native conditional FFT generator directly to the actual
pretrained release25 LGCNN. It advances RQ1 and supplies initial RQ2 metrics.
No new KL approximation is needed for direct MC. Existing KL/RQMC/PCE workflows
remain available; this command does not interpret an FFT manifest as a KL artifact.

## Experimental axes and information state

- `n_measurements`: maximum synthetic measurement count in the generator YAML.
  The temperature study declares increasing `measurement_counts`, starting at
  zero and ending at this maximum. Zero is the unconditioned **same prior law**.
- `n_samples`: independent full-field draws per fixed measurement design and
  count. Increasing `sample_budgets` are nested prefixes ending at `n_samples`.
  More draws improve integration precision, not geological knowledge.
- `n_designs`: independent repetitions of the location/reading policy. Repeated
  designs quantify design variability plus the finite MC error of each estimate.

For each reference/policy/design, draw the maximum location set once. Smaller
counts use its exact prefixes, including frozen noisy readings. Count does not
enter the field seed: each comparison uses the same prior FFT realization and
the same prefix of the independent noise used by Matheron's update. This is an
explicit common-random-number coupling within one covariance, not an equivalence
between unrelated FFT/KL realizations or different kernels.

The law remains `Y=m+u`, `K=10**Y`, conditioned by the declared exact/noisy
observations. `sigma_R`, covariance transfer from the reference, noise and design
policies remain assumptions. With `reference_median`, all reference pixels are
already in the prior mean; this studies uncertainty around a known structure,
not hidden-field reconstruction from only n observations. `global_log_mean`
also uses a full-reference statistic and must be labeled as informative prior
information. Gradient/hybrid location policies inspect the reference structure.

Each RUN keeps its own reference/covariance. The pressure/material/source setup
and checkpoint weights are held fixed by `fixed_run_id`, which is distinct from
the reference RUN. There is no inferred RUN probability, policy pooling or
posterior pooling. Different pressure/source contexts require separate studies.
References must match the native prepared model grid. A cutout requires matching
prepared model inputs; an arbitrary crop is not silently resized into the network.

## Git Bash execution

From the repository root, use the project Python environment:

```bash
python -m pip install -e ".[geostat]"
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

python -m subsurface_uq.experiments.conditioning_temperature \
  --config configs/conditioning_temperature.pilot.example.yaml \
  --output-dir run_output/conditioning_temperature_pilot_001
```

The installed equivalent is `subsurface-uq-conditioning-temperature`.
First set actual checkpoint/upstream paths in `configs/frozen_temperature.example.yaml`.
Paths resolve relative to each YAML, not the current shell directory. Existing
RQ1 YAML may provide the model's `release25`, `qoi` and `streamlines` blocks;
its `grf`/`sampling` blocks are ignored here because the conditional generator
owns the input law and budgets. For older repository-relative RQ1 asset paths,
set `model_asset_root: ..` in a study YAML located in `configs/`.

The portable reduced pilot has three references, one variance-guided policy,
two designs, counts `[0,4,8]`, and four fields per count (checkpoints 2 and 4):
**72 generated-input LGCNN evaluations plus three references**. This is an
execution demonstration; four draws and two designs do not establish convergence.

After inspecting that pilot, edit and run `configs/conditioning_temperature.example.yaml`.
Its three references, two policies, four designs, five counts and 32 samples
require **3,840 generated-input evaluations plus three references**. Do not use
that larger example as an accidental smoke test. Check sample/design budgets
progressively before interpreting tails or location-policy differences.

Each output directory must be new. Interrupted directories remain partial with
manifest status `running`; this initial command does not resume or overwrite.
The generator and temperature examples are independently configurable. Their
maximum counts/sample sizes must agree; inconsistent values fail explicitly.

## Outputs and interpretation

At the root, `manifest.json` records configuration, implementation hashes,
runtime versions, checkpoint/normalizer/upstream-source hashes, fixed input
hashes and the evaluated model context. Inputs are rechecked before completion.
Large arrays/checkpoints remain local, outside Git.

`RUN/policy/design-NNN/maximum_measurements.csv` freezes the complete reading
sequence. Under each `n-NNNN/`:

- `measurements.csv` stores that count's prefix (absent for zero observations).
- `posterior_log10_std.npy` is analytic input uncertainty on the native grid.
- Temperature mean, SD, q05/q50/q95 and q95−q05 maps describe that fixed posterior.
  Exact empirical quantiles use a temporary disk-backed ensemble with spatial
  chunking. `retain_temperature_ensemble: false` removes that ensemble afterward;
  scalar samples, final products and predeclared picture samples remain.
- `qoi_samples.npy` and `summary.json` report receptor temperatures and optional
  region-mean anomaly, sample SD/variance, quantiles, mean shifts from reference,
  and mean standard errors. Receptors use the actual valid **output** grid.
- `temperature_uncertainty.png` displays reference/mean/difference/SD/interval
  width/exceedance. `sample-NNNN.png` displays the generated permeability and
  actual model temperature beside their references, with shared pairwise color
  scales and symmetric difference scales. Native selected arrays are also saved.
- `reference_diagnostics.json` retains distributional, directional variogram/
  correlation, radial/angular/2-D spectral comparisons on the declared diagnostic
  point-stride grid. Every sample also records full-grid agreement, native short-
  range gradients/correlations and patch descriptors. Diagnostic subsampling
  does not coarsen the generator. No sample is clipped, rejected or replaced.
- Checkpoint normalization-range excursions are descriptive. Reference-relative
  patch agreement and moderate output changes do not certify training membership
  or predictive accuracy. No physical pass/fail cutoff is invented.

`density_summary.json/.csv` and `RUN.policy.density.png` report the average
**conditional variance** across designs for each count and sample checkpoint.
For the field target this is spatial mean temperature variance (`U_RMS**2`);
scalar targets use their conditional sample variance. They also report paired
variance-estimate differences from the zero-measurement baseline. Negative
estimated differences are retained. The SEM across design estimates includes
both design variability and finite MC noise; one design has no SEM. This is not
the variance of a pooled design mixture and not a calibrated posterior over RUNs.

Prefix convergence uses the largest same-design empirical ensemble as its
comparison, not truth. Its last difference is zero by construction. Independent
repetitions and larger budgets are needed to establish numerical convergence.

## Initial RQ2 products

For explicit warming thresholds and a declared `>` (default) or `>=` boundary:

\[
P_\tau(x)=P[T(x)-T_{bg}>\tau],\qquad
E_\tau(x)=E[(T(x)-T_{bg}-\tau)_+].
\]

`risk/` holds probability/expected-excess maps and each draw's exceeding-cell
area, using physical cell area. `risk.json` reports area distributions and
receptor exceedance counts, Clopper–Pearson 95% binomial intervals, expected
excess and conditional excess. Conditional excess is JSON null when no event
occurs; zero observed events do not imply zero event probability. Intervals
apply to independent MC draws **within that fixed design**, not pooled RQMC
points or a mixture over designs. Threshold comparisons match the output dtype
at nominal boundaries. The old accumulator's `>=` behavior is unchanged.

Area includes all exceeding cells in the valid temperature crop. This does not
implement source-connected plume extent, plume merging, planning-distance
probabilities, off-grid receptor interpolation or boundary-censoring corrections.
These remain separate RQ2 additions. The example thresholds are study choices,
not claimed regulatory limits.

## Boundary to RQ3–RQ5

No automatic RQMC/PCE claim is made for the full FFT coordinates. A covariance-
matched reduced map needs a downstream QoI convergence check before those
methods and mode-attribution analyses are applied to the selected native law.

Variance-guided placement is an **input-space benchmark** for RQ5. The command
does not sample hypothetical readings from the current predictive distribution,
average nested posterior temperature variances, or rank locations by expected
temperature-QoI variance reduction. Those steps require a separate declared
baseline information state and candidate-location experiment. PCE is optional
for accelerating that study; direct MC can demonstrate it on a small candidate
set. Real borehole-to-RUN mapping and original kriging uncertainty remain
unverified; the measurements here are synthetic reference readings.
