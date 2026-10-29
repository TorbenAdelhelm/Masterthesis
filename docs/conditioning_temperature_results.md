# Conditional FFT → LGCNN density pilot (2026-10-24)

The native pilot executed implementation commit `d31e5928f606a94407833a277d854e562e5220d5` on working branch `feat/reference-field-uq`.
All three 2560 × 2560 references use the separate Gaussian 40–2560 m texture fits,
`sigma_R=0.05`, reference-median centering, 5 m cells and exact synthetic readings.
The pretrained real-K release25 weights, RUN_2 pressure/material/source inputs and
8 CPU threads are fixed. Covariance transfer and amplitude remain model-form assumptions.

## Executed study

- Three separate references; one sequential variance-guided location policy.
- Two independent maximum designs per reference; nested reading counts 0, 4, 8.
- Four independent conditional draws per count; sample checkpoints 2 and 4.
- **18 cases, 72 generated-input LGCNN evaluations, three nominal evaluations.**
- 18 retained native permeability/temperature pairs, comparison PNGs, posterior input SD,
  temperature mean/SD/quantiles, warming-risk maps and receptor/area summaries.
- Every draw contributes diagnostics; no rejection, clipping or support-based filtering.

The committed [pilot configuration](../configs/conditioning_temperature.pilot.example.yaml)
and [workflow](conditioning_temperature.md) reproduce the study after local model paths are set.
Arrays are local in `run_output/conditioning_temperature_pilot_20261024` and are not committed.

## Observed temperature uncertainty

The table uses four draws per fixed design. Its RMS SD is the square root of
the mean across designs of spatial mean conditional temperature variance.
Reduction compares those variance estimates with the zero-measurement case.
These are exploratory finite-sample estimates, not converged reduction factors.

| Reference | Measurements | RMS temperature SD [°C] | Estimated variance reduction |
|---|---:|---:|---:|
| RUN_1 | 0 | 0.043905 | 0.00% |
| RUN_1 | 4 | 0.040129 | 16.46% |
| RUN_1 | 8 | 0.037405 | 27.42% |
| RUN_2 | 0 | 0.061602 | 0.00% |
| RUN_2 | 4 | 0.056114 | 17.02% |
| RUN_2 | 8 | 0.048755 | 37.36% |
| RUN_3 | 0 | 0.041350 | 0.00% |
| RUN_3 | 4 | 0.041753 | -1.95% |
| RUN_3 | 8 | 0.042282 | -4.56% |

`density_summary.json/.csv` retains all receptor/ROI targets, paired reductions
and SEM across design estimates. Per-design `risk.json` reports the 0.1 and 1 °C
warming thresholds, exact binomial 95% intervals, expected/conditional excess
and exceeding-cell areas. Zero observed events retain a nonzero upper interval
bound, and conditional severity is null if no event occurs. Source-connected
plume length/merging are not evaluated.

Analytic input variance decreased at every conditioning step, while estimated
temperature variance did not decrease for RUN_3. Local receptor curves can also
be non-monotone, and two- versus four-draw estimates differ substantially.
This pilot cannot distinguish a population response effect from finite MC/design
error. Increase budgets before interpreting these curves as measurement recommendations.

## Verification

- Full local Python 3.12 suite: **569 passed**, no failures/errors/skips. Two existing
  constant-field skewness/kurtosis precision warnings remain; new focused suite: 80 passed.
- After the presentation changes, 14 density/risk regression tests passed.
- Built the wheel and exercised the installed new console command in workspace-only paths.
- Independent audit verified every saved artifact and implementation/model/reference hash,
  nested frozen readings, shared field seeds and monotonically decreasing analytic input variance.
- Every retained conditioned field exactly reproduces the reference float32 values at its
  selected cells. Saved receptor samples agree with the actual saved temperature fields.
- Risk maps/areas satisfy their integration identity; receptor intervals and
  expected-excess = probability × conditional-excess identities were checked.
- All three nominal temperature arrays are bitwise identical to the independently
  saved nominal arrays from the earlier matched-kernel LGCNN study.
- Native input/temperature comparison and uncertainty PNGs were inspected visually.
- **CI passed on Python 3.11 and 3.12**, including the installed new command. Run for the implementation:
  [GitHub Actions](https://github.com/TorbenAdelhelm/Masterthesis/actions/runs/40126204373).

Subsequent presentation changes shorten crowded panel titles and constrain probability
color bars to [0,1] (SD/width bars to nonnegative ranges). They do not change saved
pilot predictions or statistics; the frozen pilot retains its original source/artifact hashes.

The workspace audit is `analysis/conditioning_temperature_20261024/validation.json`.
The complete native pilot manifest contains checkpoint/normalizer/upstream/fixed-input
hashes, configurations, seeds and the artifact ledger.

## Limits and next evidence

Four fields and two designs do not establish sampling convergence, stable tail probabilities,
an optimal location policy or predictive accuracy. Prefix-to-largest convergence has a zero
last error by construction. Increase field and design budgets independently and examine
local QoIs as well as the field average. No physical pass/fail cutoff is imposed.

The complete reference is already the prior log-mean. This is uncertainty around a known
structure, not reconstruction using only n revealed pixels. The readings are synthetic;
real borehole alignment and historical kriging uncertainty remain unverified. Normalization
range checks and nominal patch agreement do not certify LGCNN training support.

This establishes the direct MC bridge for RQ1 and initial RQ2 risk products. A matched
reduced representation for RQMC/PCE and RQ3–RQ4 attribution, source-connected RQ2 geometry,
and predictive-measurement temperature-based RQ5 EVR remain subsequent experiments.
