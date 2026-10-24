# Conditional reference generator: validation on 2026-10-24

The component draws synthetic measurement locations and readings from one fixed
reference, then generates a Gaussian log-permeability field conditional on that
frozen design. The reference itself does not supply a recovered historical
kriging covariance. Covariance, residual amplitude and location policy remain
explicit assumptions. See [workflow and Git Bash commands](reference_conditioning.md).

## Mathematical and software checks

- 41 conditioning/location-policy tests cover dense analytic Gaussian posteriors,
  exact interpolation, both mean conventions, independent random streams, all
  covariance families, rotation, constraints, invalid configurations and numerical
  rank loss. A 1,600-draw noisy Matheron test checks empirical means and covariance.
- Seven CLI tests exercise separate frozen designs, deterministic retained fields,
  output/diagnostic files, overwrite refusal, strict configuration validation,
  and the within/between-design total-variance identity in noisy reconstruction mode.
- Three rotated FFT regressions independently check signed off-axis covariances
  and empirical sample covariance. The full focused regression command reports
  **58 tests passed**, including existing FFT/matched-study regressions.
- The installed console command was built and checked in a workspace-only prefix;
  the user's global Python environment was not changed.
- The complete local suite passed: **546 tests, zero failures/errors/skips** on
  Python 3.12. Existing constant-field skewness/kurtosis checks emit two precision
  warnings. Legacy Perlin dependencies and installed commands were supplied in
  workspace-only paths for this run. CI retains Python 3.11 and 3.12 and checks
  the newly installed conditioning command.

The observation solver uses whitening rather than an explicit covariance inverse.
Numerically rank-deficient designs are rejected before allocating the FFT prior.
Exact-data residual checks terminate on a numerical failure without replacing a draw.

## Native-grid demonstration

The local output directory is `run_output/conditional_reference_pilot_20261024`.
The portable configuration is `configs/reference_conditioning.matched_pilot.yaml`.
With the prepared reference arrays available, reproduce from the repository root
in Git Bash (use a fresh output directory):

```bash
export PYTHONPATH=src
python -m subsurface_uq.experiments.reference_conditioning \
  --config configs/reference_conditioning.matched_pilot.yaml \
  --output-dir run_output/conditional_reference_pilot_repeat
```

Each of RUN_1/RUN_2/RUN_3 uses its actual 2560 x 2560 reference on the native
5 m grid and its separate Gaussian correlation-length fit over 40–2560 m from
the frozen matched study. Zero covariance rotation and `sigma_R=0.05` log10
are retained as assumptions; the latter gives unconditioned variance 0.0025.

The experiment compares uniform, sequential variance-guided, and
gradient/variance hybrid policies. Each policy has two independent location
designs per reference, 12 exact synthetic readings per design, and two
conditional field draws per design: **18 designs and 36 generated fields**.
The location candidate stride is 32 cells, the margin 8 cells, and minimum
spacing 500 m. Candidate subsampling does not coarsen the generated fields.

| Reference | Uniform variance reduction | Variance-guided reduction | Hybrid reduction |
|---|---:|---:|---:|
| RUN_1 | 50.09% | 51.39% | 51.46% |
| RUN_2 | 60.30% | 63.46% | 61.15% |
| RUN_3 | 11.53% | 11.32% | 10.65% |

These numbers describe the spatially averaged analytic conditional variance,
averaged over the two sampled designs, relative to the assumed unconditioned
variance. They are **not converged rankings** or temperature-QoI uncertainty
reductions. Two designs cannot establish the best location prior. The weaker
reduction for RUN_3 is consistent with its shorter assumed correlation lengths.

Independent verification of the saved outputs established:

- All 18 retained sample fields exactly reproduce the original float32 reference
  values at their 12 selected cells (216 checked synthetic readings).
- All 36 generated fields contributed diagnostics; no draws were filtered.
- Posterior mean/variance/SD maps retain the native shape. Every default exact
  design keeps the reference log-mean, as required by its zero innovation.
- The stored within + between = total design-mixture variance identity holds.
  Between-design mean variance is zero in this particular mean/noise convention;
  different designs nevertheless have different covariance maps.
- Reference and implementation hashes match the frozen study manifest.
- Saved PNGs for all three references were visually inspected. Each contains the
  reference, conditional center, sample, log difference, conditional SD and locations.

The local audit is recorded in
`analysis/conditional_reference_pilot_20261024/validation.json` at the project
workspace root. Source arrays and large study outputs are local artifacts and
remain outside Git. Distributions, directional variograms, spectra, native
gradients, patch descriptors and reference agreement are descriptive diagnostics;
training-support compatibility remains unassessed without a verified baseline.

## Scientific limits and next experiments

This establishes the conditional-generator and synthetic design-prior workflow.
The native demonstration does not evaluate the LGCNN, identify measurement error,
recover real well locations, or demonstrate integration/design convergence.
The exact reference-mean experiment varies covariance around a known structural
baseline. `global_log_mean` is available for a separately labeled sparse-reference
reconstruction experiment. Positive-noise synthetic experiments also vary the
drawn reading values. Neither is an independent geological calibration.

Increase design replicates to study the location-prior expectation, increase
field draws for conditional integration precision, and query the fixed LGCNN
within each frozen design for temperature-QoI comparisons. The resulting
design mixture should not be treated as one Gaussian posterior or one Hermite
response PCE law. Reference RUNs and distinct location priors remain separate.
