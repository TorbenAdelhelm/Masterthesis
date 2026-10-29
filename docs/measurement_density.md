# Sparse-reading reconstruction and temperature uncertainty

This is the RUN-based **realism challenge**, alongside the main
[controlled conditional-field benchmark](controlled_field_benchmark.md).
Its validation configuration is `configs/measurement_density.validation.yaml`. It
tests whether a conditional permeability law represents reconstruction errors
at spatially withheld locations before allocating full native ensembles or
querying the frozen LGCNN. Pressure, material/source configuration and model
weights stay fixed. Only permeability varies. References and coverage scenarios
remain separate; no geological scenario probabilities are inferred.

## Information available to the model

1. Reserve complete spatial test blocks using geometry and a recorded seed.
   Exclude these blocks and a declared buffer from all candidate observations.
2. Draw nested balanced maximin or clustered layouts outside the exclusion.
   Repeat each coverage scenario with independent location seeds.
3. Reveal exact readings at the smallest measurement count. Select a covariance
   from explicitly declared candidates using spatial-block cross-validation of
   **only those readings**. Score each observation block equally with normal CRPS.
4. Freeze that family, correlation lengths, rotation and amplitude for all larger
   counts in the same design. Future readings and test values never select it.
5. At each count, ordinary kriging estimates its constant mean from that count's
   revealed readings and retains uncertainty in the mean. Predict on the same
   withheld spatial blocks and compare errors with predicted uncertainty.

The full reference supplies synthetic readings and separate test scores. Neither
its full spatial pattern nor its spatial mean enters the prior. The candidate
grid is external modelling information, not an estimate from the full reference.
The included isotropic 500/1500/3000 m length assumptions and 0.1/0.2/0.3 log10
amplitudes are a finite comparison grid. They do not cover every geological law,
and covariance-selection uncertainty is not included in conditional variance.

An unknown constant mean with a flat prior requires at least one reading, so this
path has no zero-reading case. The selected covariance is frozen at the smallest
budget to avoid borrowing information from a future measurement budget. A
different experiment could re-estimate parameters at each budget, but would
require explicit model-selection uncertainty and different pairing claims.

The earlier `configs/measurement_density.study.yaml` uses full-reference texture
fits and reference-informed screening. It remains available for reproducibility
as a benchmark. Its scores do not provide the independent validation described
here, and its legacy mode has no adequacy gate. Use the validation configuration
for the primary sparse-information experiment.

The [completed validation study](measurement_density_results.md) records 15 of
54 cases meeting its exploratory criteria. Its failed assessment prevents
temperature propagation for that configuration.

## Current validation design

- Three separate native 2560 x 2560 reference arrays at 5 m.
- Counts 32, 128 and 512, with three layouts for each of balanced and clustered
  coverage: 54 reconstruction cases.
- A shared 25% random selection of 1600 m spatial blocks per reference and a
  320 m observation exclusion buffer. The location candidate lattice is 160 m.
- Four spatial selection folds, using only the first 32 readings. If fewer than
  four observation blocks are occupied, use the number actually occupied and
  record it. Fewer than two blocks is an error.
- Reconstruction scoring at 80 m point support in the test blocks. These are
  point samples of the reference and covariance, not block-averaged observations.

Outputs include RMSE, bias, RMS predictive SD, normal CRPS/NLPD, standardized
error, 50/80/90/95/99% interval coverage, spatial-block summaries and uncertainty
bins. Test point predictions are retained in NPZ files for independent checking.
Sampling error is not involved in these analytic Gaussian input diagnostics.

The configured exploratory checks require at least eight evaluated test blocks,
equal-block 90% coverage between 0.80 and 0.98, standardized RMS error between
0.5 and 2.0, and absolute standardized bias no greater than 0.5. These explicit
tolerances flag gross mismatch or unnecessarily broad predictions. They are
not physical safety limits, formal hypothesis tests or a calibration certificate.
Spatial pixels and nearby blocks are dependent; no independent-pixel binomial
confidence interval is applied to reconstruction coverage.

All configured cases must pass before the validation workflow prepares native
ensembles or loads the LGCNN. A selected model that becomes numerically singular
at a larger count is recorded as unavailable; it is not replaced using the test
results. Failure writes an analysis with status `validation_failed`. It does not
delete fields, alter parameters, relax exact readings or silently continue.

Test results must not be reused to tune the candidate grid and then described
as an untouched test. A revised model needs a fresh assessment. Results describe
the supplied reference windows and declared layouts; these are not independent
geological realizations simply because their RUN names differ.

## Run and recover

From the repository root, with the reference arrays and scientific environment
available:

```bash
python -m subsurface_uq.experiments.measurement_density \
  --config configs/measurement_density.validation.yaml \
  --output-dir run_output/measurement_density_validated \
  --stage validate
```

Inspect `reconstruction_validation.md`, `.csv`, `.json` and `.png`. This stage
requires no model assets. If all declared checks pass, the same frozen study can
continue after setting the existing model YAML and asset paths in its configuration
**before validation**:

```bash
python -m subsurface_uq.experiments.measurement_density \
  --config configs/measurement_density.validation.yaml \
  --output-dir run_output/measurement_density_validated \
  --stage inputs

python -m subsurface_uq.experiments.measurement_density \
  --config configs/measurement_density.validation.yaml \
  --output-dir run_output/measurement_density_validated \
  --stage temperatures

python -m subsurface_uq.experiments.measurement_density \
  --config configs/measurement_density.validation.yaml \
  --output-dir run_output/measurement_density_validated \
  --stage report
```

`--stage all` runs this sequence and stops after the written validation analysis
if the checks fail. The installed alias is `subsurface-uq-measurement-density`.
Configuration/source/reference hashes and validation artifact hashes are checked
before continued execution. Changed scientific configuration requires a fresh
output directory. Old frozen studies remain unchanged.

Native input generation checkpoints each prior draw and each completed reading
count, verifies recovered arrays, and recomputes an interrupted count using the
same prior. Geometric features can be recomputed after interruption; recovery is
not at individual spatial chunks. Temperature predictions are saved and hashed
one at a time. Completed predictions and nominal temperature caches are verified
before reuse. A failed prediction is never replaced with a different input.

The native generator uses the selected full radial covariance through validated
FFT sampling and ordinary-kriging conditioning, with exact zero-noise readings.
Draws are shared across counts within one design/covariance. Different designs
can select different covariances and remain separate. Temporary geometry/prior
files are removed after successful completion; complete native input ensembles
remain local and can require substantial disk space.

## Temperature results and limits

On passing validation, temperature products include field mean/SD/quantiles,
receptors, ROI mean anomaly, warming exceedance probability/severity/area and
regional maximum absolute deviation from the nominal reference prediction.
The regional maximum is taken within each realization before its VaR/expected
shortfall is calculated. Independent half-batches and sample prefixes expose
finite-MC variability. With 32 draws, tail convergence is not established.

Receptor/ROI coordinates and areas refer to the model's valid temperature crop,
not the full input domain. The real-K pilot crop is 1028 x 1028, offset by 766
native cells per side. Source-connected plume lengths, output-informed borehole
placement, Sobol attribution and PFLOTRAN discrepancy are separate work.

The mathematical unknown-mean convention follows [ordinary kriging](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/05_kriging/01_ordinary_kriging.html).
Separating model selection from final evaluation follows the usual
[held-out evaluation principle](https://scikit-learn.org/stable/modules/cross_validation.html).
