# Validate generated fields around one reference

Run this diagnostic before using a scenario for LGCNN uncertainty propagation.
It compares a generated ensemble with the **same reference and serialized input
law that produced it**. It reports evidence for scientific review; it does not
filter fields, pool different RUNs, calibrate covariance parameters, or establish
that the fields lie within the LGCNN training support.

The inputs are intrinsic permeability in m². Keep RUN_1, RUN_2 and RUN_3 in
separate validation directories. Those reference scenarios are not iid
geological draws. A spatial histogram across their heterogeneous locations is
also not a pointwise marginal distribution.

## Execute from Git Bash at the repository root

Use the Python environment in which the repository is installed. If necessary:

```bash
python -m pip install -e ".[geostat]"
```

To diagnose an existing `generated_fields.npy`, use its matching
`stochastic_input_model.yaml`. Replace `path/to/scenario` with the actual
scenario directory:

```bash
python -m subsurface_uq.experiments.reference_field_diagnostics \
  --input-model "path/to/scenario/stochastic_input_model.yaml" \
  --generated-fields "path/to/scenario/generated_fields.npy" \
  --output-dir "run_output/reference_validation/RUN_2_existing" \
  --diagnostic-factor 8
```

This reads the saved arrays without changing them. Existing files do not prove
how the sampling design was made. If it is known, add `--method MC` or
`--method RQMC` to record that assertion. The file mode does not accept
`--n-samples` or `--seed`, because those would imply that a new design was
generated.

To produce and diagnose an explicit 32-field validation design from the saved
law, without querying the LGCNN:

```bash
python -m subsurface_uq.experiments.reference_field_diagnostics \
  --input-model "path/to/scenario/stochastic_input_model.yaml" \
  --n-samples 32 \
  --seed 4901 \
  --method MC \
  --output-dir "run_output/reference_validation/RUN_2_MC32" \
  --diagnostic-factor 8 \
  --probe-grid-size 4 \
  --max-lag-cells 24
```

For scrambled Sobol validation, change `--method MC` to `--method RQMC` and use
a power-of-two sample count, such as 32, 64 or 128. The seed makes the design
reproducible. It does not turn a single scramble into independent observations.
For RQMC integration error, compare independent scrambles; do not use an iid
standard-error formula on points from one scramble. See the [SciPy QMC
documentation](https://docs.scipy.org/doc/scipy/reference/stats.qmc.html).

Each output directory must be empty or absent. Use a new directory for a new
design or scenario. The diagnostic factor must divide both field dimensions,
and the reduced diagnostic grid must retain at least three cells per axis.
Use `--diagnostic-factor 1` for a small field or when native spatial resolution
is affordable.

## Read the outputs

Open `validation_report.md` for the human-readable assessment and PNG figures.
`validation.json` retains the numeric diagnostics, source hashes, sampling
design, grid support, selected native probe locations and assumptions. The
reported support distinguishes three calculations:

| Calculation | Support and meaning |
| --- | --- |
| Native marginals and field/reference agreement | All native pixels; physical and log-space moments, ranges and quantiles, plus paired differences at matching locations. |
| Native short-range structure | Row/column gradient RMS and lag-one correlation on the original cell size, before block averaging. This exposes fine-grid changes that can disappear at coarse support. |
| Spatial texture | Matched geometric block means at the declared diagnostic factor; directional variograms and 2-D, radial and angular spectra. |
| Pointwise law and dependence | Repeated generated values at predetermined native cells, combining global coverage with a local stencil at declared correlation scales, compared with the actual retained KL law. |

`--probe-grid-size 4` selects up to four global rows and four global columns.
The workflow also selects the central native cell and local cells at ±0.5,
±1 and ±2 times the declared row/column correlation lengths, plus all four
diagonal sign combinations at 0.5 and 1 times those lengths. Offsets are rounded
to native cell offsets with a minimum of one cell, clipped to the field bounds
and deduplicated. The global grid provides spatial coverage; the local stencil
checks dependence at distances where correlation should be visible. A coarse
global grid alone can place every probe many correlation lengths apart.

Probe selection uses the serialized prior correlation lengths before examining
the generated fields. The report records the strategy, declared lengths and
exact native indices. This does not fit or identify correlation lengths. A
maximum global grid size of 16 adds at most 256 global probes and 21 local
stencil probes before deduplication; the exact total is reported. On a small
field, clipping can merge several requested offsets. A correlation scale below
one cell cannot be resolved by this native grid.

The PNGs are `validation_distribution.png`, `validation_correlation.png`,
`validation_spectra.png` and `validation_probe_law.png`. The field count and
sampling provenance remain in the report. To inspect individual permeability
images alongside the statistical review, use the [reference field preview
workflow](reference_field_previews.md); these are input fields, not temperature
predictions from the LGCNN.

Review the diagnostics together:

1. **Distributional agreement:** compare ranges, log-space moments and
   quantiles of each generated field with the reference; examine paired log
   differences and multiplicative ratios. A wider spatial histogram is not
   itself evidence of a broken generator, since the scenario intentionally
   adds uncertain residuals to a heterogeneous reference.
2. **Correlational agreement:** compare directional variograms, spectral shape
   and local texture, including native-grid gradient and lag-one descriptors.
   A checkerboard perturbation can cancel exactly inside each geometric block;
   the native descriptors reveal it even when the coarse spatial plots agree.
   A pixel permutation can preserve every marginal quantile
   while destroying spatial structure. These diagnostics address that failure.
   A diagnostic factor larger than one cannot resolve texture below the block
   scale; decrease it before concluding that fine-scale structure agrees.
3. **Generator consistency:** compare empirical log means, variances and
   covariance at native probes with the serialized Gaussian-coordinate map.
   This compares with the retained KL covariance, including truncation and any
   conditioning, rather than silently substituting the untruncated covariance
   kernel. With one field, repeated-draw covariance is inconclusive.

For median centering, the unconditional reference is the pointwise median;
the physical arithmetic expectation is larger where variance is positive. For
arithmetic-mean centering, the deterministic log correction makes the
unconditional physical expectation equal to the reference using the retained
pointwise variance. Conditioning can change both of these relationships: the
appropriate prediction is the conditional map, not an unchanged reference
center. The diagnostic records the distinction. The [SciPy lognormal
documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)
describes the Gaussian-to-lognormal transformation.

Empirical field-to-field spread describes this finite ensemble. It is not a
confidence interval for the population diagnostic, and the many correlated
pixels must not be counted as iid replicates for significance tests. Likewise,
quantile bounds for a lognormal law are high-probability bounds, not hard
support; fields outside them remain in the ensemble.

## Assessment and the decision to proceed

The default assessment remains `reference_review_required`. Numerical
similarity to one reference cannot establish scientific plausibility or LGCNN
validity by itself. Optional `--criteria path/to/criteria.yaml` thresholds
express explicitly chosen review criteria; they are not universal tolerances
or inferred probabilities. Even when all configured checks pass, the report
retains the review requirement. A failed check is visible and does not reject
or replace a generated field.

A criteria file is a flat YAML mapping containing any subset of the following
keys. Omitted checks remain unconfigured. Maximum limits must be finite and
non-negative; the minimum-correlation limit must be within `[-1, 1]`. Choose
limits with a scientific rationale and record that rationale with the study.

| Optional key | Compared quantity |
| --- | --- |
| `max_abs_log10_bias` | Largest absolute native-grid mean log ratio across fields. |
| `max_log10_rmse` | Largest native-grid root mean square log ratio across fields. |
| `min_log10_spatial_correlation` | Smallest native-grid paired log-permeability correlation with the reference. |
| `max_marginal_wasserstein_log10` | Spatial log-distribution distance on the declared diagnostic support. |
| `max_directional_variogram_relative_rmse` | Largest relative deviation among the reference-fidelity directional variograms. |
| `max_radial_power_relative_l2` | Relative radial spectrum deviation on the declared diagnostic support. |
| `max_angular_power_relative_l2` | Relative angular spectrum deviation on the declared diagnostic support. |

The criteria combine native and diagnostic support only where explicitly
labelled. Undefined metrics, such as correlation with a constant reference,
remain unassessed and cannot satisfy a corresponding configured check.

Before committing to the full scenario matrix, review a reduced design across
the proposed amplitudes, covariance families, correlation lengths and KL
truncations. Then compare the selected scenarios with **explicit, documented
LGCNN training support**, and verify downstream temperature-QoI convergence
across sample counts and KL dimensions. Retained covariance energy alone does
not establish temperature convergence. The diagnostics assess behavior under
the selected model-form assumptions; they do not establish a calibrated
geological law.
