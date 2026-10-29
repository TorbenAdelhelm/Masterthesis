# Training-aware reference-centered permeability generation

This document defines the current workflow for realistic permeability-field UQ around the released real-`K` LGCNN.

Use the same Python environment for installation and execution. From the repository
root in Git Bash, install the geostatistical dependencies, including `h5py` for
the HDF5 RUN files:

```bash
python -m pip install -e ".[geostat]"
```

The optional historical Perlin dependency is unnecessary for this workflow.

## 1. Resolve the actual real-`K` training support

Do **not** assume that `order_data: 0` means `RUN_1`. Release25 maps `order_data` indices onto the numerically sorted prepared `RUN_*.pt` inventory. Resolve CNN1 and CNN3 separately from their original `command_line_arguments.yaml` plus either the original prepared `Inputs/` directory or the exact raw dataset inventory. If the evidence is unavailable, leave membership unresolved rather than guessing.

The support profiler therefore has two legitimate modes:

- with CNN1/CNN3 training metadata it emits proven `training_support` and distinguishes training/validation/test RUNs;
- without that evidence it emits `realistic_dataset_support`, which is useful for distribution/spatial diagnostics but is **not** called training support.

Git Bash, unresolved-membership mode:

```bash
python -m subsurface_uq.experiments.real_k_training_support \
  --dataset-root "C:/path/to/dataset_100hp_giant_real_fixP0_0025" \
  --permeability-convention historical-training \
  --diagnostic-factor 8 \
  --output-dir "run_output/real_k_support"
```

When the original metadata are available, add
`--cnn1-command-line`, `--cnn3-command-line` and optionally the two prepared-Inputs paths. Only then may the resulting common RUN set be called proven training support.

`--runs` selects the fields actually profiled. The report distinguishes the proven
common training membership from its profiled subset; an excluded training set
does not yield an empty or fabricated training summary. Spatial diagnostics use
geometric block means with `--diagnostic-factor 8` by default (40 m cells for
5 m inputs), and record that support explicitly. Marginal statistics retain
the native grid. Set the factor to 1 for native-grid variograms and spectra.

## 2. Keep permeability and hydraulic conductivity units explicit

The LGCNN/PFLOTRAN fields are intrinsic permeability `k [m^2]`. Hydrogeological distribution assumptions are expressed as hydraulic conductivity `K_h [m/s]`. Two conversion conventions exist and are never mixed silently:

- `historical-training`: the realistic data generator used `k = K_h / 7.5e6`; this is the correct inverse mapping when validating the released real-K training fields against their original hydraulic-conductivity distribution;
- `physical`: `K_h = k rho g / mu`, with `rho`, `mu` and `g` stored explicitly.

Generated arrays passed to the pretrained real-K LGCNN always remain intrinsic permeability in the declared LGCNN convention. Hydraulic conductivity is a modelling/validation view of those same arrays, not a replacement LGCNN input.

## 3. Declare the marginal target and model-form assumptions

Global distribution targets are **opt-in diagnostic assumptions**. By default,
the generator, matrix runner and support profiler report descriptive statistics
without assessing a fixed global distribution. A spatially varying reference
has pointwise lognormal marginals, but its pooled histogram need not be lognormal.
The reference-centered law does not require that histogram to match a fixed normal
law in log space. Scenario anchor mean/variance or quantile bounds, when supplied,
remain separate consistency-checked assumptions and are unchanged by these options.

The supplied geostatistical teaching example, available as an optional target, states

`log10(K_h [m/s]) ~ Normal(-3, 0.5^2)`.

The interval `[1e-4, 5e-2] m/s` is treated as a high-probability/validation interval, not hard support. A true lognormal law is positive and unbounded, so generated fields are never clipped, rejected, or resampled because individual pixels leave that interval.

The value `0.5` in the teaching material is `sigma_log10`, i.e. the **standard deviation in base-10 log space**. The corresponding log10 variance is `0.25`. If a variance of `0.5` is intended in a separate experiment, specify `log10_variance: 0.5` explicitly; the configuration forbids supplying both std and variance.

On the CLI, `--distribution-target` enables these documented defaults. Supplying
any explicit `--target-*` option or `--minimum-interval-fraction` also opts in;
unspecified target parameters then use the documented defaults. For example,
`--target-log10-variance 0.5` explicitly requests a different spread. In a matrix,
omit `distribution_target` or set it to `null` to disable comparisons; an explicit
mapping (including `{}` for defaults) enables them. To reproduce earlier reports
that implicitly used the documented target, now select it explicitly. The generated
Gaussian coordinates and permeability fields do not change.

The reference-centered stochastic law remains

`log10 k_j(x, eta) = m_j(x) + R_theta(x, eta)`,  `eta ~ N(0, I)`.

`theta` contains sensitivity/model-form assumptions such as residual amplitude `sigma_R`, covariance family, row/column correlation lengths, and KL truncation. `sigma_R` is the residual spread around the reference field and must **not** be confused with the pooled marginal `log10_std` of the resulting heterogeneous field. Consequently the generator reports both quantities separately rather than claiming that `sigma_R=0.5` forces the full-field histogram to have standard deviation `0.5`.

## 4. Analyze the real fields before choosing sensitivity ranges

For every profiled RUN, report at least:

- log10 hydraulic-conductivity mean, standard deviation, variance and physical quantiles;
- with an explicit target: interval/tail fractions and log-space skewness, excess kurtosis, QQ error, Cramér-von Mises and KS effect-size statistics;
- directional row/column variograms and inferred/practical correlation scales;
- radial/angular spectral summaries and patch/support descriptors where available.

The geostatistics teaching material motivates the same overall sequence: analyze observed values with histograms and variograms, generate equiprobable parameter-field realizations, calculate derived quantities, and statistically evaluate the results. The implementation follows that logic while using realistic reference-centered residual fields because only a few real-K simulation fields exist. No pixelwise PCA/KLT is fit across differently located/rotated RUN cutouts.

The dimensionless `lambda_x/lambda_y` examples in the slides are illustrative only. They are **not** converted to metres unless an explicit normalization convention is known. New interfaces therefore use `ell_row_m` / `ell_col_m`. Legacy `ell_y` / `ell_x` remain loadable and map to row/column respectively.

## 5. Generate without post-hoc filtering

A direct Git Bash example is:

```bash
python -m subsurface_uq.experiments.reference_field_permeability \
  --dataset-root "C:/path/to/dataset_100hp_giant_real_fixP0_0025" \
  --reference-run RUN_2 \
  --permeability-convention historical-training \
  --residual-std-log10-k 0.05 \
  --length-scale-row-m 500 \
  --length-scale-col-m 800 \
  --covariance-model matern32 \
  --energy-thresholds 0.95 0.99 \
  --n-samples 32 \
  --output-dir "run_output/reference_generation/RUN_2"
```

For a scenario study, edit `configs/reference_scenarios.example.yaml` and run:

```bash
python -m subsurface_uq.experiments.reference_scenarios \
  --config "configs/reference_scenarios.example.yaml" \
  --output-dir "run_output/reference_scenarios"
```

The same iid standard-normal coordinates remain compatible with MC, scrambled-Sobol RQMC, and Hermite response PCE.

Validation is diagnostic, not a selection mechanism. Do not clip, reject, or redraw individual realizations to make their diagnostics look better; doing so would change the declared probability law. Instead, change the **scenario assumptions** (`sigma_R`, covariance family, row/column lengths, KL representation) and rerun the scenario if the ensemble is implausible.

## 6. Validate marginal and spatial behavior

Every generated scenario now writes `distribution_validation.json`. It contains:

- exact streaming pooled log10 mean/std/variance;
- per-field marginal diagnostics;
- `K_h` quantiles;
- with an explicit target: moment errors, interval/tail fractions, log-space shape and normality discrepancies;
- the corresponding marginal diagnostics of the unperturbed reference field;
- residual-law `sigma_R` reported separately from full-field `sigma_log10`;
- the explicit `k <-> K_h` conversion convention;
- reference-relative variogram/spectral diagnostics plus optional training-patch support diagnostics;
- `sample_filtering: null` / `no_filtering: true` as an auditable guarantee that diagnostics did not alter the sampled law.

Without a target, target errors, coverage and normality comparisons are `null` and
the assessment is `not_assessed`. Reference-relative variograms, spectra and
available training descriptors remain active.

Optional thresholds in `distribution_validation` require an explicit target and
yield only a `heuristically_compatible` / `heuristic_deviation` flag. Supply any
subset of the three limits; omitted limits/checks are `null`. Nonfinite/negative
limits fail before outputs are created. These flags are assumptions for screening
scenario families, not statistical proofs and never reject samples. The direct
generator exposes the same limits as `--max-abs-log10-mean-error`,
`--max-abs-log10-std-error` and `--max-qq-rmse-standardized`.

Pooled quantiles and shape discrepancies use all pixels up to 200,000 values,
then a reproducible uniform pixel sample with replacement (diagnostic seed 0).
The report records the method, capacity, seed and population size. New fields
receive weight according to pixel count, without preferentially weighting later
fields. These bounded-sample statistics are approximate; pooled moments and
interval counts use every pixel. This diagnostic pixel pool does not assign
geological scenario probabilities or combine RUNs into a posterior. Results
remain per scenario unless explicit scenario-weight assumptions are supplied.

Pixels are spatially correlated. Therefore iid goodness-of-fit p-values are not used as evidence that a field is or is not lognormal; effect-size discrepancies, QQ behavior, interval coverage and ensemble consistency are the relevant diagnostics.

## 7. Run the frozen LGCNN and evaluate QoIs

Before model propagation, use the dedicated
[reference-field validation command](reference_field_diagnostics.md) to turn
existing generated arrays (or an explicit fresh validation design) into PNGs and
a review report. It compares the full-field spatial distribution and directional
structure to the fixed reference, adds empirical realization envelopes, and
checks predetermined native-cell ensemble moments/correlations against the
actual retained KL law. These are distinct supports: geometric block averages
are used for tractable spatial plots, native cells for pointwise law checks.

Assess the input scenario relative to its reference, available real-K support and
any explicitly selected global target, then serialize the exact input model and
propagate draws through the frozen real-K LGCNN. Keep reference scenario,
model-form assumption `theta`, and random coordinate `eta` conceptually separate.
Report MC/RQMC convergence and choose KL dimension using downstream temperature-QoI
convergence rather than covariance-energy retention alone.

The workflow does not claim that the declared target or residual covariance has been statistically identified. Its purpose is to make every assumption explicit, generate realistic perturbations around actual real-K reference fields, and provide enough marginal/spatial diagnostics to reject an implausible **scenario definition before expensive UQ propagation** without modifying individual Monte Carlo draws.
