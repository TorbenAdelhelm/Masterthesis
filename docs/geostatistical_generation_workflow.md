# Training-aware reference-centered permeability generation

This document defines the current workflow for realistic permeability-field UQ around the released real-`K` LGCNN.

## 1. Resolve the actual real-`K` training support

Do **not** assume that `order_data: 0` means `RUN_1`. Release25 maps `order_data` indices onto the numerically sorted prepared `RUN_*.pt` inventory. Resolve CNN1 and CNN3 separately from their original `command_line_arguments.yaml` plus either the original prepared `Inputs/` directory or the exact raw dataset inventory. If the evidence is unavailable, leave membership unresolved rather than guessing.

The result distinguishes training, validation, and test/scaling scenarios. Only proven training RUNs may populate `training_support` diagnostics.

## 2. Keep permeability and hydraulic conductivity units explicit

The LGCNN/PFLOTRAN fields are intrinsic permeability `k [m^2]`. Hydrogeological marginal assumptions may be specified for hydraulic conductivity `K_h [m/s]`. They are converted only through

`K_h = k rho g / mu`,  `k = K_h mu / (rho g)`.

`rho`, `mu`, and `g` must be recorded in every analysis artifact. The historical dataset-generation conversion remains provenance only and must not be mixed silently with this physical SI conversion.

## 3. Declare the marginal target and model-form assumptions

The default documented target is

`log10(K_h [m/s]) ~ Normal(-3, 0.5^2)`.

The interval `[1e-4, 5e-2] m/s` is a high-probability/validation interval, not hard support. A true lognormal law is positive and unbounded, so generated fields are never clipped, rejected, or resampled because individual pixels leave that interval.

This distinction is important: `log10_std = 0.5` is the **standard deviation in base-10 log space**, so its corresponding log10 variance is `0.25`. It is not a physical-space variance of `K_h`.

The reference-centered stochastic law remains

`log10 k_j(x, eta) = m_j(x) + R_theta(x, eta)`,  `eta ~ N(0, I)`.

`theta` contains sensitivity/model-form assumptions such as residual amplitude `sigma_R`, covariance family, row/column correlation lengths, and KL truncation. `sigma_R` is the residual spread around the reference field and must not be confused with the pooled marginal `log10_std` target of the generated hydraulic-conductivity values.

## 4. Analyze the real training fields before choosing sensitivity ranges

For each proven training RUN, report at least:

- log10 hydraulic-conductivity mean, standard deviation, variance and physical quantiles;
- fraction inside `[1e-4, 5e-2] m/s` plus lower/upper tail fractions;
- directional row/column variograms and inferred/practical correlation scales;
- radial/angular spectral summaries and patch/support descriptors.

The attached geostatistics teaching material motivates exactly this sequence: analyze observed values with histograms and variograms, generate equiprobable parameter-field realizations, compute derived quantities, then evaluate the results statistically. Our implementation follows that logic but uses realistic reference-centered residual fields because only a few real-K training simulations are available. No pixelwise PCA/KLT is fit across the different RUN cutouts.

The dimensionless `lambda_x/lambda_y` examples in the slides are illustrative only. They are **not** converted to metres unless an explicit normalization convention is defined. New scenario interfaces should therefore use `ell_row_m` / `ell_col_m` rather than ambiguous geographic `ell_x` / `ell_y` naming.

## 5. Generate without post-hoc filtering

For each chosen reference RUN and sensitivity parameter set, generate fields from the serialized Gaussian-coordinate law. The same iid standard-normal coordinates remain compatible with MC, scrambled-Sobol RQMC, and Hermite response PCE.

Validation is diagnostic, not a selection mechanism. Do not clip, reject, or redraw realizations to make their diagnostics look better; doing so would change the declared probability law.

## 6. Validate marginal and spatial behavior

Each scenario should record a `distribution_validation` report containing:

- generated log10 mean/std/variance;
- `K_h` quantiles, interval coverage and tail fractions;
- descriptive Cramér-von Mises normality discrepancy in log10 space;
- Wasserstein distance to the declared normal target and, where available, to training-RUN marginals;
- directional variogram mismatch;
- radial/angular spectral mismatch;
- patch/training-support descriptor deviations;
- reference-relative agreement metrics.

Pixels are spatially correlated. Therefore iid goodness-of-fit p-values are not used as evidence that a field is or is not lognormal; effect-size discrepancies, QQ behavior and ensemble consistency are more appropriate.

## 7. Run the frozen LGCNN and evaluate QoIs

After the input scenario passes basic *descriptive* plausibility checks, serialize the exact input model and propagate draws through the frozen real-K LGCNN. Keep reference scenario, model-form assumption `theta`, and random coordinate `eta` conceptually separate. Report MC/RQMC convergence and choose KL dimension using downstream temperature-QoI convergence rather than covariance-energy retention alone.
