# RQ1 evaluation experiment

Research question:

> How much permeability input uncertainty reaches the predicted temperature field?

The implementation keeps the release25 LGCNN frozen and holds the prepared
pressure/material/heat-pump scenario fixed. It therefore measures permeability
input uncertainty only.

## Experimental variants

The RQ1 command executes the finalized four-part design:

- **A — Perlin + MC:** in-generator-family release25 baseline using the existing
  two-coordinate Perlin map with iid uniform offset coordinates. This defines a
  reproducible continuous reference law from the historical `perlin_v2`
  generator family; it does not claim to reproduce the dependence structure of
  the finite historical training fields, which shared one base offset and used
  successive x-shifts.
- **B — unconditional log-GRF/KL + MC:** effect of replacing the Perlin input
  law by the explicit Gaussian random-field prior.
- **C — conditional log-GRF/KL + MC:** primary input-UQ experiment.
- **D — conditional log-GRF/KL + randomized QMC:** scrambled Sobol comparison
  at equal frozen-LGCNN evaluation budgets.

The default design encoded in the example configuration uses main checkpoints
N = {32, 64, 128, 256, 512, 1024} and repeated MC/RQMC comparison budgets
N = {32, 64, 128, 256, 512}.

The largest conditional iid-MC run is an **empirical reference**, not ground
truth. The configuration accepts larger powers of two (for example 2048 or
4096) if the reference-convergence check later shows that this is needed.

RQMC uses independently scrambled Sobol designs. A point u in (0,1)^d is
transformed component-wise by the standard-normal inverse CDF, so the
conditional KL map still receives independent standard-Gaussian coordinates.
The same explicit repetition-seed list is used to pair the MC and RQMC
repetitions.

## Temperature uncertainty metrics

For the final main runs the command writes exact empirical fields

mu_T(x), sigma_T(x), q05(x), q50(x), q95(x),
and W90(x) = q95(x) - q05(x).

Mean and variance are accumulated online. Exact empirical field quantiles are
computed from a temporary disk-backed float32 temperature ensemble, spatially
chunked so the full [N,H,W] array is not held in RAM.

Global summaries are

U_RMS = sqrt(mean_x sigma_T(x)^2)

and

U95 = q95({sigma_T(x)}),

plus mean and maximum spatial standard deviation.

Field convergence against the corresponding largest empirical MC run is

e_mu(N) = sqrt(mean_x (mu_N(x)-mu_ref(x))^2)

and

e_sigma(N) = sqrt(mean_x (sigma_N(x)-sigma_ref(x))^2).

These are absolute RMS errors in degrees Celsius.

## Scalar QoIs

Receptor locations are configured explicitly as temperature-output grid cells;
the implementation does not choose them automatically. For Q_r = T(x_r), it
reports mean, unbiased standard deviation and empirical q05/q50/q95.

If `qoi.mean_anomaly_roi` is configured, the continuous mean-anomaly QoI is
evaluated over that fixed ROI Omega_T:

Q_mean = mean_{x in Omega_T} [T(x)-T_bg].

The same statistics are reported for this scalar QoI. Setting
`qoi.mean_anomaly_roi: null` disables this optional scalar QoI without
affecting field or receptor metrics.

Thresholded plume/risk quantities are deliberately not part of RQ1; they belong
to RQ2.

## Input diagnostics

Every main variant observes the exact permeability batches that are passed into
the frozen LGCNN. Diagnostics are evaluated in Y = log10(K).

The output records/plots spatial mean and standard deviation, global physical
minimum/maximum, and the fraction outside the release25 training range

1.0193679918450561e-11 <= K <= 5.09683995922528e-09 m^2.

For conditional fields it also records

e_b^(n) = Y^(n)(x_b) - Y_obs(x_b)

with per-sample and aggregate residual diagnostics. No universal conditioning
or compatibility threshold is hard-coded. If a justified
conditioning_tolerance_log10 value is supplied, the run fails when it is
violated.

## Repeated MC versus randomized QMC

The conditional-GRF empirical MC result at the largest main budget is used as
the common reference. For each independent repetition and equal budget the code
records errors for:

- temperature field mean and standard deviation;
- global U_RMS, U95, mean SD and max SD;
- receptor mean/SD/q05/q50/q95;
- mean-anomaly mean/SD/q05/q50/q95.

Across repetitions it computes estimator RMSE and

G(N) = RMSE_MC(N) / RMSE_RQMC(N).

G(N) > 1 means randomized QMC produced lower error at the same number of LGCNN
evaluations.

## Command

Start from the example and replace the illustrative GRF and QoI locations by the
final thesis values:

    cp configs/rq1.example.yaml configs/rq1.yaml

Run with

    python -m subsurface_uq.experiments.release25_rq1 --config configs/rq1.yaml

or, after package installation,

    subsurface-uq-release25-rq1 --config configs/rq1.yaml

The command loads the release25 models once and reuses the existing sampler,
conditional-KL, bounded-streamline, surrogate and Monte Carlo abstractions.

## Result layout

The output root contains the required machine-readable artifacts:

    config.yaml
    metadata.json
    input_diagnostics.json
    global_metrics.json
    convergence.csv
    mc_rqmc_comparison.csv
    receptor_metrics.csv
    qoi_samples.csv

Each main variant has its own directory. Its fields directory contains

    temperature_mean.npy
    temperature_std.npy
    temperature_q05.npy
    temperature_q50.npy
    temperature_q95.npy
    temperature_width90.npy

and its figures directory contains input log-permeability diagnostics,
temperature mean/std/width90 maps and scalar-QoI distributions. Root-level
figures contain convergence and MC-vs-RQMC RMSE comparisons.

The temporary disk-backed temperature ensemble is removed after derived field
products have been written unless storage.retain_scratch is true.

## Scope exclusions

RQ1 does **not** add PCE, sensitivity analysis, threshold/risk or plume metrics,
or LGCNN-vs-PFLOTRAN/model-discrepancy uncertainty. Those are assigned to later
research questions in the finalized evaluation design.
