# Release25 GRF Monte Carlo experiment

`release25_grf_mc` connects the KL log-Gaussian permeability model, optional
borehole conditioning, the frozen release25 LGCNN, and the existing field Monte
Carlo statistics/plotting path.

This is an input-law compatibility experiment. The GRF parameters are explicit
command-line inputs and are **not** treated as calibrated scientific defaults.
The pretrained LGCNN was trained on the release25 permeability regime, so a GRF
run should be interpreted cautiously until the generated permeability fields
have been compared with that regime.

## Unconditional GRF smoke test

The following parameter values are illustrative only:

```bash
python -m subsurface_uq.experiments.release25_grf_mc \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --mean-log10-k -9.64 \
  --std-log10-k 0.35 \
  --length-scale-y-m 250 \
  --length-scale-x-m 400 \
  --n-modes 20 \
  --n-samples 5 \
  --seed 2907 \
  --streamline-mode bounded \
  --output run_output/release25_grf_mc_5.npz \
  --plots-dir run_output/release25_grf_mc_5_temperature_plots \
  --permeability-plots-dir run_output/release25_grf_mc_5_permeability_plots \
  --permeability-preview-count 3
```

The installed alias is `subsurface-uq-release25-grf`.

Instead of `--n-modes`, the CLI can use `--energy-threshold`, for example
`--energy-threshold 0.95`. The two truncation options are mutually exclusive so
that the stochastic dimension is always an explicit experimental choice.

## Conditional GRF / kriging smoke test

Conditioning observations currently use integer permeability-grid cells and
physical permeability values in `m^2`. Repeat `--observation ROW COL K_M2` for
multiple boreholes:

```bash
python -m subsurface_uq.experiments.release25_grf_mc \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --mean-log10-k -9.64 \
  --std-log10-k 0.35 \
  --length-scale-y-m 250 \
  --length-scale-x-m 400 \
  --n-modes 20 \
  --observation 40 70 2.0e-10 \
  --observation 150 110 5.0e-10 \
  --observation-std-log10-k 0.05 \
  --n-samples 5 \
  --seed 2907 \
  --streamline-mode bounded \
  --output run_output/release25_conditional_grf_mc_5.npz \
  --plots-dir run_output/release25_conditional_grf_mc_5_temperature_plots \
  --permeability-plots-dir run_output/release25_conditional_grf_mc_5_permeability_plots
```

`--observation-std-log10-k 0` is the default and performs exact conditioning
relative to the retained KL model. Exact observations that the truncated KL
subspace cannot represent are rejected; increasing `--n-modes` or specifying a
measurement uncertainty is preferable to silently projecting incompatible data.

## Permeability input diagnostics

When `--permeability-plots-dir` is supplied, the experiment observes the exact
permeability batches that are subsequently propagated through the LGCNN. It does
not draw a second set of GRF samples. A small sampler wrapper updates streaming
statistics and yields each physical permeability batch unchanged.

The diagnostics are computed in the modeled Gaussian variable

\[
Y(x)=\log_{10}K(x),\qquad K(x)=10^{Y(x)}.
\]

The output directory contains

```text
log10k_mean.png
log10k_std.png
log10k_sample_001.png
log10k_sample_002.png
...
```

up to `min(--permeability-preview-count, --n-samples)` individual realizations.
The preview default is three. The individual preview maps share one color range
within a run so visual differences are not hidden by per-image rescaling.

The ensemble mean and standard deviation are accumulated online with the same
variance `ddof` used by the temperature Monte Carlo result. The implementation
retains only the running mean, running second moment, and the requested preview
fields; it does not build or save an `[N,H,W]` permeability ensemble.

For conditional runs, the conditioning cells are overlaid on the mean, standard
deviation and individual permeability maps. These markers represent the exact
integer model-grid cells supplied through `--observation`; no additional spatial
interpolation is introduced.

These figures are **input-distribution diagnostics**, not heat-plume UQ outputs.
They are intended to assess GRF spatial structure, magnitude, conditioning and
compatibility with the permeability regime of the pretrained LGCNN before the
downstream temperature uncertainty is interpreted scientifically.

## Outputs

The `.npz` archive and JSON metadata sidecar use the existing Monte Carlo result
format. They contain temperature mean, variance/range information and configured
Delta-T exceedance probabilities. `--plots-dir` reuses the existing Monte Carlo
visualization layer for mean/std/range, Delta-T, and exceedance maps.

Permeability plots are deliberately not embedded into the `.npz` archive and the
full permeability realizations are not stored. The metadata instead records
whether permeability diagnostics were enabled, the requested/saved preview
count, statistics sample count and plot directory.

The metadata also records the GRF map, KL truncation, conditioning information,
random seed, release25/checkpoint provenance, and streamline mode. No
permeability clipping is applied.

## Scope

The current command deliberately does not implement Hermite PCE, variogram
fitting, automatic GRF calibration, physical-coordinate-to-grid borehole
mapping, or a new experiment framework. Those steps should follow only after a
small frozen-LGCNN GRF compatibility run and its permeability-input diagnostics
have been inspected.
