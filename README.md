# Master Thesis — Subsurface UQ

This repository contains the thesis-specific forward uncertainty-quantification
(UQ) framework around deterministic groundwater heat-plume surrogate models.
The current scientific baseline focuses on uncertainty in the permeability
field while pressure and heat-pump locations are held fixed.

The primary [reference-centered lognormal scenario family](docs/reference_field_uncertainty.md)
uses spatial reference fields, explicit correlated input perturbations,
matched-support spectra/truncation diagnostics and Gaussian/Hermite response PCE.
Amplitude and covariance are explicit model-form/sensitivity assumptions, not calibrated truth.
RUNs remain separate reference scenarios. The measurement-kriging workflow remains
available as a legacy/comparison baseline with unchanged command defaults for reproducibility.

The implementation is designed around interchangeable components:

```text
PermeabilitySampler
        -> TemperatureSurrogate
        -> MonteCarloRunner
        -> TemperatureAccumulator(s)
             |- field statistics
             |- exceedance probabilities
             `- future QoIs
```

The original `Heat-Plume-Prediction` release25 implementation remains the
external deterministic reference, and `VampireMan` / the PFLOTRAN generation
repository remain external data-generation references. Scientific data, model
checkpoints and generated outputs are intentionally not committed.

## Documentation

The mathematical definition of the current pipeline is documented in
[`docs/methodology.md`](docs/methodology.md), including the LGCNN composition,
Monte Carlo estimators, streaming Welford statistics, exceedance probabilities,
the historical Perlin transformation, release25 normalization and output
alignment.

Runtime/data details are in
[`docs/release25_darus.md`](docs/release25_darus.md), the synthetic Perlin
baseline is documented in
[`docs/release25_perlin_uq.md`](docs/release25_perlin_uq.md), and calibration/
generation of realistic conditioned permeability fields is documented in
[`docs/realistic_permeability_generator.md`](docs/realistic_permeability_generator.md).

## Installation

Core package:

```bash
python -m pip install -e .
```

For the realistic DaRUS/geostatistical permeability workflow:

```bash
python -m pip install -e ".[geostat]"
```

The historical Perlin baseline is optional and uses the legacy C-extension
package `noise`:

```bash
python -m pip install -e ".[perlin]"
```

Keeping `noise` out of the core/geostat dependencies is intentional. On
Windows, especially with newer Python versions, `noise` may otherwise require
a locally configured MSVC + Windows SDK toolchain (for headers such as
`io.h`). The DaRUS calibration, KL/GRF, GSTools and radial-exponential
workflows do not require `noise`.

For the complete Linux CI test environment:

```bash
python -m pip install -e ".[test,perlin]"
python -m pytest
```

## Deterministic release25 integration

The real pretrained LGCNN execution path is

```text
physical p,k,i
      -> pretrained CNN1 / Step 1
      -> physical vx,vy
      -> original release25 Step-2 streamline solver
      -> [i,vx,vy,s,k,s_outer]
      -> pretrained CNN3 / Step 3
      -> physical temperature
```

A deterministic smoke run is:

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --device cpu \
  --output run_output/release25_single.npz
```

`--cnn2-dir` is retained as a backwards-compatible CLI name. It denotes the
**second CNN in the LGCNN, i.e. Step 3 / CNN3**.

For a one-sample deterministic diagnostic, the experiment automatically uses
`ddof=0`. For a real Monte Carlo ensemble the default is `ddof=1`; the code does
not report a one-sample sample variance as zero because that estimator is
undefined for `N=1, ddof=1`.

## Empirical Monte Carlo baseline

Existing permeability fields can be propagated while pressure and heat-pump
locations stay fixed to one prepared run:

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --permeability-run-ids RUN_1,RUN_2,RUN_4 \
  --device cpu \
  --output run_output/release25_empirical_mc.npz
```

Alternatively, `--ensemble-file` accepts an `[N,H,W]` permeability array/tensor
stored as `.npy`, `.npz`, `.pt`, or `.pth`.

## Historical Perlin forward UQ

`Release25PerlinPermeabilitySampler` reproduces the historical `perlin_v2`
spatial formula used for the released random-permeability data. For raw Perlin
field `P(x)`,

$$
U(x)=\frac{P(x)-\min P}{\max P-\min P},
$$

and the permeability is generated in base-10 logarithmic space,

$$
K(x)=10^{\log_{10}k_{\min}
+U(x)(\log_{10}k_{\max}-\log_{10}k_{\min})}.
$$

The released synthetic defaults are a `2560 x 2560` grid, `(18,18)` Perlin
frequency and permeability range
`1.0193679918450561e-11 ... 5.09683995922528e-09 m^2`.

A five-sample end-to-end smoke run is:

```bash
python -m subsurface_uq.experiments.release25_perlin \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --n-samples 5 \
  --seed 2907 \
  --device cpu \
  --output run_output/release25_perlin_mc_5.npz \
  --plots-dir run_output/release25_perlin_mc_5_plots
```

The exact initial groundwater temperature in the historical synthetic PFLOTRAN
setup is **10.6 °C** and the injection temperature is `15.6 °C`. Therefore the
Perlin baseline defines

$$
\Delta T = T-10.6\ ^\circ\mathrm C
$$

and by default accumulates the empirical spatial probabilities

$$
\widehat P_\tau(x)=\frac1N\sum_{m=1}^N
\mathbf 1[\Delta T^{(m)}(x)\ge\tau]
$$

for `tau=0.1 °C` and `1.0 °C`. Background and thresholds are configurable.
These statistics are accumulated online; `--store-all` is only needed if the
individual temperature realizations are required.

## Perlin PCE proof of concept

The coordinate-aware PCE experiment uses the same release25 asset paths as the
Perlin Monte Carlo command. It fits on a Latin-hypercube design and validates
against a separate expensive LGCNN ensemble:

```bash
python -m subsurface_uq.experiments.release25_perlin_pce \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --degree 4 \
  --n-train 60 \
  --n-validation 100 \
  --train-seed 2907 \
  --validation-seed 2908 \
  --device cpu \
  --output run_output/release25_perlin_pce_d4_n60.npz
```

The output contains the training and validation coordinates and QoIs, PCE
coefficients, held-out predictions, diagnostics and reproducibility metadata.
See [`docs/perlin_pce_proof_of_concept.md`](docs/perlin_pce_proof_of_concept.md)
for the mathematical definition and the important Perlin-offset smoothness
caveat.

## Realistic conditioned permeability fields

Real permeability ensembles can be used to calibrate log10-permeability
covariance parameters before drawing synthetic boreholes and conditional fields.
The workflow compares separable Matérn-3/2, separable exponential, and radial
anisotropic exponential covariance models. The first two reuse the explicit KL
coordinate path; the radial model uses GSTools conditioned random fields for
scalable Monte Carlo generation.

```bash
subsurface-uq-realistic-permeability calibrate \
  --fields data/real_k.npy \
  --cell-size-m 5 \
  --spatial-stride 4 \
  --output run_output/realistic_k/calibration.yaml
```

See `docs/realistic_permeability_generator.md` for the held-out synthetic
borehole experiment and the distinction between separable and radial
exponential covariance. The recommended pre-RQ2 inspection is the
`evaluate` subcommand, which performs leave-one-out calibration and compares
all three covariance candidates with the same exact full-covariance simple
kriging posterior (no KL truncation and no Monte Carlo sampling), then writes
variogram fits, fitted length scales, and held-out conditional reconstructions.

## Exact geospatial mapping for real LGCNN domains

Before evaluating measurement-conditioned stochastic fields on the release25
5 m grid, infer the projected Munich origin and raw-array orientation of the
chosen DaRUS real-permeability run:

```bash
python -m subsurface_uq.experiments.realistic_permeability georeference-domain \
  --raw-dataset data/dataset_100hp_giant_real_fixP0_0025 \
  --run RUN_1 \
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --reference-column K_P50 \
  --reference-z-mode top \
  --require-validated \
  --plots-dir run_output/realistic_k/georeference_plots \
  --output run_output/realistic_k/georeference_RUN_1.yaml
```

The matcher tests PFLOTRAN-array axis/flip conventions, searches the 100 m
reference model in log space, refines the crop at 5 m resolution and writes an
explicit cell-centre/domain-edge manifest. See
`docs/realistic_permeability_generator.md` for validation thresholds and
alternative vertical reference-surface modes.

## Multi-run georeference sweep

When a single Munich-reference match is weak, run the predefined nine-way
diagnostic sweep instead of lowering the thresholds:

```bash
python -m subsurface_uq.experiments.realistic_permeability georeference-sweep \
  --raw-dataset data/dataset_100hp_giant_real_fixP0_0025 \
  --runs RUN_1 RUN_2 RUN_3 \
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --output-dir run_output/realistic_k/georeference_sweep
```

It evaluates `K_P10/K_P50/K_P90 x top/bottom/log_geomean`, writes one
comparison CSV, and explicitly reports whether any reference representation is
individually valid for every run and cross-run consistent in orientation and
unit shift. See `docs/realistic_permeability_generator.md` for the decision
rule.

## Legacy measurement-conditioned comparison workflow

This comparison path keeps its original command defaults for reproducibility.
The primary input architecture is the reference-centered scenario family linked
above; fitting reference texture does not identify its residual uncertainty law.

The measurement permeability workflow distinguishes the **three complete
real-permeability fields used to train the frozen LGCNN** from the much larger
set of overlapping patches that the CNN actually saw during optimization.
The real-permeability **training data** are published as
[DaRUS-5065](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-5065).
The separate DaRUS-5082 dataset contains the pretrained real-`k` LGCNN model
artifacts. DaRUS-5065 contains the raw 4+1 PFLOTRAN simulations; the publication
documents that three of the four standard 12.8 km fields were used for training
and one for validation.
For both real-`k` LGCNN steps the selected cutout hyperparameters are
`box_length=1280` cells and `skip_per_dir=8` cells.

The three complete training fields are **not** a pooled geological prior.
Their overlapping cutouts are also not treated as tens of thousands of
independent geological realizations. In this legacy workflow, the real Munich
measurements define the geostatistical mean/covariance model; DaRUS-5065 is used
to test whether those generated inputs remain inside the spatial/statistical
support seen by the frozen LGCNN.

Calibrate the DaRUS-5065 training-field support descriptors on exactly the three
training runs:

```bash
python -m subsurface_uq.experiments.realistic_permeability calibrate \
  --fields data/dataset_100hp_giant_real_fixP0_0025 \
  --runs <TRAIN_RUN_A> <TRAIN_RUN_B> <TRAIN_RUN_C> \
  --cell-size-m 5 \
  --spatial-stride 4 \
  --output run_output/realistic_k/training_calibration.yaml
```

The exact three `RUN_*` names must come from the pretrained-model metadata /
training command. They are intentionally not inferred from directory order.

First run `measurement-evaluate` so the real Munich observations define
directional variograms, structured variance, nugget and covariance length scales.
Then generate the conditional field ensemble:

```bash
python -m subsurface_uq.experiments.realistic_permeability new-domain-generate \
  --measurements "C:/path/to/kf_werte_190201.xlsx" \
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --training-fields data/dataset_100hp_giant_real_fixP0_0025 \
  --training-runs <TRAIN_RUN_A> <TRAIN_RUN_B> <TRAIN_RUN_C> \
  --training-calibration run_output/realistic_k/training_calibration.yaml \
  --measurement-calibration run_output/realistic_k/measurements/measurement_calibration.yaml \
  --training-info-yaml <PATH_TO_DARUS_5082_INFO_YAML> \
  --input-law measurement-kriging \
  --energy-threshold 0.95 \
  --n-samples 8 \
  --output-dir run_output/realistic_k/new_domain
```

When `--model` is omitted, the best Matérn-3/2 or exponential candidate by
measurement spatial-block-CV RMSE is used. A CLI model choice is only needed for
an explicit sensitivity comparison.

Patch support is evaluated at the published `1280 x 1280` / skip-8 geometry.
For tractability, a deterministic subset of that highly correlated patch
population is used to compute log-permeability marginal, gradient and local
correlation descriptors for both training and generated fields. Full-field
statistics remain secondary diagnostics.

This comparison law uses measurement-derived simple kriging represented in a
finite KL basis. The measurement workflow estimates the mean, structured
variance, nugget and directional correlation lengths in `log10(K_h)`; conversion
to intrinsic permeability adds a constant log shift and therefore leaves the
covariance structure unchanged. The legacy comparison map is

```text
real Munich measurements
  -> measurement variogram + spatial block CV
  -> log10(k) Gaussian/KL prior
  -> continuous-point conditioning on the same measurements
  -> eta ~ N(0,I)
  -> conditional log10(k) field
  -> K
```

Thus MC, randomized QMC and Hermite PCE still use explicit iid Gaussian
coordinates. DaRUS-5065 fields are used as **surrogate-support/fidelity
references**, not to override the site-specific measurement covariance. The
previous `normal-score-copula` and `legacy-lognormal` laws remain available as
explicit training-derived sensitivity alternatives.

The output `stochastic_input_model.yaml` remains the reusable stochastic-law
artifact. It now stores both the primary patch-support reference and the
secondary full-field reference, together with the optional release25
normalization metadata. Point RQ1 at it with `grf.input_model`; do not duplicate
the GRF parameters manually.

When `--save-samples` is enabled (the default), every realization keeps a
lossless float32 `.npy` file and two visual products. The simple field PNG uses a shared robust `log10(K [m^2])` scale spanning the
envelope of the training q01--q99 and measurement q01--q99 ranges.
Each `samples/comparisons/sample_XXXX_measurement_overlay.png` overlays the
original measured intrinsic-permeability values directly on the generated
raster using **exactly the same log10(k) color scale**. The companion
`sample_XXXX_comparison.png` shows the generated realization, the
measurement-derived kriging reference, the closest DaRUS-5065 training field and
the generated-minus-reference residual. The generator also reports conditioning
errors at the measurements plus marginal Wasserstein/quantile errors and
directional variogram mismatch against the training ensemble.

## Real Munich measurement calibration

The preferred realistic-permeability calibration path can now use the actual
Munich hydraulic-conductivity measurements together with the active XY footprint
of the 3-D reference field. The workflow keeps the original continuous
measurement coordinates, fits directional irregular-point variograms in
`log10(K_h)` with a fitted nugget and pair-count weighting, compares
covariance families with spatial block cross-validation, runs an upper-tail
robustness check, and conditions the selected model on all accepted real
observations.

```bash
python -m subsurface_uq.experiments.realistic_permeability measurement-evaluate \
  --measurements "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf_werte_190201.xlsx" \
  --reference-grid "C:/Users/Torbe/Desktop/MT/Daten/Messdaten/kf-Werte München/kf-Werte-3D Modell/3D_K_Field_Munich_K_P10_P50_P90.csv" \
  --output-dir run_output/realistic_k/measurements
```

See `docs/realistic_permeability_generator.md` for the exact filtering,
spatial-CV metrics, hydraulic-conductivity/intrinsic-permeability distinction,
and the follow-up `measurement-condition` command.

## Monte Carlo statistics and extensible QoIs

For propagated temperature fields `T^(m)(x)`, the core computes

$$
\hat\mu_T(x)=\frac1N\sum_{m=1}^N T^{(m)}(x)
$$

and

$$
\hat\sigma_T^2(x)=\frac1{N-d}\sum_{m=1}^N
(T^{(m)}(x)-\hat\mu_T(x))^2,
$$

where `d` is `ddof`. Mean, variance, standard deviation and extrema are updated
with a streaming batch-merge form of Welford's algorithm.

Additional quantities of interest use the `TemperatureAccumulator` interface.
An accumulator receives temperature batches through `update(...)` and returns
its final result through `finalize()`. This keeps future monitoring-point,
plume-length, plume-area or other QoIs out of the propagation core.

## Monte Carlo/UQ visualization

A Perlin run with `--plots-dir` creates physical-space maps of:

- mean temperature;
- standard deviation;
- sample range `max-min`;
- mean temperature with standard-deviation contours;
- mean `Delta T`;
- one exceedance-probability map per threshold.

Existing MC archives can be plotted without rerunning the surrogate:

```bash
python -m subsurface_uq.visualization.cli \
  --input run_output/release25_perlin_mc_5.npz \
  --output-dir run_output/release25_perlin_mc_5_plots
```

For older archives that do not store a background temperature, use
`--background-temperature 10.6` for the historical synthetic dataset.

## Deterministic diagnostics and temperature validation

The empirical release25 runner can additionally create velocity, streamline and
temperature diagnostic plots with `--plots-dir`.

A physical prediction can be compared against a prepared reference-temperature
label with:

```bash
python -m subsurface_uq.validation.cli \
  --prediction run_output/release25_single.npz \
  --reference-dir data/prepared_pki_temperature \
  --run-id RUN_1 \
  --output run_output/release25_RUN_1_validation.npz \
  --plots-dir run_output/release25_RUN_1_plots
```

The validation layer reverse-normalizes the reference, center-crops the larger
reference to the valid-convolution prediction shape when requested, and reports
MAE, MSE, RMSE, maximum absolute error, bias and spatial error diagnostics.
Optional overlays regenerate the release25 central streamline feature and
heat-pump locations.

## Reproducibility

New Monte Carlo results use a versioned NPZ schema and write the same run
metadata to a neighboring `.metadata.json` sidecar. Release25 experiment runs
record, where available, the realized sample count, variance `ddof`, sampler
parameters, fixed run, Git SHAs, SHA-256 hashes of the two model checkpoints,
Python/platform and dependency versions, release25 settings, and historical
Perlin generator provenance.

The historical Perlin formula is tested against an independent transcription of
the original generator. The historical generator itself did not effectively use
the DaRUS `seed_id` because its NumPy seeding line was commented out; the thesis
sampler makes the seed explicit for reproducible *new* UQ experiments and does
not claim bitwise recovery of the original training fields.

## Current validation scope

CI uses lightweight fixtures and tests sampler behavior, batch-invariant Monte
Carlo statistics, custom accumulator extensibility, historical Perlin formula,
result metadata, UQ plotting, release25 normalization/model reconstruction,
three-stage adapter execution and temperature validation utilities.

The actual released DaRUS models/data have also been exercised locally with a
real deterministic RUN_1 prediction and prepared reference comparison. This is
not yet the stronger claim of numerical equivalence to a separately executed
original release25 runtime; that remains a distinct validation step.

## Next phases

The cleaned baseline supports the next scientific layers without changing the
propagation core. A borehole-conditioned geostatistical permeability workflow is
now available, including real-field covariance calibration and Matérn-3/2,
separable exponential and radial anisotropic exponential candidates. The next
scientific step for the primary reference family is to compare explicitly assumed
residual laws and KL representations using matched training support and downstream
temperature QoIs. Reference-texture fits do not identify residual uncertainty.
MC, scrambled Sobol RQMC and Hermite response PCE share the Gaussian-coordinate law.
