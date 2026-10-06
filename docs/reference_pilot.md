# Reduced reference-scenario study

This pilot uses the primary scenario architecture without changing the law. The
three RUNs remain separate nominal references, and scenario weights are absent.
It exercises the real-K pretrained models on the native 2560×2560, 5 m grid.

## Frozen design

The main design contains 12 scenarios: RUN_1/RUN_2/RUN_3 × sigma_R 0.05/0.10
log10 units × separable Matérn-3/2/exponential, with ell_x=800 m, ell_y=500 m and
95% covariance energy. Two additional RUN_2, sigma_R=0.05, Matérn scenarios use
99% and 99.9% energy for paired temperature-QoI representation sensitivity.
These amplitudes, kernels and lengths are assumptions. No physical moment or
quantile bounds were supplied for this pilot, and none are inferred from the
reference histograms.

Length labels follow the map's array convention: ell_y=500 m along rows and
ell_x=800 m along columns. In the inspected raw RUN_2 mesh, these correspond to
local physical x and y respectively. They are not fitted geographic principal
directions or an anisotropy rotation inferred from the three RUNs.

Permeability diagnostics use eight iid Gaussian samples per scenario, seed 4901.
Factor-8 geometric block means are used only for the 40 m variogram/spectral
diagnostics. Model inputs retain the complete native grid. Full generated
ensembles are not retained; hashed reference artifacts reproduce the generators.

Temperature propagation uses eight scrambled-Sobol Gaussian samples, seed 5901.
The three KL truncations share coordinates, and the largest dimension is a
comparison reference, not truth. Other scenario results stay separate. Eight
samples and one scramble provide exploratory sensitivity estimates, not reliable
tail probabilities or integration-error estimates. Prefix budgets 4 and 8 can
indicate instability but do not establish convergence.

The paired anchor uses the first coordinates of the 5049-dimensional design.
Other primary scenarios construct their own dimension-specific designs. Sharing
a seed therefore does not make every RUN/amplitude/kernel contrast paired;
those contrasts include integration variability.

## Actual fixed setup and QoIs

Pressure, material IDs and the 100 source locations are extracted from the
initial-time RUN_2 PFLOTRAN output and held fixed while permeability varies.
RUN_1 permeability is recovered from its original input file because the local
copy lacks simulation output. All permeability values retain the historical
training convention, with no extra physical conversion.

Preparation uses the actual upstream `pki` extraction/transforms and checks
normalization against the upstream implementation. The persisted prepared input
is recovered and compared with the original physical channels. SHA-256 manifests
record the raw data, reference arrays, models, metadata and preparation code.

The real-K chain maps input 2560×2560 to velocity 1796×1796 and temperature
1028×1028. The temperature crop begins at native cell 766 in each direction.
The chosen output receptors `[312,532]`, `[372,532]`, `[301,489]` lie 100/400 m
downstream of two actual interior RUN_2 sources. Raw row increases physical x.
The mean-temperature-anomaly ROI is `[296:596,456:636]`, with background 10.6 °C.
These are exploratory plume QoIs, chosen before stochastic predictions. They
are not an independent surrogate-validation experiment.

Published `skip_per_dir=8` describes the stride of overlapping 1280-cell patch
starts, not an eightfold reduction of the model grid. A separate diagnostic
feature stride of 8 may be used to bound descriptor cost. A 1280-cell input cannot
pass both valid-convolution networks in this adapter; coarsening it to 40 m would
also conflict with the upstream streamline conversion, which assumes 5 m.

Checkpoint metadata identifies prepared-list indices, but the original prepared
inventory has not been recovered. Actual training RUN membership therefore stays
unverified. Comparisons to the supplied reference patch envelopes and to stored
model-normalization ranges are descriptive partial checks. They cannot certify
training compatibility. No generated sample is clipped, rejected or replaced.

## Reproduce in Git Bash

Run from the repository root with the scientific dependencies installed. Set the
three external asset locations, keeping extracted standard model folders intact:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
DATASET_ROOT='C:/path/to/dataset_100hp_giant_real_fixP0_0025'
RELEASE25_ROOT='C:/path/to/Heat-Plume-Prediction'
CNN1_ROOT='C:/path/to/LGCNN_step1_realK/standard_model_folder'

python examples/reference_pilot/prepare_data.py \
  --dataset-root "$DATASET_ROOT" \
  --cnn1-dir "$CNN1_ROOT" \
  --release25-repo "$RELEASE25_ROOT" \
  --output-dir data/reference_pilot

python -m subsurface_uq.experiments.reference_scenarios \
  --config configs/reference_scenarios.pilot.yaml \
  --output-dir run_output/reference_pilot/input_scenarios

python -m subsurface_uq.experiments.reference_scenarios \
  --config configs/reference_scenarios.pilot_kl.yaml \
  --output-dir run_output/reference_pilot/kl_scenarios
```

Copy `configs/rq1.reference.pilot.example.yaml` to
`run_output/reference_pilot/rq1.local.yaml` and set the actual upstream and real-K
checkpoint paths. The two model directories refer to Step 1 and Step 3;
`cnn2_dir` is the existing adapter's name for Step 3. Then run:

```bash
python examples/reference_pilot/run_temperature_pilot.py \
  --matrix-manifest run_output/reference_pilot/input_scenarios/manifest.json \
  --kl-manifest run_output/reference_pilot/kl_scenarios/manifest.json \
  --rq1-config run_output/reference_pilot/rq1.local.yaml \
  --reference-arrays data/reference_pilot/RUN_1.reference.npy \
    data/reference_pilot/RUN_2.reference.npy data/reference_pilot/RUN_3.reference.npy \
  --n-samples 8 --seed 5901 --method RQMC \
  --output-dir run_output/reference_pilot/temperature
```

Preparation refuses to replace existing outputs. For a different data/grid/law
instance or temperature-run context, select a fresh directory and update the
reference/config paths. The temperature cache allows the same interrupted
context to resume. Scientific arrays and caches remain ignored local assets.

## Decision before the 96-scenario matrix

To store and inspect a subset of the exact permeability inputs used in this
study, see [reference/realization PNG comparisons](reference_field_previews.md).
The exporter verifies each regenerated input against its original model-query
hash and can retain selected NPY arrays without saving the full ensemble.

Inspect scenario-wise input descriptors, normalization-range excursions and
temperature changes, then examine paired KL changes. Resolve the original
training inventory and run additional scrambles/budgets before committing to a
large UQ design. The pilot does not select a calibrated covariance or establish
that 95% energy is sufficient for the temperature QoIs.

On the pilot's full grid and exponential lengths, preflight dimensions are 9316,
71290 and 413094 for 0.95/0.99/0.999 energy. The installed SciPy Sobol maximum
dimension is 21201. The two larger exponential choices therefore cannot run
through the existing Sobol path. Their Gaussian MC law remains valid. Even the
9316-dimensional degree-2 dense Hermite basis has 43,407,903 terms, so this pilot
does not attempt response PCE. Fixed 20 modes retain only 14.46% exponential
energy here; reducing dimension also changes the represented residual variance.
Selecting QoI-adequate truncations or another integration/approximation strategy
is a prerequisite for an all-method full matrix, not a calibration claim.
