# Compact three-kernel study

Use `configs/controlled_field.kernels.compact.500.yaml` for the requested
Gaussian/Matérn/exponential × 64/128/256-observation × 500-sample study.
The conditional distributions, paired seeds, hidden truths, matching RUN inputs
and full-resolution temperature statistics are the same scientific experiment
as [the full-archive workflow](kernel_density_study.md). Compact storage changes
the execution and retention policy. It does not reduce the MC sample count,
coarsen the model inputs, add a nugget or filter generated fields.

## Run on Linux

Use the existing model/upstream and matching-reference asset bundles. No new
binary assets are needed. From the repository root, in the Python environment:

```bash
git switch feat/reference-field-uq
git pull --ff-only
python -m pip install -e ".[geostat,perlin]"

subsurface-uq-controlled-fields \
  --config configs/controlled_field.kernels.compact.500.yaml \
  --reference-id RUN_4 \
  --output-dir run_output/kernel_study_compact \
  --stage all
```

Use a **fresh output directory**. Existing full-archive studies keep their old
configuration and format. Results are isolated below the selected RUN ID.
Use `controlled_field.kernels.compact.pilot.yaml` with a different output parent
for the eight-draw pilot. It has the same geometry and random-stream prefixes.

After interruption repeat the same command with `--stage resume`. Keep the
recorded code, configuration, assets and numerical environment. The compact
runner verifies these before continuing. `plan` writes metadata only;
`validate` checks native FFT/conditioning and independent dense marginals;
`inputs` prepares hidden truths, exact readings, analytic maps and regeneration
recipes. Batched MC input generation takes place during `temperatures`.

The production budget remains **6,003 successful model evaluations per RUN**:
4,500 conditional draws, 1,500 zero-observation baseline draws and three hidden
truths. One hidden truth is used per kernel; 500 conditional samples do not
constitute 500 independent truths for coverage calibration.

## Retained data and disk budget

The default compact policy retains:

| Item | Approximate size per selected reference |
|---|---:|
| All 6,000 native 1028² temperature outputs, float32 | 25.36 GB |
| First eight native 2560² permeability inputs in each of 12 cases | 2.52 GB |
| Log-permeability previews of every sample, 96² float32 | 0.22 GB |
| Analytic input moments, temperature statistics/risk maps and hidden truths | 1.39 GB |
| Browser images, records, recipes and figures | Depends on image compression |

The core retained arrays total **29.49 GB**. Native PNGs of the selected inputs
and outputs and overview atlases keep retained results within approximately
**30–35 GB per reference**. The plan records raw-array sizes and conservative
RGBA image bounds, without assuming compression savings.

The default batch size is 16. One batch of float64 prior fields plus its four
float32 input variants occupies at most **2.52 GB of temporary disk space**.
It is removed after the corresponding prediction records and retained data are
committed. No 500-field prior scratch and no native grid-by-observation cache
are created. For this preset the conservative estimated peak is approximately
**33.94 GB**, including the separate report-image phase; allow filesystem
overhead and inspect `manifest.json -> compact_storage` for the exact plan.

These values include all three kernels and their baselines. Additional RUNs,
outer truths, observation designs or larger preview/retention settings add
storage. Disk space and RAM are separate: native fields are memory-mapped and
spatially blocked; all 500 inputs are never held in RAM together.

## Computation and recovery

Analytic conditional means/variances are prepared once using the original
whitened Gaussian equations. For each MC batch, the small observation systems
are solved for the innovations, and spatial blocks apply
`C_XB @ solve(C_BB, innovation)`. This is the same Matheron conditional draw as
the whitened implementation. It avoids repeating the full spatial whitening
solve for each batch. There is no reduced-precision covariance or added noise.
Native preflight checks the batch formulation's exact interpolation as well.

Every batch uses **global** sample indices and the unchanged semantic FFT seeds.
Prefixes share the same prior draw, and profiles retain their paired driving
modes. The input-generation BLAS thread count is fixed at one to make native
input fingerprints independent of the model's thread setting; FFT worker
counts remain configurable. Numerical versions are recorded and checked.

After each successful prediction, the runner flushes the native temperature,
input preview and, for the fixed retained subset, full native input. It then
atomically commits the per-sample record containing their fingerprints. On
restart it verifies those committed rows. A partial prior batch continues from
its recorded draws; an interrupted conditioning pass restarts its batch; a
partial prediction batch reuses its committed inputs and finished predictions.
An invalid prediction or failed interpolation stops the run without replacing
the draw. Temporary batches are deleted only within their checked study paths.

## Visualization and analysis

Open `ensemble_browser.html` inside the selected RUN directory. Every sample has
input/output, hidden-truth-error and conditional-deviation overview maps with
common scales. **Open native-resolution image** links are provided for the
fixed first eight samples per case: the inputs are 2560² and the outputs 1028².
The 96² previews can omit narrow plumes; their resolution does not constrain
the model or the uncertainty calculations. The browser releases image caches
when changing cases.

Each case retains `temperature_ensemble.npy`, `permeability_selected.npy`,
`log_permeability_previews.npy`, `prediction_records.json`, analytic input maps,
temperature moments/quantiles, risk products, scalar QoIs, convergence checks
and spatial previews. The retained input indices are fixed before viewing
results. They are not selected for appearance or agreement with the truth.
Existing profile-sensitivity and calibration reports use native output arrays.

## Regenerate an unretained input

Each profile/truth has `input_recipe.json` and its integrity ledger. This stores
the covariance law, mean, native grid, exact revealed readings, geometry, seed
contract, batch settings, numerical versions and generator-source hashes.
For example, regenerate global sample index 137 (the 138th draw):

```bash
subsurface-uq-regenerate-input \
  --study-dir run_output/kernel_study_compact/RUN_4 \
  --profile RUN_4_gaussian \
  --n-measurements 128 \
  --sample-index 137 \
  --output-field regenerated/RUN_4_gaussian_n128_sample137.npy
```

The temperature case must be complete. The tool rebuilds the original batch
with its original RHS shape/order and verifies the requested native input
against the fingerprint recorded before inference. Only then does it write
the fresh `.npy` file and a provenance sidecar. It needs one temporary batch's
space and deletes that scratch afterwards. It performs no model evaluations.

Original source maps, prepared model inputs, model weights and the original
configuration file are unnecessary for regeneration. Stored results can be
moved together; internal recipes use relative case paths. Use the recorded
generator version and numerical packages. A fingerprint mismatch is an error,
not an approximately matching replacement. Cross-environment bitwise identity
is not assumed. `--design` and `--truth-index` select additional configurations
when present. All sample indices in the regeneration command are zero-based.

## Implementation checks

The complete local suite passed **692 tests**, including agreement with the
full-archive MC results to float32 precision, interruptions during prior
generation and inference, partially completed batches, the final short batch,
multiple observation layouts, selected-input/preview corruption, matching RUN
selection and standalone regeneration after moving results and removing the
original source/configuration files.

All three kernels at 64/128/256 observations passed native 2560² FFT,
conditioning and independent dense-equation checks for RUN_0/2/3/4. The largest
batch-formulation interpolation discrepancy in the deliberately adverse
preflight readings was 5.06e-8 log10 units (Gaussian RUN_0); the guard is 1e-5.
[Native validation record](compact_kernel_native_validation.json).

A complete native Gaussian RUN_4 integration run used four MC samples per case,
batch size three and two retained inputs. It completed **17 real LGCNN queries**,
all at native 1028² output resolution, produced the browser and 16 native PNGs,
and regenerated an unretained input from the final one-sample batch with an
identical 2560² float32 array fingerprint. No disposable batches remained.
[Integration record](compact_kernel_native_integration.json).
This is an implementation check; the full 500-sample production study was not
executed here. Its storage figures above are calculated estimates.
