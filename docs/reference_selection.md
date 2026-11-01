# Select a reference and its matching physical context

Use `--reference-id RUN_N` to restrict a controlled benchmark or density gallery
to one source reference. The selection binds the source-map structural profile
to the same RUN's pressure/material/heat-pump inputs. It is not a change of a
plot label, and there is no fallback to RUN_2 when matching inputs are absent.

## The four complete DaRUS cases

The published [DaRUS-5065](https://doi.org/10.18419/DARUS-5065) describes four
native 2560 by 2560 simulation cases and a separate larger scalability case.
The local `real_perm_4dp.zip` has MD5 `e008c65555bf8e435ba84a678990236b`, matching
the published archive checksum. Inspecting that ZIP's directory and the extracted
HDF5 initial-time channels gives:

| RUN ID | Permeability input | PFLOTRAN output | Matching native model context |
|---|---|---|---|
| RUN_0 | Present | Present | Available |
| RUN_1 | Present | Absent from the published ZIP | Unavailable from this archive |
| RUN_2 | Present | Present | Available |
| RUN_3 | Present | Present | Available |
| RUN_4 | Present | Present | Available |

The historical preparation script explicitly defaulted to RUN_1, RUN_2 and
RUN_3. Consequently the earlier three-map catalog contains an input-only RUN_1
and omits complete RUN_0 and RUN_4. This was a reduced pilot selection, not an
inventory of the four published simulations. RUN_1's absent output is also
absent inside the checksum-verified archive; it is not a local extraction loss.
The separate `real_perm_1dp_scaling.zip` is the 5120 by 2560 scalability case,
not the explanation for RUN_1's missing simulation output.

Historical catalogs, inputs and results remain unchanged. The new
`structural_profiles.complete_references.json` covers RUN_0, RUN_2, RUN_3 and
RUN_4 separately, using the existing Matérn-3/2 structural estimator. The source
maps and each case's preparation provenance are independently hashed. No
geological weights are assigned. The Gaussian/exponential structural fits and
the full three-kernel 500-sample experiment remain separate preparation work.

## Prepare matching inputs on Linux

Copy or extract the original dataset and retain the model/upstream assets as
described in the Linux asset instructions. Inspect available native cases with:

```bash
python examples/reference_pilot/prepare_data.py \
  --dataset-root /path/to/dataset_100hp_giant_real_fixP0_0025 \
  --list-references
```

Prepare one selected reference and its matching physical inputs:

```bash
python examples/reference_pilot/prepare_data.py \
  --dataset-root /path/to/dataset_100hp_giant_real_fixP0_0025 \
  --cnn1-dir models/LGCNN_step1_realK \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --reference-id RUN_4 \
  --output-dir data/reference_cases
```

This writes `data/reference_cases/RUN_4/`, containing `RUN_4.reference.npy`, its
own `prepared_pki/info.yaml`, `prepared_pki/Inputs/RUN_4.pt` and preparation
manifest. Repeating for another RUN uses a separate directory. RUN_1 is rejected
for matched-context preparation. Existing preparation outputs are never replaced.
For legacy multi-reference extraction, explicit `--reference-runs` and
`--fixed-run` remain supported; the new default discovers complete native cases.

The committed catalog checks the exact prepared arrays and provenance used for
its fit. Use the supplied matching asset bundle to reproduce it byte for byte.
Fresh data preparation can change provenance timestamps or serialized bytes;
in that case fit a fresh catalog using
`configs/structural_profile_fit.complete_references.yaml`, a fresh `--output`
and `--report-dir`, and update the study's catalog and scenario laws together.
Do not edit or bypass the frozen catalog's hashes.

## Select and run one case

The matched-reference presets require the argument explicitly. First plan a
small Matérn pilot:

```bash
subsurface-uq-controlled-fields \
  --config configs/controlled_field.references.pilot.yaml \
  --reference-id RUN_4 \
  --output-dir run_output/reference_study \
  --stage plan
```

Continue using the same config, reference ID and output parent with stages
`validate`, `inputs`, `temperatures`, then `report`. For a fresh output parent,
`--stage all` executes those stages including planning. Do not run `all` after
`plan` in the same output, because `all` requires a fresh case directory.

The single-reference 500-draw Matérn configuration is
`configs/controlled_field.references.500.yaml`. It includes the mandatory
zero-observation baseline plus 64/128/256 readings, one hidden truth and one
balanced layout: 2001 model evaluations for the selected RUN. Its configuration
can be planned and densely validated without completing that propagation budget.
It is not the three-kernel production study or a calibration/convergence claim.
The general controlled path currently stores a native geometry cache; at 256
readings it occupies 12.5 GiB per layout. Use a fresh output and evaluate machine
memory/storage before choosing that budget.

The eight-draw spatial illustration also accepts the reference argument:

```bash
subsurface-uq-density-gallery \
  --config configs/conditional_density.references.yaml \
  --reference-id RUN_4 \
  --output-dir run_output/reference_gallery \
  --stage all
```

Results are isolated as:

```
run_output/reference_study/RUN_4/manifest.json
run_output/reference_study/RUN_4/RUN_4_structural/truth-0000/...
run_output/reference_gallery/RUN_4/manifest.json
run_output/reference_gallery/RUN_4/n-0064/...
```

Manifests record the selected reference, source hashes, effective law, matching
prepared input, model context and seeds. Resume only the same case and execution
context. Model factories reporting a different fixed RUN are rejected before
prediction. Selection does not change the existing truth/posterior seed contract.

The new model template is `configs/frozen_temperature.by_reference.example.yaml`.
Its `prepared_pki_dir` contains `{reference_id}`, expanded by the frozen loader.
The study uses that YAML and `model_asset_root: ..`, resolving asset paths from
the repository root. The template retains the existing spatial receptors/ROI;
these are fixed output cells and are not automatically relocated relative to
each RUN's heat pumps. Declare different QoIs explicitly when that is intended.

Without `--reference-id`, historical configurations retain their previous
multi-profile/fixed-context behavior and output layout.
