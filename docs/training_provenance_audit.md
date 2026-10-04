# DaRUS-5065 training-field provenance audit

The realistic LGCNN inputs are documented as permeability-field cutouts derived from Munich-region borehole data. Each raw `RUN_*` additionally carries `realistic_params.yaml` with

```yaml
orig resolution [m]: ...
rotation angle [°]: ...
start position [m]:
- ...
- ...
```

These values are treated as historical provenance metadata, **not** as a fully specified affine transform. In particular, the metadata alone does not identify the rotation center/sign, crop-before/after-rotation order, pixel-center convention, parent-map origin, or the exact 20 m -> 5 m interpolation kernel. The code must not guess those details.

## Audit command

Run the audit against the unpacked DaRUS-5065 dataset, the Munich measurement workbook and the same full Munich reference table used by the measurement workflow:

```powershell
python -m subsurface_uq.experiments.training_provenance `
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" `
  --measurements "C:/path/to/kf_werte_190201.xlsx" `
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" `
  --runs RUN_1 RUN_2 RUN_3 RUN_4 `
  --output-dir run_output/realistic_k/provenance
```

Omit `--runs` to inspect every `RUN_*` containing both `pflotran.h5` and `realistic_params.yaml`. The default `--reference-mode sweep` tests all nine combinations of `K_P10/K_P50/K_P90` and `top/bottom/log_geomean` and accepts an absolute mapping only if one representation maps consistently across all requested runs under the existing georeference validation thresholds.

## Outputs

The audit writes

```text
training_provenance.yaml
training_provenance.json
georeference_rows.csv
measurement_training_summary.csv
measurement_training_pairs.csv
overlays/
  RUN_1_measurement_overlay.png
  ...
```

For every validated run, the measurement overlay displays the georeferenced training field and the converted intrinsic-permeability measurements on the same `log10(k)` color scale. The second panel shows `training - measurement` residuals at the measurement locations.

The numerical comparison reports, per run,

- number of measurement locations lying in the reconstructed cutout;
- RMSE and MAE in `log10(k)`;
- mean bias `training - measurement`;
- log-space correlation;
- fractions agreeing within 0.05 and 0.10 log10 units.

The audit also compares the YAML `start position` values with independently inferred georeferences. It only checks whether simple run-to-run translation relations are consistent. It deliberately does not declare a historical rotation/crop convention resolved from those offsets alone.

## Interpretation rules

A successful absolute template match plus good measurement agreement supports the statement that a training cutout is spatially consistent with the Munich measurement-derived permeability map. It does **not** prove that `kf_werte_190201.xlsx` is the exact historical workbook used to build the map.

Likewise, a validated match against the available 100 m reference table does not by itself reconstruct the exact historical 20 m parent surface or the 20 m -> 5 m resampling procedure. `training_provenance.yaml` therefore keeps

```yaml
exact_historical_rotation_crop_resampling_reconstructed: false
exact_measurement_workbook_used_historically_verified: false
safe_for_residual_generator_calibration: false
```

until independent evidence resolves those steps.

## Consequence for stochastic generation

Do not yet fit the final residual generator solely from the per-run YAML. The intended next model is

```text
log10(k)(x, eta) = m_measurement(x) + R(x, eta)
```

where `m_measurement` is reconstructed from the measurement-derived parent-map family and `R` is calibrated from registered training-field residuals. This becomes scientifically justified only after the provenance audit establishes a defensible common spatial mapping and the historical map/resampling relation is either recovered or shown to be negligible for the intended residual scale.
