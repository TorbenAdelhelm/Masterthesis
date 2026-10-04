# DaRUS-5065 training-field provenance audit

The historical generation path for the realistic DaRUS permeability inputs has now been recovered from `JuliaPelzer/Dataset-generation-with-Pflotran`, branch `windowed_real_vals_v2`, commit `c13ccea3fccd8180ea682ea30d5622fac0c333eb`.

This is stronger evidence than inferring the per-run YAML convention from the finished fields. The recovered code establishes the following pipeline.

## Historical realistic-permeability pipeline

The parent hydrogeological properties were loaded from 20 m GeoTIFFs. The hydraulic-conductivity source file is named

```text
Hydraulic_conductivity_20m_resolution.tif
```

and the repository contains the reprojection script used to produce EPSG:25832 source rasters.

For every valid run, `realistic_params.yaml` records

```yaml
orig resolution [m]: 20
rotation angle [°]: ...
start position [m]:
- ...
- ...
```

The historical generator writes `start position [m]` as the selected parent-array cell index multiplied by the 20 m source resolution. Thus these values are positions in the parent raster's array coordinate system, not projected EPSG:25832 Easting/Northing coordinates.

The rotated extraction mesh is exactly

```text
rotation_rad = deg2rad(rotation_angle_deg + 90)
rot_x = x*cos(rotation_rad) - y*sin(rotation_rad) + start_x_cell
rot_y = x*sin(rotation_rad) + y*cos(rotation_rad) + start_y_cell
```

and parent values are then selected with

```text
data[rot_y.astype(int), rot_x.astype(int)]
```

so the rotated floating source indices are truncated by NumPy integer conversion.

The historical code converts hydraulic conductivity to the permeability supplied to PFLOTRAN as

```text
permeability = hydraulic_conductivity / 7.5e6
```

This conversion is part of the historical data provenance and differs slightly from the fluid-property conversion `k = K_h*mu/(rho*g)` used by the current measurement-UQ workflow. Direct numerical comparisons with the stored training fields must therefore distinguish **historical training permeability units/conversion** from the later physical intrinsic-permeability convention.

The 20 m cutout is finally interpolated to the PFLOTRAN mesh with `scipy.interpolate.RegularGridInterpolator`. The query coordinates are the PFLOTRAN cell centers, e.g. `(i+0.5)*5 m` and `(j+0.5)*5 m` for the 5 m training grid. A TODO in the historical interpolation code explicitly notes a possible source-pixel `+0.5*resolution` correction, but that correction was not applied. The reconstruction code reproduces the implementation as it was actually used rather than applying the TODO retrospectively.

## Stage 1: exact historical field reconstruction

If the historical/reprojected 20 m parent hydraulic-conductivity GeoTIFF is available, use the exact reconstruction command:

```powershell
python -m subsurface_uq.experiments.historical_realistic_reconstruction `
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" `
  --parent-hydraulic-conductivity-tif "C:/path/to/Hydraulic_conductivity_20m_resolution.tif" `
  --runs RUN_0 RUN_1 RUN_2 RUN_3 `
  --output-dir run_output/realistic_k/historical_reconstruction
```

Omit `--runs` to reconstruct every `RUN_*` containing both `pflotran.h5` and `realistic_params.yaml`.

The command reproduces the recovered historical rotation, integer source indexing, `K_h/7.5e6` conversion and 20 m -> 5 m interpolation. It writes reconstructed NPY fields, stored-vs-reconstructed comparison PNGs, CSV metrics and YAML/JSON provenance. By default a run is called numerically reconstructed only if the best raw-array orientation has

```text
RMSE(log10 k) <= 1e-3
correlation(log10 k) >= 0.999
```

These thresholds are verification tolerances, not stochastic-model tuning parameters.

The supplied parent raster is also checked against the recovered historical metadata: 20 m resolution and EPSG:25832 by default.

## Stage 2: place training runs on the measurement/reference coordinate system

The existing reference-map matching remains useful because the historical start coordinates are parent-array positions, whereas the measurement workbook/reference table uses projected coordinates. Run:

```powershell
python -m subsurface_uq.experiments.training_provenance `
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" `
  --measurements "C:/path/to/kf_werte_190201.xlsx" `
  --reference-grid "C:/path/to/3D_K_Field_Munich_K_P10_P50_P90.csv" `
  --runs RUN_0 RUN_1 RUN_2 RUN_3 `
  --output-dir run_output/realistic_k/provenance
```

The default `--reference-mode sweep` tests all nine `K_P10/K_P50/K_P90 x top/bottom/log_geomean` representations and only accepts an absolute mapping if one representation maps consistently across all requested runs under the configured validation thresholds.

The audit writes

```text
training_provenance.yaml
training_provenance.json
georeference_rows.csv
measurement_training_summary.csv
measurement_training_pairs.csv
overlays/
  RUN_0_measurement_overlay.png
  ...
```

For every validated run, the overlay places measurement values on the reconstructed training-field footprint and reports per-run RMSE/MAE/bias/correlation in log space.

### Important conversion caveat

The stored DaRUS training fields were historically constructed with `k_training = K_h/7.5e6`. The current measurement-UQ pipeline uses `k_physical = K_h*mu/(rho*g)`. These differ by a constant log shift. For provenance claims about exact training-field values, the historical conversion is authoritative; for physical UQ reporting, the fluid-property conversion remains explicit. Do not interpret a constant offset between workbook-derived values and training fields as geological disagreement until this conversion difference has been removed.

## What can and cannot be concluded

If Stage 1 validates all requested runs, then the stored DaRUS permeability fields are numerically reproducible from the supplied 20 m parent hydraulic-conductivity map using the historical extraction code. That establishes the training-field provenance independently of the stochastic generator.

If Stage 2 additionally finds the workbook measurements at the corresponding locations and they agree after the historical conversion/preprocessing is respected, that is strong evidence that the fields and measurements belong to the same Munich permeability-map lineage.

It still does **not** prove that `kf_werte_190201.xlsx` is the exact historical workbook used to create the parent GeoTIFF. Establishing that requires source-data provenance for the R preprocessing that produced the 20 m hydraulic-conductivity raster, or an equivalent direct record.

## Consequence for stochastic generation

The intended next model remains

```text
log10(k)(x, eta) = m_measurement(x) + R(x, eta)
```

but the residual `R` should only be calibrated after the historical parent map and stored RUN fields have been registered consistently. At that point we can compare the historical parent-map prediction and the stored fields at exactly corresponding pixels, then model only the remaining uncertainty/structure instead of asking a low-rank stationary KL expansion to recreate the complete training morphology from scratch.

Until that numerical provenance check is run on the local data, the production generator should not silently switch to a residual model merely because the YAML metadata exists.
