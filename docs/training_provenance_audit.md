# DaRUS-5065 training-field provenance audit

The realistic LGCNN permeability inputs are spatial cutouts of one Munich parent
hydraulic-conductivity raster, not independent realizations of a stochastic
permeability field. The historical generation code stores, for every `RUN_*`,

```yaml
orig resolution [m]: 20
rotation angle [°]: ...
start position [m]:
- ...
- ...
```

and then

1. builds a local source-cell mesh,
2. rotates it with `angle + 90°`,
3. translates it by the stored start position,
4. truncates floating source indices with NumPy `astype(int)`,
5. extracts the 20 m parent-raster values,
6. converts `K_h` to the training permeability convention with
   `k = K_h / 7.5e6`, and
7. uses `scipy.interpolate.RegularGridInterpolator` to sample the extracted
   window at the PFLOTRAN cell centres.

The historical interpolation code contains a TODO for a half-source-pixel
correction but does not apply it. Reconstruction intentionally reproduces that
behavior rather than correcting it.

## Exact parent raster supplied with the thesis data

The supplied files establish the parent source much more strongly than the
earlier reference-map matching heuristic:

```text
Hydraulic_conductivity_20m_resolution.tif
CRS: EPSG:5678
name: DHDN / 3-degree Gauss-Kruger zone 4 (E-N)
shape: 3797 x 3583 [rows x columns]
resolution: 20 m x 20 m
SHA-256:
6d50f2c6f9b96e3136fe77ad177ef4f70cc2d4cfc632f33ff655e59bdcc096ee
```

`Cond_20.tif` supplied with the same source material is byte-for-byte identical
to `Hydraulic_conductivity_20m_resolution.tif`, including its GeoTIFF metadata.

This also corrects an earlier provenance assumption: the exact supplied parent
raster is **not** EPSG:25832. The separate historical `reproject_tiffs.py`
script targets EPSG:25832, but the parent raster actually supplied to the thesis
workflow is EPSG:5678 and the historical realistic-window generator operates on
the raster array itself. Consequently the reconstruction code now validates
against EPSG:5678, 20 m resolution and the exact observed raster dimensions. An
optional strict checksum switch can additionally demand the exact byte-level
source file.

The accompanying R preprocessing script supports the 20 m lineage: it aggregates
the depth raster to 20 m and bilinearly resamples the conductivity raster onto
that grid before writing the 20 m output. It does not establish how the earlier
10 m conductivity surface itself was inferred from borehole measurements.

## 1. Exact RUN reconstruction

Run against the local DaRUS-5065 directory:

```powershell
subsurface-uq-historical-realistic-reconstruction `
  --dataset-root "C:/Users/Torbe/Desktop/MT/Daten/Trainingsdaten/dataset_100hp_giant_real_fixP0_0025" `
  --parent-hydraulic-conductivity-tif "C:/path/to/Hydraulic_conductivity_20m_resolution.tif" `
  --require-exact-parent-checksum `
  --output-dir run_output/realistic_k/historical_reconstruction
```

Omit `--runs` to reconstruct every `RUN_*` containing both `pflotran.h5` and
`realistic_params.yaml`.

The command tries the release25 raw-array orientation candidates when comparing
the reconstructed field with the stored PFLOTRAN permeability array. It writes

```text
historical_reconstruction.yaml
historical_reconstruction.json
historical_reconstruction_summary.csv
reconstructed/
comparisons/
```

The default numerical reconstruction thresholds are

```text
RMSE(log10 k) <= 1e-3
correlation(log10 k) >= 0.999
```

These are code-validation thresholds, not stochastic-model acceptance
thresholds.

## 2. Parent raster versus the measurement workbook

The parent GeoTIFF has a complete affine georeference, so the borehole workbook
can now be compared directly with the exact parent field without first fitting a
100 m reference-map alignment:

```powershell
subsurface-uq-parent-measurement-audit `
  --parent-hydraulic-conductivity-tif "C:/path/to/Hydraulic_conductivity_20m_resolution.tif" `
  --measurements "C:/path/to/kf_werte_190201.xlsx" `
  --output-dir run_output/realistic_k/parent_measurements
```

The default semantic filters remain

```text
Strategrap == q
GW_Zustand == ungespannt
```

The audit reports both bilinear and nearest-cell parent values, with

- log10 RMSE and MAE,
- parent-minus-measurement bias,
- log-space correlation,
- agreement fractions within 0.05, 0.10 and 0.30 log10 units,
- median parent/measurement ratio,
- a per-measurement CSV, and
- spatial/scatter diagnostics.

Agreement demonstrates consistency between the workbook and parent map. It
does **not** prove that `kf_werte_190201.xlsx` was the exact or sole historical
input from which the parent map was constructed.

## 3. Consequence for the stochastic input model

The three/four realistic LGCNN training fields must not be treated as
independent geostatistical realizations. They are deterministic windows of one
parent Munich field. Therefore:

```text
training RUN fields -> LGCNN support / numerical-provenance reference
parent Munich raster -> spatial trend / historical field reference
borehole measurements -> observational information and uncertainty evidence
```

The production input-UQ law should remain measurement-driven until the
measurement/parent relation is explicitly included and sensitivity-tested. A
scientifically defensible extension is a residual model

```text
log10(Kh)(x, eta) = log10(Kh_parent)(x) + R(x, eta)
```

with `R` inferred from measurement-minus-parent residuals. This is preferable
to estimating a random-field prior from between-RUN variability because the
RUNs are windows of the same underlying map.

However, the parent map may itself have been constructed using overlapping
borehole data. Consequently a residual model cannot be presented as an
independent validation model unless the upstream interpolation provenance is
recovered. For the thesis, the current measurement-only kriging/KL law remains
the primary UQ model; a parent-trend residual law should be added as a
provenance-informed sensitivity/alternative model and compared against it.

## 4. Remaining validation step

The exact TIFF provenance is now established. The remaining local-data check is
to run the exact reconstruction command against the stored `RUN_* / pflotran.h5`
files. If all training RUNs pass the numerical thresholds, the complete chain

```text
20 m parent raster
  -> historical crop/rotation
  -> historical 20 m -> 5 m interpolation
  -> PFLOTRAN permeability input
  -> stored training field
```

is validated directly and the older 100 m template-search georeference becomes
a secondary cross-check rather than the primary source of spatial provenance.
