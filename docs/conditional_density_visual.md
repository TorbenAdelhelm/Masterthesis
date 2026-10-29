# Spatial study at 32–512 observations

This is a small, paired illustration of how observation density changes
conditional permeability fields and their frozen-LGCNN temperature outputs.
It uses the fixed RUN 2 structural profile, one new synthetic hidden field,
and **eight draws at each of 32, 64, 128, 256 and 512 exact observations**.
All 40 conditional inputs are queried, along with one hidden-reference input.
See the [completed results and all-count atlas index](conditional_density_visual_results.md).

## What is held fixed

The source profile is the frozen `RUN_2_structural` law in
`configs/structural_profiles.json`: mean log10 permeability -9.3206109166,
pointwise SD 0.2067964072, anisotropic Matérn-3/2 scale parameters
1816.91/2669.82 m and major-axis angle 84.32 degrees. The field grid is
2560 by 2560 at 5 m. Pressure, material, heat-pump locations, checkpoints,
streamline settings and the native temperature crop remain fixed.

The experiment seed is 2026102606. The hidden truth and eight Gaussian driving
draws have separate semantic random streams. Each draw is reused across
observation counts, so sample 1 follows the same underlying randomness through
all five counts. The 512-location sequence is chosen by a geometry-only
maximin rule on a 160 m candidate lattice. Each smaller observation set is
the corresponding prefix. Locations are chosen without inspecting field or
temperature values. Their readings come from the single hidden log-field.

Parameters are not refitted at any count. There is no measurement nugget,
draw filtering, histogram matching, moment normalization or temperature-driven
selection. These are synthetic observations, not registered historical Munich
boreholes. The [structural profile assumptions](structural_profiles.md) and the
separate [RUN realism challenge](measurement_density_results.md) retain their
previous interpretations.

## Exact conditioning without a large geometry file

The conditional simulation is the same Matheron update used in the controlled
benchmark. With R the unit correlation, factor the largest observation matrix
as `R_BB = L Lᵀ`. For each spatial block, compute `F = R_XB L⁻ᵀ` once. For
each prefix n and draw s, solve

```text
alpha_n,s = L_n^-1 (d_n - mu - sigma Z_s(B_n))
Y_n,s(X) = mu + sigma Z_s(X) + F[:, :n] alpha_n,s.
```

All prefix/draw corrections are applied together before advancing to the next
block. The analytical mean and diagonal conditional variance use the same
features. This avoids storing the full grid-by-512 geometry, which would be
**25 GiB** in float64. All native permeability draws are retained in five
float32 ensemble arrays, about **0.98 GiB** in total.

The 512-observation correlation matrix has condition number approximately
20018; its smallest/largest eigenvalue ratio is about 5.0e-5, well above the
1e-12 numerical-rank threshold. Conditioning at 512 observations is feasible
without a nugget or changing the covariance.

## Spatial outputs and deviations

The local run directory is
`run_output/conditional_density_visual_20261026/`:

- `report.html`: offline interactive gallery, count selector, all eight samples,
  individual-sample views, observation overlays, shared/detail colour scales,
  spatial overview and native-array links.
- `report.md`: complete static spatial report with the input and output atlases
  for every observation count.
- `figures/n-NNNN-inputs-shared.png` and `...outputs-shared.png`: every input
  or output beside its signed error against the hidden reference.
- `...inputs-detail.png` and `...outputs-detail.png`: explicitly labelled
  per-count error contrast for inspecting the smaller high-density deviations.
- `figures/n-NNNN/sample-SS-*.png`: all individual input/output fields, reference
  errors, and deviations from the empirical eight-draw log/temperature mean.
- `n-NNNN/permeability_samples.npy`: the eight actual native model inputs;
  `temperature_samples.npy`: their eight actual native outputs.
- Analytical log-space posterior moments, empirical ensemble means/SDs,
  observation values, sample seeds, query hashes and the frozen model context.

Permeability reference error is `log10(K_sample/K_hidden)`. A value +0.301
means twice the hidden permeability; -0.301 means half. Temperature reference
error is `T_sample - T_hidden_LGCNN` in C. The eight-draw ensemble deviations
are different quantities: they show disagreement around an estimated mean.
Neither error nor ensemble spread is silently substituted for the other.

Shared colour limits are fixed across counts by default. Detail limits change
only the display scale and are marked as local; they cannot support comparisons
of amplitudes across counts. Input previews use a four-cell point stride
(20 m); native 5 m inputs remain available. Temperature preview images retain
the full 1028 by 1028 output resolution. The dashed input-domain square shows
the temperature crop at local coordinates 3.83–8.97 km on both axes.

## Reproduction and recovery

```powershell
subsurface-uq-density-gallery --config configs/conditional_density.visual.yaml --output-dir run_output/conditional_density_visual_new --stage all
```

The stages are `plan`, `inputs`, `temperatures` and `report`. `all` and `plan`
require a fresh directory. Interrupted model queries resume at the next
uncommitted sample using the same input and seed, after saved results are
verified. Source/configuration/model changes require a fresh run. The gallery
template is included in the installed package and in the run's source hashes.

## Interpretation limits

The hidden temperature reference is the frozen LGCNN evaluated on the synthetic
hidden permeability, not a physical-simulator solution. This study isolates
the response of one known input law and one fixed layout/truth. It does not
establish calibration, geological accuracy, tail-risk convergence or a universal
best observation count. Individual temperature errors or finite eight-draw
spread estimates need not decrease at every location or every count.
