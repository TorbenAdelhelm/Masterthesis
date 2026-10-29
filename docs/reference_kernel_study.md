# Compare covariance kernels around each reference

The kernel study compares **separate, explicit generator assumptions** for
RUN_1, RUN_2 and RUN_3. It preserves the reference-centered lognormal law,
Gaussian latent coordinates and frozen real-K LGCNN. A useful result is a
documented shortlist for each RUN, with separate evidence about reference
coherence, training support and downstream response. The study does not infer
geological probabilities, pool RUNs into a posterior, filter generated fields
or identify one statistically calibrated covariance.

## Paired settings and comparable ranges

[`reference_scenarios.kernels.example.yaml`](../configs/reference_scenarios.kernels.example.yaml)
contains **3 RUNs × 3 kernels × 3 coupled settings = 27 scenarios**. Each
`parameter_sets` mapping is one complete setting; row and column lengths are
not independently crossed. The initial amplitude is `sigma_R: 0.05` in log10
units, with median centering and a retained-energy threshold of 0.95.

| Setting | Effective row range | Effective column range |
| --- | ---: | ---: |
| short | 250 m | 400 m |
| medium | 500 m | 800 m |
| long | 1000 m | 1600 m |

These common ranges are **illustrative model-form assumptions**, not estimates
from the three reference fields. The medium values are declared effective
ranges; they do not preserve the old Matérn numerical kernel lengths. To study
different ranges for each RUN, use separate configurations with that RUN fixed
and declare its own three settings.

An effective range `L` is the axial separation where the untruncated residual
kernel correlation equals `exp(-1)`. It resolves to the kernel's length `ell`:

| Family | One-axis correlation at separation `h` | Conversion |
| --- | --- | --- |
| `gaussian` | `exp(-h²/(2 ell²))` | `ell = L / sqrt(2)` |
| `exponential` | `exp(-abs(h)/ell)` | `ell = L` |
| `matern32` | `(1 + sqrt(3) abs(h)/ell) exp(-sqrt(3) abs(h)/ell)` | Solve `(1 + sqrt(3)t) exp(-sqrt(3)t) = exp(-1)`; `ell = L/t`. |

All three use the existing **separable product of row and column kernels**.
In particular, `matern32` denotes a product of one-dimensional Matérn-3/2
kernels, not an isotropic radial two-dimensional Matérn kernel. Matching axial
ranges makes comparisons interpretable; it does not make diagonal dependence
or spectral smoothness identical. Retained KL covariance can differ from the
untruncated declared kernel, so the validator checks the actual retained law.

Manifests retain the setting label, declared ranges, resolved kernel lengths
and assumption status. Scientific scenario IDs contain the resolved scientific
configuration, not display labels. Relabeling a setting preserves its ID;
changing its resolved covariance changes its ID. Explicit kernel lengths can
also be supplied as `ell_row_m` and `ell_col_m`. A parameter set may couple
`sigma_R` or `truncation: {n_modes: ...}` / `{energy_threshold: ...}` with its
lengths. Declaring a range and a kernel length for the same axis, or repeating
a coupled parameter on an independent matrix axis, is rejected.

## Prepare and inspect the computational dimensions

Run commands from Git Bash at the repository root, using the installed project
environment. Install optional scientific I/O dependencies when needed:

```bash
python -m pip install -e ".[geostat]"
```

Prepare the three native reference arrays following
[the existing pilot instructions](reference_pilot.md). Check that they exist
under `data/reference_pilot/`. The example uses the native 5 m grid and
`historical-training` permeability convention. A diagnostic factor of eight
changes the support of spatial texture diagnostics to 40 m; it does not
coarsen generated model inputs or their native distribution/gradient checks.

First create serialized input laws and inspect dimensions without generating
ensembles or calling the model:

```bash
python -m subsurface_uq.experiments.reference_kernel_study \
  --config configs/reference_scenarios.kernels.example.yaml \
  --output-dir run_output/kernel_study \
  --prepare-only
```

Inspect the saved plan and the scenario matrix manifest before starting the
next stage. Exponential kernels generally require more KL coordinates for a
given energy threshold than smoother kernels. Native-grid KL preparation is
itself a computation, even in prepare-only mode. A 0.95 threshold is an initial
**representation sensitivity choice**, not a guarantee of input fidelity,
temperature-QoI convergence or affordable RQMC/PCE. The existing
0.95/0.99/0.999 and explicit-dimension sensitivity workflow remains relevant.
Scrambled Sobol additionally requires a supported latent dimension (SciPy's
maximum is 21201) and a power-of-two sample budget.

Repeat the same command without `--prepare-only` for the 32-field input
diagnostic stage. The runner reuses the prepared matrix:

```bash
python -m subsurface_uq.experiments.reference_kernel_study \
  --config configs/reference_scenarios.kernels.example.yaml \
  --output-dir run_output/kernel_study
```

Completed per-scenario reports are checksummed and reused on resume; unfinished
diagnostic rendering is staged before publication. If the initial matrix
preparation is interrupted before its manifest/context exists, choose a fresh
output directory. Existing incomplete or changed artifacts are never silently
overwritten. Changing the scientific design or implementation requires a new
study directory.

Without `--prepare-only`, the `--config` form performs preparation and input
diagnostics in one run. Alternatively, start from an existing matrix with
`--matrix-manifest PATH` instead of `--config`. To inspect only selected
scenarios, add `--scenario-ids ID1 ID2` using actual IDs from its manifest. Full
27-scenario execution involves 864 generated fields for the initial budget;
these diagnostics do not query the LGCNN. The example avoids retaining all
native arrays. Each scenario has five diagnostic PNGs under
`diagnostics/SCENARIO_ID/`, plus batch comparison figures. To show a fixed subset
of the generated permeability inputs, use the existing preview exporter with
the same method, seed and full design size:

```bash
python -m subsurface_uq.visualization.reference_fields_cli \
  --input-model run_output/kernel_study/scenario_matrix/SCENARIO_ID/stochastic_input_model.yaml \
  --n-samples 32 --seed 4901 --method MC \
  --sample-indices 0 1 \
  --output-dir run_output/kernel_study/previews/SCENARIO_ID
```

Replace `SCENARIO_ID` with a real ID. The comparison PNGs show permeability,
its reference and their agreement; LGCNN temperature PNGs are separate outputs.

## Review input evidence per RUN

Use the batch comparison table/plots together with the per-scenario
`validation_report.md`, `validation.json` and PNGs. The underlying calculations
are described in [field diagnostics](reference_field_diagnostics.md).

| Evidence | What to inspect |
| --- | --- |
| Distributions and paired agreement | Physical/log-space quantiles, ratios and errors relative to the same reference; a heterogeneous reference histogram is not a pointwise lognormal marginal. |
| Native and matched-support dependence | Native gradients/lag-one dependence plus directional variograms, correlations and 2-D/radial/angular spectra at declared support. High field correlation alone is insufficient. |
| Retained-law consistency | Ensemble moments and selected native-cell covariance against the serialized KL law, including centering and conditioning. This checks implementation consistency, not physical identification. |
| Sample-budget sensitivity | Changes with the number of generated fields; a 32-field pilot provides exploratory evidence and finite-sample variability. |
| Training support | Use only fields with documented training membership and matching normalization, native resolution, patch geometry and relevant joint input channels. Reference agreement alone does not prove membership. |

No universal coherence threshold is imposed. Retain competing assumptions when
the evidence does not distinguish them. Selecting only the generator with the
smallest predicted temperature spread would bias UQ toward low uncertainty.
If training metadata is unavailable, preserve an **unassessed training support**
label. A nominal/reference patch envelope describes proximity to those
references; it must not be presented as an empirical training distribution.

## Query an explicit shortlist with the frozen LGCNN

After input review, choose at most two scenario IDs per RUN as an economical
first screen. Use the existing real-K RQ1 configuration, with verified release25
paths, checkpoints, prepared inputs and QoIs. For six selected scenarios, eight
fields each plus three nominal references require **51 predictions** before
cache reuse. This is an exploratory budget, not a convergence claim.

Create an explicit shortlist manifest from the cached input results. Replace
all six illustrative ID placeholders with IDs from the prepared matrix:

```bash
python -m subsurface_uq.experiments.reference_kernel_study \
  --config configs/reference_scenarios.kernels.example.yaml \
  --output-dir run_output/kernel_study \
  --shortlist-ids RUN1_ID_A RUN1_ID_B RUN2_ID_A RUN2_ID_B RUN3_ID_A RUN3_ID_B
```

Then query that shortlist:

```bash
python examples/reference_pilot/run_temperature_pilot.py \
  --study-mode kernel-screen \
  --matrix-manifest run_output/kernel_study/shortlist_manifest.json \
  --rq1-config configs/rq1.yaml \
  --n-samples 8 \
  --seed 5901 \
  --method MC \
  --save-temperature-samples 2 \
  --plume-threshold-c 0.5 \
  --output-dir run_output/kernel_study/temperature_screen
```

The model weights stay frozen. The selected prepared context fixes pressure,
sources and boundary-related input channels across the screen: changing the
reference permeability RUN does **not** automatically switch those inputs.
The existing pilot template uses RUN_2 as its fixed context. Record and assess
this joint input choice before interpreting the results or comparing against
simulations. A different fixed context requires a separate run and matching
physical assumptions.

Inspect nominal and sampled temperature maps, mean/spread maps, receptor/ROI
QoIs and plume descriptors alongside velocity/streamline diagnostics. The
plume anomaly threshold is an explicit descriptive assumption, not a learned
acceptance criterion. The runner caches predictions with provenance/checksums;
immutable context mismatches are rejected rather than silently reusing a
different experiment. Resume with the same options and output directory.
`--method RQMC` supports a power-of-two budget when all selected laws satisfy
the Sobol dimension limit. Use independent scrambles for RQMC error assessment.

Join the resulting output summary to the cached diagnostic comparison:

```bash
python -m subsurface_uq.experiments.reference_kernel_study \
  --config configs/reference_scenarios.kernels.example.yaml \
  --output-dir run_output/kernel_study \
  --temperature-summary run_output/kernel_study/temperature_screen/summary.json
```

This association preserves scenario IDs and separate reference results; it
does not assign probabilities or automatically rank/select generators.

Plausible LGCNN temperature maps are useful evidence about response behavior,
but do not establish in-distribution inputs or predictive accuracy. Where
feasible, compare selected generated cases with PFLOTRAN under exactly the
same permeability, pressure/boundary conditions, sources and QoI definitions.
For each retained generator, increase sample budgets and compare KL
truncations by **downstream temperature-QoI convergence** before committing to
the full UQ study.
