# Output-ensemble KL and Gaussian-field assessment

This is the first step towards a reduced input-to-output distribution model. It
asks two separate questions using the **retained native LGCNN outputs**:

1. Can a training-only KL/PCA basis represent unseen temperature fields well?
2. Can a multivariate Gaussian law on those coefficients reproduce useful
   spatial risks and affected-area distributions?

The command performs the assessment immediately. It neither regenerates
permeability fields nor queries the LGCNN. It needs no model weights, upstream
LGCNN installation or original site maps. Source study files are read-only.
The output is a standalone local HTML report, figures, native probability maps,
numerical diagnostics and a reusable sample-indexed coefficient dataset.

## Run one configuration first

After installing this branch (`python -m pip install -e .`), on Linux:

```bash
python -m subsurface_uq.experiments.output_field_assessment \
  --study-dir run_output/reference_study/RUN_4 \
  --output-dir run_output/output_field/RUN_4_gaussian_n256_r32 \
  --profile RUN_4_gaussian --n-measurements 256 \
  --max-rank 32 --variance-fraction 0.99 \
  --gaussian-draws 1000 --threads 4
```

The equivalent installed command is `subsurface-uq-output-field-assessment`.
Replace the source path with the directory that contains the completed
`manifest.json`. Use a **fresh** destination outside that source directory.
Open its `index.html` locally. No server or browser assets are required.

Repeat `--profile` and/or `--n-measurements` to select multiple values. Omit both
filters to assess every completed configuration, including the zero-observation
prior. Cases remain separate; no geological weights are assigned to kernels.
The exact profile names in the current matched RUN_4 study are
`RUN_4_gaussian`, `RUN_4_matern32_radial`, `RUN_4_exponential_radial`.

The default random split is **350 training / 150 held-out** for 500 fields. The
same seed and global draw indices are used across equally sized cases, preserving
the production study's pairing. Each case fits its own basis and law; coefficients
from different case-specific bases cannot be compared coordinate by coordinate.

Optionally declare an affected-area decision event, for example:

```bash
  --area-event 0.1:9000000
```

This means `P(area with warming > 0.1 °C exceeds 9,000,000 m²)`. The number is
an **example**, not a proposed regulatory limit. The report exports probabilities
and exact binomial intervals for the direct, reconstruction and Gaussian samples.
Only thresholds already declared by the source study are allowed.

## Formal model and fitting

Let `t_i` be a temperature field flattened over `M` native output cells, with
training mean `mu` and centered training matrix `X` (rows are independent draws).
For `n` training draws, the snapshot covariance is

```text
G = X Xᵀ / (n−1),                    G u_j = lambda_j u_j
phi_j = Xᵀ u_j / sqrt((n−1) lambda_j)
a_ij = (t_i − mu)ᵀ phi_j
T_r(x) = mu(x) + sum[j=1..r] a_j phi_j(x).
```

This is an empirical **nonstationary** basis. We do not assume the output has
the stationary Gaussian/Matérn/exponential covariance used to generate the
log-permeability inputs. The output covariance reflects the fixed flow,
heat-pump arrangement and nonlinear frozen network. Uniform physical cell weights
only rescale eigenvalues and basis normalization; they do not change reconstructed
temperatures. Reported eigenvalues use the unweighted cell-sum convention.

Rank is the smallest training-spectrum rank meeting the requested variance
fraction, **capped by `--max-rank` and numerical rank**. A capped fit explicitly
reports an unmet target. Explained variance is a training compression diagnostic,
not an output-distribution validation. Constant fields give a rank-zero,
degenerate Gaussian model. The discarded residual covariance is **not** added
back as independent noise; its consequences appear in held-out errors.

The candidate Gaussian field is

```text
A ~ Normal(m_a, S_a),     T_G(x) = mu(x) + phi(x)ᵀ A,
Cov[T_G(x), T_G(y)] = phi(x)ᵀ S_a phi(y).
```

`m_a` and `S_a` use **training scores only**, with the full coefficient covariance
retained. This is a finite-rank Gaussian random field (a GP restricted to the
native grid). Its marginal sampling is legitimate Monte Carlo under this fitted
law. The fitted law is an approximation to the LGCNN pushforward, not an exact
consequence of Gaussian input fields. PCA decorrelation does not imply Gaussianity
or independence. The model samples without clipping, rejecting or rescaling
unusual fields, so distributional failures remain visible.

## What the comparisons mean

Each case exports three distinct ensembles:

| Ensemble | Definition | What a discrepancy diagnoses |
| --- | --- | --- |
| Direct | Held-out native LGCNN fields | Finite-sample comparison target |
| Reconstruction | Projection/reconstruction of those **same** held-out fields | KL truncation and stored-basis error |
| Gaussian | Fresh normal coefficient draws fitted only to training scores | Combined KL and Gaussian-law approximation error |

Held-out field RMSE and relative squared error assess compression. Area empirical
CDFs, means, SDs, q05/q50/q95 and Wasserstein distances assess decision-relevant
distributions at each source warming threshold (currently 0.1 and 1 °C).
The area of a Gaussian field's excursion set need not itself be Gaussian.

Native spatial probability maps show whether errors concentrate in particular
regions. Gaussian probabilities and expected area are calculated analytically;
fresh Gaussian fields estimate the full affected-area distribution. Native
float32 subtraction and **strict `T-background > threshold`** are reproduced in
all empirical reductions. Analytic probabilities account for native rounding
through the midpoint boundary between the last nonexceeding and first exceeding
representable temperature. Maps cover the original valid output crop, not the
larger permeability input domain. Raster figures may be resampled for display;
the exported arrays and numerical comparisons use every native output cell.

Normality screens use held-out, training-whitened coefficient coordinates,
pairwise sums/differences and fixed-seed random projections. They can expose
non-Gaussian joint dependence even when individual marginals look normal.
D'Agostino–Pearson p-values receive Holm correction within each case. There is
no across-case multiplicity claim, no universal Gaussianity proof, and no
automatic adequacy pass from non-rejection. Inspect risk errors against explicit
application tolerances. A below-background cell/draw fraction is reported as
a descriptive diagnostic; no physical lower bound is assumed or enforced.

The mean MCSE and binomial intervals condition on this fitted law and independent
draws. More Gaussian draws reduce integration error, **not** error in estimating
the law from 350 training fields. More native LGCNN samples, repeat independent
train/test splits, or a bootstrap that refits the basis/law are needed to study
fit stability. Do not tune rank or model family repeatedly on this held-out set
and then call the same set an independent final test. A later selection study
needs an additional test split or nested validation.

## Storage, runtime and provenance

The command never builds an `M × M` covariance matrix or saves fresh native
Gaussian ensembles. It streams spatial blocks, builds an `n × n` training Gram
matrix, writes `r × M` float32 modes, and reduces Gaussian fields in blocks.
For 1028² cells, rank 32 needs about **135 MB decimal for modes per case**;
means, variance, probability maps and figures add tens of MB. This is independent
of the number of generated Gaussian fields apart from their small coefficient
matrix. RAM scales approximately with `(n + n_Gaussian) × chunk_cells`, plus
native maps, modes accessed from disk and the small Gram matrix. Reduce
`--chunk-cells` or `--gaussian-draws` for less working memory. The source native
ensemble is memory-mapped. It is still read several times, and covariance
accumulation costs roughly `O(n² M)`; this is not a zero-cost fit.

Source manifest and consumed file hashes, sample identities, library versions,
seed, fit parameters and current Git commit are recorded. SHA256 verification
reads the consumed native ensemble once before fitting. Missing cases or corrupt
consumed files fail explicitly. Progress is printed for each major stage; a
failure leaves a manifest marked `failed`. Existing destinations are refused.

Important files in each case directory:

- `assessment.json`: diagnostics, errors, complete fit definition and source hashes.
- `model/{mean,modes,training_variance}.npy`, `model/model.json`: reloadable KL model.
- `latent_scores.npz`: all output coefficients, global sample indices, training
  mask and input fingerprints. **Held-out coefficients must not train an emulator.**
- `sample_links.json`: study/reference/case/sample/seed/input/output identities.
- `gaussian_law.npz`: training coefficient law and candidate coefficients/seed.
- `area_samples.npz`: direct/reconstructed/Gaussian affected areas in m².
- `probability_*.npy`: native threshold-by-row-by-column risk maps; the analytic
  Gaussian map has no candidate MC sampling error.

The root manifest and a case-by-threshold error heatmap provide a compact
comparison across selected cases.

## Initial native execution check (not a model-selection result)

The complete workflow was executed on the downloaded RUN_4 Gaussian / 256-reading
case: all 500 native 1028² outputs, 350/150 training/held-out split, rank capped
at 32, seed 20261110, and 1000 fresh Gaussian coefficient draws. The consumed
ensemble SHA256 was
`50b6695f9645492788f6c1d1a3c71b86ab0c1d9011763f3bda515c9bae38ed64`.
This fingerprint must be checked against the native assessment provenance before
reusing these numbers with another study.

The fit retained **65.83%** of training variance, explicitly failing the 99%
variance target. Held-out temperature RMSE was **0.0445 °C**, with relative
squared reconstruction error **0.435**. For warming above 0.1 °C the mean area was
7.902 km² in held-out fields, 8.182 km² after KL reconstruction, and 8.337 km² in
Gaussian draws (analytic Gaussian expectation 8.330 km²). For warming above
1 °C the corresponding means were 0.01223, 0.00493 and 0.00581 km².

The finite projection screen did not reject Gaussianity. These area discrepancies
and the unmet variance target still prevent claiming adequacy from that result.
The check establishes that the native-data workflow runs and exposes errors; it
does not select a rank, validate all kernels/counts, or establish decision
tolerances. Rank sensitivity on fresh validation data is still needed.

## Following step: reduced input-to-output mapping

This command deliberately exports the paired output side; it **does not learn
an input-to-output emulator yet**. A defensible next experiment needs the paired
input latent coordinates for every training and held-out sample, using the
recorded native input fingerprints. The compact study retains eight native inputs
per case; 96² previews and input histograms do not preserve all fine-scale input
information. Additional native inputs can be regenerated from the saved recipes
only under the original generator-source and numerical-runtime compatibility
guards (`subsurface-uq-regenerate-input`). Do not weaken those guards.

Then fit a regression `a_out = g(a_in, information_state)` on paired **training**
draws, validate predicted held-out temperatures and decision risks, and compare
its runtime/accuracy with direct frozen-LGCNN propagation. GP regression or PCE
are candidate emulators; neither requires or proves a Gaussian output field.
Treat emulator predictive uncertainty separately from input-induced uncertainty.
A joint model across configurations requires a common output basis, meaningful
input/information-state coordinates and entire held-out configurations. Twelve
case distributions alone do not identify a general distribution-to-distribution
mapping. None of these stages quantifies LGCNN-versus-physics discrepancy without
additional physical simulation/reference data.

Primary methodological references:
[GP definition and finite-dimensional Gaussian laws](https://gaussianprocess.org/gpml/chapters/RW2.pdf),
[snapshot KL/SVD](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.KarhunenLoeveSVDAlgorithm.html),
[Higdon et al. (2008), basis reduction with GP emulation](https://www2.stat.duke.edu/courses/Spring14/sta961.01/ref/HigdGattWillRigh2008.pdf).
