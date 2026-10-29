# Controlled conditional-field benchmark

This is the main LGCNN input-UQ experiment. The RUN reconstruction challenge is
the separate realism test. The first establishes a correct, useful experimental
pipeline under an explicit geological law; the second tests whether a candidate
law represents withheld error in the available fields.

The primary laws and configurations are now the
[documented map-informed structural profiles](structural_profiles.md).
Their [completed 204-evaluation sensitivity pilot](structural_profile_results.md)
reports paired profile effects and the remaining MC limits.
The [32–512 observation spatial study](conditional_density_visual_results.md)
shows every input/output realization and its deviations for a single RUN 2
structural truth, with eight matched samples at each count.
The earlier [82-evaluation native pilot](controlled_field_results.md) and
its larger illustrative matrix are historical execution evidence.

## Probability model and information state

For each named scenario, declare

\[
Y(x)=\log_{10}(k(x)/[\mathrm{m}^2])=\mu+\sigma Z(x),\qquad
Z\sim\mathcal{GP}(0,\rho_\theta).
\]

Every experiment assumes **known** constant mu, sigma and correlation
parameters after its input law is declared. The current profiles use source-map
statistics before drawing any synthetic hidden truths or observing temperatures;
no hidden-truth statistics or future readings enter their fitting. These are
synthetic structural priors, not a posterior inferred from the Munich workbook.
The earlier illustrative pilot used mu=-9.3, sigma=0.25 and an
isotropic Matern-3/2 correlation with length parameter 1000 m. This gives median
permeability approximately 5.01e-10 m2 and a pointwise prior 90% range about
1.95e-10 to 1.29e-9 m2. These are transparent experimental scales within the
general range of the prepared real-K model inputs, not a training-support
certificate or a fitted geological range. The correlation uses
rho(r)=(1+sqrt(3)r)exp(-sqrt(3)r), with r the distance divided by 1000 m.

Each independent outer trial draws a hidden truth Y*. A frozen geometry-only
layout reveals d=Y*(B_n) at nested counts. For a fixed prefix, the conditional law is

\[
m_n=\mu+C_{xB}C_{BB}^{-1}(d-\mu),\qquad
C_n=C-C_{xB}C_{BB}^{-1}C_{Bx}.
\]

With an independent prior residual u, the Matheron realization is

\[
Y^{(s)}=\mu+u^{(s)}+C_{xB}C_{BB}^{-1}(d-\mu-u_B^{(s)}).
\]

At zero readings this is exactly the proper prior. The known-mean conditioner
accepts geometry and revealed readings only. It has no hidden-field argument.
Exact observations are checked after conversion to the model's float32
permeability. Numerical rank failure stops the run; there is no nugget, clipping,
empirical normalization, rejected draw, covariance replacement or truth selection.

The full covariance is represented by the existing GSTools-matched circulant
embedding, including its recorded covariance error tolerance. There is no KL
truncation. Native 2560x2560 fields at 5 m are passed directly to the LGCNN.
The same pressure, material/source inputs, checkpoints and streamline settings
apply to every truth and posterior realization. Output-grid receptors and risk
areas refer only to the valid cropped temperature domain.

## Independent trials, layouts and Monte Carlo

Truth and posterior streams are separate SHA256-derived semantic seeds. Each
outer trial has independent truth and posterior randomness. Within a trial,
common prior draws pair the different measurement counts and layouts. The
unconditioned LGCNN evaluations are shared within that trial, avoiding duplicate
queries. Balanced maximin and clustered layouts are generated without field
values, remain fixed across outer trials, and are reported separately. Layout
replicates and additional covariance scenarios can be declared explicitly.
They are not pooled into a geological probability mixture.

The historical illustrative **pilot** specifies 2 hidden truths, 8 posterior draws, counts
0/8/32, and one layout per coverage scenario: **82 LGCNN evaluations** in total,
including 2 truth evaluations. Its purpose is native execution and artifact QA.
With two truths, calibration intervals are necessarily very wide.

The historical larger illustrative **matrix** specifies 64 truths, 64 draws, counts 0/8/32/128 and
the same two coverage scenarios: **28,736 evaluations**, including 64 truths.
This is a prepared experiment specification, not a claim that it has run or
that 64 draws suffice for tail estimates. Use the pilot timings, storage needs,
half-batch differences and 16/32/64 MC checkpoints before allocating the full
matrix. The main matrix uses a fresh seed; it is not a selected continuation of
favourable pilot truths. A final convergence study can increase independent MC
budgets and truth counts using new configurations, without changing the law.

For nonlinear temperatures, realized conditional variance need not decrease
for every reading set. Report conditional variance estimates and their paired
changes across outer trials; a negative estimate or local increase is retained.
This is not temperature-driven observation placement or a validated RQ5 ranking.

## Calibration and useful outputs

1. **Independent dense marginal validation:** sample joint Gaussians at the
   native coordinates of observations and fixed withheld probes. Compare the
   implemented conditioner with independent dense covariance equations. Report
   analytical 90% coverage and finite-ensemble ranks across 256/512 independent
   truths per fixed probe/layout/count. This checks conditioning algebra and
   model-consistent calibration without loading the LGCNN. It does not establish
   full-grid FFT correctness or physical predictive accuracy; native FFT joint
   moments are also covered by tests.
2. **Native input checks:** preserve exact readings, export analytical posterior
   mean/SD maps, and report withheld error/spread on a declared grid. Calibrate
   fixed input-probe ranks across independent native hidden truths.
3. **Output UQ:** compute mean, SD, q05/q50/q95 and width maps, receptor
   temperatures, ROI mean/max warming, and areas exceeding declared warming
   thresholds. Save probability/severity maps, paired density comparisons,
   per-trial CRPS and calibration ranks. A generic model factory supports small
   independent tests; the scientific configurations use the actual frozen LGCNN.

For any fixed scalar functional f, Y* and the conditional draws are exchangeable
given d under the declared law. Applying the same deterministic LGCNN and QoI
preserves this property. Thus the truth rank among S draws is uniform on
0,...,S. Ties are randomized, which matters for float32 predictions, constant
temperatures and discrete threshold areas. The rank-window coverage target is
computed from its included integer ranks: with S=8 the central conservative
window has target 7/9, **not 90%**; with S=64 it is 57/65. Empirical q05/q95
maps are retained, but their finite-budget spatial truth coverage is descriptive.

Exact binomial intervals use independent hidden truths at a fixed scalar,
scenario, layout and count. Correlated pixels, layouts and nested counts are
never treated as independent calibration trials. Rank histograms and pointwise
intervals are diagnostics, not a simultaneous multiple-comparison test or an
automatic parameter-tuning rule. All truths remain in the results.

## Run and recover

Install the package with its geostatistical dependencies and prepare the existing
real-K model assets referenced by the model YAML. Paths in the study YAML are
resolved relative to that YAML; model asset paths use its declared asset root.

```powershell
python -m subsurface_uq.experiments.controlled_field_benchmark --config configs/controlled_field.profiles.pilot.yaml --output-dir run_output/controlled_field_profiles_pilot --stage all
```

The installed alias is `subsurface-uq-controlled-fields`. Separate stages are
`plan`, `validate`, `inputs`, `temperatures` and `report`. `plan` and `all` require
a fresh output directory. Resume the interrupted stage using the same config
and output; `temperatures` skips verified completed predictions. Run `report`
after propagation. Planning records the exact query budget without loading the
model. A model path is unnecessary for the dense validation if explicitly null.

```powershell
subsurface-uq-controlled-fields --config configs/controlled_field.profiles.main.yaml --output-dir run_output/controlled_field_profiles_main --stage plan
subsurface-uq-controlled-fields --config configs/controlled_field.profiles.main.yaml --output-dir run_output/controlled_field_profiles_main --stage validate
```

The source/configuration hashes, model/fixed-input hashes, seed contract,
geometry, exact readings and query input/output hashes are recorded. Complete
artifacts are checked before reuse. Interrupted geometry or field-moment maps
are rebuilt deterministically; interrupted model queries continue at the last
verified row. Resume through the same Python/package installation. A changed
configuration, execution source/installation, model, completed array or ledger
must use a fresh run rather than being silently accepted. Keep run directories
and manifests together.

Storage retains hidden truths, geometry features, analytical input maps,
temperatures and the first permeability realization per case. Other input
realizations are reproducible from their seeds and observation context; they
are not all retained as native arrays. Geometry scales as
8*rows*columns*max_readings bytes per layout (6.7 GB at 128 readings). The main
temperature ensembles alone require about 121 GB at 1028x1028 output; geometry,
input arrays, retained fields, temperature products and risk maps add roughly
75 GB. Check available disk space
and runtime before the main matrix. Values are decimal GB and exclude temporary
arrays and filesystem overhead.

## How the supplied Guthke paper informs this design

The supplied PDF is **Reiser, Aguilar, Guthke and Buerkner, “Uncertainty
Quantification and Propagation in Surrogate-based Bayesian Inference,”
arXiv:2312.05153v3 (5 February 2025)**. Its Table 1 and Sections 2.1 and 2.3 are
relevant: uncertainty sources must be distinguished, and calibration should be
checked by repeated simulation with known truths rather than by narrowness
alone. Its Section 2.3/Algorithm 1 motivates the repeated-truth rank diagnostics.

The paper's main task is Bayesian **inverse inference**, including propagation
of surrogate parameter/approximation uncertainty from a training step. The
current experiment is **forward input UQ through an already frozen LGCNN**.
It does not implement that two-step algorithm, retrain a Bayesian LGCNN, or
claim to propagate surrogate approximation error. Here the hidden output is
the same frozen LGCNN applied to a hidden input. Checking this output against
conditional predictions is a model-consistency check. LGCNN versus PFLOTRAN
comparisons remain a separate model-discrepancy experiment (RQ6).

## Realism test and later extensions

Keep [the completed RUN assessment](measurement_density_results.md) and
`configs/measurement_density.validation.yaml` unchanged as the empirical
reconstruction challenge. The current 15/54 adequacy passes do not support
unrestricted propagation from that plug-in sparse-reconstruction model.
Passing a controlled benchmark cannot convert that failed test into site
calibration. Revised geological models require fresh spatial validation data
or a clearly labelled exploratory reuse of the already inspected holdout.

After the known-law experiment, useful separate extensions are uncertain proper
trend/covariance priors (with truth generation and conditioning under the same
hierarchy), alternative declared covariance scenarios, noisy measurements with
matched conditioning, and paired physical-simulator comparisons. The workbook
still requires unit/CRS/support and duplicate-location decisions before it
justifies a real site posterior. These extensions must not be described as
already quantified in the first experiment.

References: [supplied paper](https://arxiv.org/abs/2312.05153),
[GSTools model/conditioning reference](https://gmd.copernicus.org/articles/15/3161/2022/).
