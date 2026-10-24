# Conditional Gaussian fields from synthetic reference observations

This workflow draws measurement locations from a fixed permeability reference,
reads the reference values at those cells, and generates Gaussian log-fields
conditioned on that frozen observation design. Its purpose is to compare explicit
priors for where measurements are drawn and the resulting field uncertainty.
RUN_1, RUN_2 and RUN_3 remain separate reference scenarios.

The readings are **synthetic pseudo-observations**, not the real Munich borehole
records. Coordinates are local array row/column cell centers in the reference's
existing orientation. This avoids claiming a geographical alignment between the
real workbook and the RUNs, which remains unverified.

## Prior law and conditional simulation

For one fixed reference, let

\[
Y(x)=m(x)+u(x),\qquad u\sim\mathcal N(0,C_\theta),\qquad K(x)=10^{Y(x)}.
\]

`mean_mode: reference_median` sets `m=log10(K_ref)` and keeps the reference's
orientation and deterministic spatial structure. `global_log_mean` instead uses
the reference's spatial mean log-permeability as a constant prior mean; it is an
explicit sparse-reconstruction comparison, not the default law.

The radial Gaussian, Matérn-3/2 and exponential covariance families reuse the
matched-kernel FFT sampler. Correlation lengths and amplitude are supplied in
the configuration, optionally separately for each reference. Reference texture
can guide those choices. Neither reference pixels nor successful fitting recover
the original map's kriging uncertainty: its observation set, noise and upstream
interpolation provenance are not known. `sigma_R` is a model-form/sensitivity
parameter, distinct from synthetic measurement noise.

`angle_deg` explicitly rotates the covariance anisotropy relative to the local
array-column/row axes; it does not rotate or resample the reference array. Zero
retains the fitted study's grid-aligned convention. Rotation is a declared
sensitivity choice unless independently estimated; the default reference mean
already preserves the reference's structural orientation.

For selected locations D and readings d, define observation-noise covariance N.
The fixed-design conditional law is

\[
m_c=m+C_{xD}(C_{DD}+N)^{-1}(d-m_D),\qquad
C_c=C-C_{xD}(C_{DD}+N)^{-1}C_{Dx}.
\]

Sampling uses a full-covariance Matheron update of the native FFT draw:

\[
Y_c=m+u+C_{xD}(C_{DD}+N)^{-1}(d-m_D-u_D-\varepsilon_u),
\quad\varepsilon_u\sim\mathcal N(0,N).
\]

The independent noise realization in this update is required for the correct
noisy posterior covariance. The full field covariance is not materialized;
cross-covariances are processed in chunks. There is no KL or random-frequency
truncation in this path. The numerical FFT embedding checks remain in force.
The method is consistent with the conditional circulant-embedding construction
of [Dietrich and Newsam (1996)](https://doi.org/10.1029/94WR02977).

Observation systems are solved through eigenvector whitening, without adding a
hidden nugget. Designs that lose numerical rank at a relative threshold of
`1e-12` fail explicitly before FFT allocation. Increase spacing, use fewer
observations, or declare a scientifically justified positive noise level rather
than silently dropping exact constraints. Exact-data samples also check the
maximum observation residual against `max(sigma_R * 1e-7, 1e-10)` log10 units;
failure stops the computation without replacing the draw. These are numerical
solver checks, not geological acceptance thresholds.

With `measurement_mode: exact`, readings equal reference log-values and the
observation standard deviation must be zero. With `noisy`, a positive
`observation_std_log10_k` is required and independent Gaussian noise is added to
each reference reading before conditioning. Location, observation-noise and
conditional-field random streams are separately seeded and recorded.

With the default reference mean and exact reference readings, `d-m_D=0`.
Consequently **conditioning retains the reference log-mean and reduces the
covariance**. It does not demonstrate recovering a hidden reference from sparse
information. Use the explicit constant-mean comparison for that separate question.
Individual realizations are never rescaled to force the reference's spatial
histogram, clipped, rejected, or replaced according to their diagnostics.

## Engineering a location prior

The YAML `location_priors` mapping gives each strategy a user-chosen name and
`kind`. Each measurement is selected probabilistically from the remaining
candidate cells. Candidate stride, margin and minimum physical spacing constrain
locations only; generated fields retain the native grid.

- `uniform`: a uniform baseline on admissible cells, modified by an optional
  user-supplied spatial weight map.
- `gradient`: preferential sampling of reference log-gradient structure.
- `variance`: sequential sampling guided by the remaining kriging variance from
  the same covariance and already selected observations.
- `hybrid`: combines reference-gradient structure with remaining variance.

`variance_power`, `gradient_power` and `exploration_fraction` control preference
strength and exploration. Optional nonnegative `location_weights` NPY arrays
express accessibility or a spatial design prior; zero-weight cells are excluded.
All of these are assumptions about sampling design, not inferred probabilities
for reference RUNs or recovered historical well locations. For a stationary prior
with uniform base weights, variance-only selection starts uniformly because its
initial pointwise variance is constant. Subsequent draws can favor less-informed
regions. The `measurements.csv` records the selected probability at each step.

Gradient/hybrid strategies inspect the complete fixed reference when defining
the design prior. They are therefore idealized reference-informed design
experiments. Applying the same rule at a genuinely unknown site would require
an independently available structural map or other pre-measurement information;
the synthetic experiment alone does not establish operational design skill.

## Repeated designs and uncertainty decomposition

Each location/reading set is frozen before drawing its conditional ensemble. For
that fixed design the log-field is Gaussian. Across randomly drawn designs the
aggregate is generally a **mixture of Gaussian laws**, and it must not be treated
as one Gaussian posterior or automatically passed to one Hermite PCE model.

Within each reference and location-prior assumption the study reports

\[
\operatorname{Var}(Y)=
\mathbb E_{\mathcal D}[\operatorname{Var}(Y\mid\mathcal D)]
+\operatorname{Var}_{\mathcal D}(\mathbb E[Y\mid\mathcal D]).
\]

Here the empirical design mixture uses explicitly equal weights over independently
drawn design replicates. Its within-design term averages analytic conditional
variances; its between-design term is the population variance of conditional
means across those replicates. In noisy mode that second term combines location
and simulated-reading variation. In reference-median exact mode it is zero,
although each design can have a different uncertainty map. The variances are
pointwise log-space quantities; spatial averages are summaries of those maps.
They are not the variance of a pooled spatial histogram. RUNs and priors are not
pooled. Few designs do not establish convergence of the design-prior expectation.

MC error is reported separately through empirical fixed-design field means,
variances and standard errors. More conditional draws improve numerical precision;
they do not reduce the conditional geological uncertainty.

## Configuration and execution in Git Bash

From the repository root, with the environment used for the earlier study:

```bash
python -m pip install -e ".[geostat]"
export PYTHONPATH=src
python -m subsurface_uq.experiments.reference_conditioning \
  --config configs/reference_conditioning.example.yaml \
  --output-dir run_output/reference_conditioning_001
```

The installed entry point is `subsurface-uq-reference-conditioning`. Reference
and weight paths are resolved relative to the YAML file. Unknown keys and
inconsistent noise/centering parameters fail explicitly. References must be
positive finite 2-D `.npy` intrinsic-permeability arrays in m². Use a documented
conductivity-to-permeability conversion before supplying m/s observations.
The example retains the 5 m grid and uses explicitly assumed lengths. Replace
them with selected per-reference matched-fit lengths through
`covariance_by_reference` when conducting that sensitivity experiment.

Every output directory is a new frozen study; existing directories are refused.
The manifest records configuration and reference hashes, implementation commit,
runtime versions, sampler accuracy and seed streams. Reference and design IDs are
deterministic hashes. Reordering the configuration does not change unchanged
cases' name-derived random streams.

## Outputs and interpretation

Each `RUN/prior/design` directory contains:

- `measurements.csv`: row/column positions, local cell centers, reference and
  observed log-values, noise SD, and sequential selection probabilities;
- `design.json`: design identity, seeds, analytic posterior statistics, MC error,
  native reference-agreement/gradient features, patch descriptors and field hashes;
- `posterior_log10_mean.npy`, `posterior_log10_variance.npy` and
  `posterior_log10_std.npy`: native posterior
  maps, without a claim of recovered real-world kriging uncertainty;
- `conditional_fields.png`: reference, conditional log-mean, first realization,
  its log-ratio, conditional uncertainty and observation locations;
- selected `sample-*.npy` fields and fixed probe samples;
- `reference_diagnostics.json`: distributions, directional variograms and
  correlations, radial/angular/2-D spectra and reference-relative differences on
  a declared matched diagnostic grid. Native gradients remain separately reported.

Fixed corner/center patches describe reference-relative support. Their identity
does not establish checkpoint training membership. Reference diagnostics have no
pass/fail cutoff and do not certify LGCNN training support or temperature accuracy.
All draws contribute, including fields not retained as full arrays. Storage
selection is by declared sample index, never diagnostic score.

Each prior's `summary.json` and variance maps expose the within/between-design
decomposition. The root `README.md`, `summary.json` and `manifest.json` keep the
reference/prior hierarchy. Retained fields can be used as explicit inputs to the
existing frozen LGCNN workflow; this command does not itself query the network
or establish a temperature-based generator recommendation.

Each reference's `reference_descriptors.json` supplies the matching nominal
native-gradient and patch descriptors for comparing the per-sample records.
