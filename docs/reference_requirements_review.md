# Reference-family requirements review

Reviewed implementation commit `6268f51c6fe789f4ef585101fbbcbb0f6704a76b`
against the original requested architecture on 6 October 2026. Review corrections
cover execution/replay, diagnostic provenance, numerical stability and examples;
the scientific input law is unchanged.

## Findings and corrections

1. **RQ1 artifact integration and replay.** A minimal artifact-only matrix template
   failed manual-GRF validation before the matrix selected its input model. Saved
   configs also repeated both the artifact and manual keys, so replay was rejected.
   The matrix and temperature CLI now pass their selected artifact before validating
   the fixed-model/QoI template. Direct input-model configs still reject manual
   duplicates. Serialized configs emit only the applicable stochastic-input block.
2. **Training-support provenance.** The nominal reference extrema were written as
   `training_reference`, and RQ1 could substitute synthetic Perlin bounds when the
   nominal range collapsed. The artifact now separates the nominal reference range
   from an explicitly sourced training profile. No profile means unassessed training
   support. Nominal fidelity diagnostics use `reference_*` labels. Actual training
   profiles retain their source checksums and descriptor comparisons.
3. **Finite-input numerical conversions.** Forming variance/mean squared could
   underflow or overflow; computing unnecessary physical moments could fail even
   when requested quantile bounds were representable. Conversions now work in log
   space and reject unrepresentable requested outputs clearly. Supplied consistent
   sigma_R is preserved, retaining existing expert-scenario artifact IDs.
4. **Examples and matrix duplication.** Added a dedicated real-K/artifact RQ1
   template while keeping the legacy synthetic/manual example. Numerically equal
   matrix configurations are rejected even if one uses integer literals and the
   other float literals; existing scenario IDs remain unchanged.

## Requirement-by-requirement assessment

| Requirement | Assessment and evidence |
|---|---|
| 1. Scenario/config and inconsistent inputs | Implemented in `sampling/scenarios.py`: run, lognormal marginal, physical moments or equal-tail bounds, amplitude, covariance, directional lengths and exclusive truncation. Expert scalars require a declared pointwise reference anchor. Inconsistent marginal/centering/amplitude/truncation inputs are rejected. Amplitude-only mode also remains available. |
| 2. Multiplicative reference law and centers | Implemented in `sampling/reference_field.py`. Median centering and arithmetic-mean correction use the retained pointwise Gaussian variance. Different RUNs remain distinct references. |
| 3. Bounds are quantiles | Equal-tail probability is `1-alpha` for the nominal untruncated anchor law. No bounded/truncated-lognormal mode, clipping or support-based sample selection is implemented. |
| 4. Conversion utilities and tests | Bidirectional moment and quantile conversions are tested at ordinary permeability scales and extreme finite inputs, with explicit numerical representability errors. |
| 5. Experiment matrix and manifests | Cartesian run/amplitude/kernel/length/truncation axes; duplicate checks; stable configuration IDs, source/artifact checksums, geometry and seeds. Optional RQ1 execution now uses replayable artifact-only configs. |
| 6. Separate scenarios and explicit weights | Results stay per scenario. Optional equal/custom weights are explicitly declared assumptions; no posterior pooling or weighted-result aggregator is implemented. |
| 7. Diagnostics without filtering | Marginal range/quantiles, directional variograms, spectra and optional training-patch descriptors. Nominal fidelity and training support are distinct. Descriptor-envelope membership is descriptive, not proof of in-distribution behavior. |
| 8. QoI-oriented KL sensitivity | Paired MC/RQMC temperature study compares receptor and ROI QoIs using common coordinates. Means/std/quantiles and paired errors accompany 0.95/0.99/0.999 energy sensitivities. Largest tested dimension is a comparison reference, not truth. |
| 9. Gaussian MC/RQMC/Hermite contract | Existing iid Gaussian target coordinates are retained. Scrambled Sobol vectors are dependent integration points under that target measure; Hermite response PCE uses the same field map. |
| 10. Architecture promotion with legacy reproducibility | Primary family name and documentation are updated. Historical schema-3 names and physical maps still load. Legacy measurement-kriging commands remain; no residual amplitude/kernel is claimed calibrated. Historical unsupported training-range metadata no longer produces a false support claim. |
| 11. CLI/docs/Git Bash | Matrix, direct generation, temperature-KL and real-K RQ1 examples use Git Bash continuation syntax and quoted paths. |
| 12. Tests and CI | New regressions cover RQ1 template loading/replay, training-profile provenance, support labels, finite-input conversion extremes and matrix duplication. Full Linux CI runs Python 3.11 and 3.12. Exact validation results are reported with the published correction commit. |

## Explicit remaining scope and assumptions

- Expert scalar moments/bounds constrain the untruncated law at an explicit anchor,
  not a heterogeneous field histogram. Truncation changes local dispersion; this
  is recorded rather than silently rescaled.
- Only the existing separable Matérn-3/2 and exponential KL kernels are exposed.
  Residual amplitude, lengths, family, centering and optional weights are assumptions.
- The paired KL study supports unconditioned reference artifacts. Conditioned
  truncations need rebuilt updates and a justified coupling before extending it.
- The matrix invokes the complete legacy RQ1 design, including Perlin comparisons
  and duplicate B/C evaluations for an unconditioned input. These duplicates are
  not evidence of a conditioning effect. Use the dedicated KL temperature CLI for
  that focused representation comparison.
- Sampled support descriptors are descriptive and depend on explicitly selecting
  the real pretrained model's training split and matching its grid/conventions.
  Compatibility and temperature convergence still require scientific experiments;
  this review does not establish either with actual pretrained-model outputs.
- Config IDs preserve existing representations for backwards compatibility; file
  checksums and grid metadata complete their scientific identity.

For the numerically stable inverse-normal operation used by the conversion
utilities, see the [SciPy 1.10 documentation](https://docs.scipy.org/doc/scipy-1.10.0/reference/generated/scipy.special.ndtri_exp.html);
it is available within the existing minimum SciPy dependency.
