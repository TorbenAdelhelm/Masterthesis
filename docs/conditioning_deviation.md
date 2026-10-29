# Permeability deviation versus measurement count

This input-only study uses all three original 2560 × 2560, 5 m reference arrays.
The [completed native study](conditioning_deviation_results.md) contains the
audited numerical results and an overview figure for all three RUNs.
It compares two quantities for each fixed location/reading design and count n:

\[
D_{ref,s}=\sqrt{M^{-1}\sum_x[\log_{10}K_s(x)-\log_{10}K_{ref}(x)]^2},
\quad
D_{pair,s,t}=\sqrt{M^{-1}\sum_x[\log_{10}K_s(x)-\log_{10}K_t(x)]^2}.
\]

The script also reports RMS relative reference error, `100*sqrt(mean((K/Kref-1)^2))`,
and absolute RMSE in m². Metrics use every native cell. Only PNG displays are
subsampled; generated fields and saved arrays are never coarsened.

For posterior log-mean m and marginal variance v, the exact expectations are

\[
E[D_{ref}^2]=M^{-1}\sum_x\{v(x)+(m(x)-\log_{10}K_{ref}(x))^2\},
\qquad E[D_{pair}^2]=2M^{-1}\sum_x v(x).
\]

The plotted RMS expectation is the square root of these expected **squared**
distances, not E[D]. With the reference as the prior log-mean and exact reference
readings, m stays the reference and pair RMS is sqrt(2) times reference RMS in
expectation. Removing nested measurements increases this analytic uncertainty;
one individual realization need not move monotonically farther from the reference.
Physical relative-error expectations use the exact pointwise lognormal moments.

## Executable native configuration

`configs/conditioning_deviation.native.yaml` uses counts `[0,1,4,16,32]`, 32 field
draws per setting, sample checkpoints 8/16/32 and two independent designs per
reference. Zero is the same unconditioned prior. The separate Gaussian length
fits over 40–2560 m, `sigma_R=0.05`, median reference mean and exact readings are
kept fixed. These remain declared model-form assumptions, not calibrated truth.
Variance-guided locations use the same 500 m spacing and 160 m candidate stride
as the earlier pilot. Smaller sets are frozen prefixes of the maximum design.

From the repository root in Git Bash:

```bash
python -m pip install -e ".[geostat]"
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m subsurface_uq.experiments.conditioning_deviation \
  --config configs/conditioning_deviation.native.yaml \
  --output-dir run_output/conditioning_deviation_native_001
```

This evaluates **960 conditional fields in 30 settings**, using 192 independent
full FFT draws reused across measurement counts. The original covariance is
preserved: cached float64 priors and blocked Matheron corrections only reorganize
computation. Independent likelihood noise uses the existing scalar sampler's
seed mapping. Tests compare the complete blocked fields and analytic moments
against that sampler for Gaussian, Matérn and exponential covariances, exact/noisy
readings and an empty observation set. No KL approximation is introduced.

Temporary full-resolution prior caches require about 1.7 GB per active design
and are removed after its budgets complete. Final maps and two retained fields
per setting require several GB. The script refuses existing output directories.
Its manifest records source/configuration/reference and output hashes, numerical
FFT metadata, seeds and interpretation. This initial command does not resume.

## Inspecting the results

- `RUN.policy.fields.png`: reference plus two paired generated fields per count,
  with shared color limits. Columns run from more measurements to fewer.
- `RUN.policy.deviations.png`: draw A minus reference, draw B minus reference,
  A minus B, and exact posterior SD. Differences share a common log10 color scale;
  SD panels share `[0,sigma_R]`. These examples use the first frozen design.
- `RUN.policy.curves.png`: native-grid reference error, within-setting pair spread,
  and relative reference error. MC checkpoints are compared to exact expectations,
  averaging squared metrics over the two separately conditioned designs.
- Each `RUN/policy/design-NNN/n-NNNN` stores two native field arrays, their native
  log differences, analytic/empirical SD maps, empirical mean log-bias, a 32 × 32
  pairwise distance matrix, per-sample metrics and summaries. All 32 draws contribute.
  With 32 draws there are 496 unordered pairs, which are **dependent**, not 496
  independent realizations. No confidence interval treats them as independent.

No sample is clipped, rejected, replaced or selected by a deviation score. The
reference is already known in the prior mean, so increasing uncertainty is not
a demonstration of reconstructing a hidden field from fewer readings. Exact
expectations validate the declared Gaussian law, not the geological realism of
its covariance/amplitude. RUNs and location policies remain separate.

This command does not query the LGCNN or establish temperature accuracy or
training-support membership. Its retained full native permeability arrays can
be used for a subsequent frozen-LGCNN evaluation.
