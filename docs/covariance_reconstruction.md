# Covariance-only measurement placement and reconstruction

See the [executed study and count tables](covariance_reconstruction_results.md).

This benchmark addresses sparse reconstruction, rather than perturbations around
a completely supplied reference mean. Each RUN is treated separately. The
reference is a synthetic truth from which selected exact readings can be revealed;
it is not an observed geological realization with identified ensemble parameters.

## Information used

- **Placement:** geometry, declared covariance and previously selected locations
  only. No reference values, mean, gradient, output or reference-error metric.
- **Conditioning:** only the reference log-values at the selected sites. An
  unknown constant mean is estimated from those readings by ordinary kriging.
- **Evaluation:** the remaining reference values are withheld truth for error
  scoring. They do not modify locations, generated draws or covariance parameters.
- **Prior information:** the existing Gaussian 40–2560 m texture-fit lengths are
  transferred to a reconstruction covariance. This uses reference-derived
  structural information and is an explicit modelling assumption. It does not
  recover the original interpolation uncertainty or historical boreholes.

The placement API has no reference-array argument. The kriging API receives
indices and selected readings, with no access to unobserved reference values.

## Location draws

The first site is uniform on the declared candidate lattice. Subsequent draws
have probability proportional to remaining **ordinary-kriging variance**:

\[
 p_n(b)=\frac{v_n(b)}{\sum_{c\in\mathcal C_n}v_n(c)}.
\]

An unknown mean contributes to this variance. Under exact measurements, covariance
amplitude cancels from these normalized probabilities. Changing the supplied
reference values or its full mean cannot affect locations. The saved CSV records
each site's order, probability, preselection variance, native-array indices and
revealed log-value. Seeded draws and nested prefixes make the experiment repeatable.

Incremental Cholesky updates avoid repeatedly fitting the same kriging system.
The explicit normalized innovation-variance floor excludes numerically redundant
candidate sites. If all candidates become redundant, the sequence stops and
reports that numerical limit. It adds no nugget, observation error or field filter.

## Ordinary conditional field

Let \(Y=\log_{10}(K/\mathrm{m}^2)\), with unknown constant mean \(\beta\), and
covariance \(\sigma^2R\). For selected sites \(B\), readings \(d\), and unit
correlation matrix \(R_{BB}\), define

\[
 a=\mathbf1^TR_{BB}^{-1}\mathbf1,\qquad
 \hat\beta=a^{-1}\mathbf1^TR_{BB}^{-1}d,\qquad
 r(x)=1-R_{xB}R_{BB}^{-1}\mathbf1.
\]

The ordinary-kriging mean and conditional covariance, with a flat prior on the
unknown mean, are

\[
 m_B(x)=R_{xB}R_{BB}^{-1}d+r(x)\hat\beta,
\]
\[
 C_B(x,x')=\sigma^2\left[R_{xx'}-R_{xB}R_{BB}^{-1}R_{Bx'}
                         +\frac{r(x)r(x')}{a}\right].
\]

The final term retains uncertainty in the estimated mean. Merely inserting an
estimated mean into simple kriging would omit this contribution. Independent
unit Gaussian FFT prior draws are conditioned using ordinary-kriging weights
\(W\): \(Y^{(s)}=Wd+\sigma[u^{(s)}-Wu_B^{(s)}]\). This has the stated covariance
and reproduces exact readings. Only afterwards is \(K^{(s)}=10^{Y^{(s)}}\) formed.
Zero observations do not define a proper posterior with this flat unknown-mean
prior; no zero-observation field is generated or centered using the full reference.

See [Rasmussen and Williams, equations 2.41–2.42](https://gaussianprocess.org/gpml/chapters/RW2.pdf)
and [GSTools ordinary kriging](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.krige.Ordinary.html).

## Stochastic errors and counts

For a fixed layout, two different random quantities are reported:

\[
 E_n^{(s)}=\sqrt{\frac1M\sum_x[Y_n^{(s)}(x)-Y_{\rm ref}(x)]^2},\qquad
 U_n^{(s)}=\sqrt{\frac1M\sum_x[Y_n^{(s)}(x)-m_n(x)]^2}.
\]

- \(q_{0.95}(E_n)\) is a **reference-relative reconstruction error**. Its stopping
  count is an oracle benchmark, since withheld truth is needed to calculate it.
- \(q_{0.95}(U_n)\) measures **spread around the kriging prediction**. It depends
  on covariance, geometry and amplitude, so it can guide an uncertainty-only
  stopping rule without inspecting unobserved reference values. Small spread
  alone does not establish a correct reconstruction under a misspecified covariance.

For each supplied error limit \(\epsilon\), the table gives the first **tested**
count with estimated \(q_{0.95}\le\epsilon\). It does not interpolate to a claimed
minimum, assume monotone reference error, or pool layouts/RUNs. Null means the
criterion was not reached before the numerical or measurement budget ended.

The code also reports order-statistic 95% Monte Carlo intervals, the first tested
crossing of the interval's upper end, and exact binomial intervals for
\(P(E_n\le\epsilon)\). These intervals are pointwise, not simultaneous guarantees
across all counts, scenarios and selected stopping rules. Half/full ensemble
quantiles help expose finite-sample sensitivity; independent repetitions remain
necessary for a reliable operational count recommendation.

An additional conservative bound follows without estimated quantiles:

\[
 q_p(E_n)\le\sqrt{\frac{\mathbb E[E_n^2]}{1-p}},\qquad
 \mathbb E[E_n^2]=\frac1M\sum_x[(m_n-Y_{\rm ref})^2+C_n(x,x)].
\]

This is Markov's inequality. Replacing the numerator by average posterior variance
gives the analogous covariance-only bound for \(U_n\). Counts meeting these
conservative bounds are recorded separately from empirical quantile crossings.
All errors use log10 permeability: 0.05 log units is an RMS measure, **not** a
pointwise physical-permeability tolerance or a temperature limit.

## Executed-study configuration and scope

[`covariance_reconstruction.study.yaml`](../configs/covariance_reconstruction.study.yaml)
declares three separate Gaussian covariance structures, two location sequences,
128 conditional draws, and total log-field SD assumptions 0.1, 0.2 and 0.3.
These amplitudes are sensitivities, not fitted uncertainty or the old residual
amplitude. Shared Gaussian draws pair amplitude and count comparisons; neither
conditional samples nor layout replicates are silently pooled.

The full 12.8 km reference domain is point sampled every eighth native pixel:
320×320 locations, 40 m spacing. Candidate locations are every fourth evaluation
cell (160 m). This is not block averaging, native-grid LGCNN propagation or a
guarantee between retained cells. Generated and reference values at evaluation
cells correspond to the native input lattice. Resolution must be checked before
interpreting measurement counts as recommendations for 5 m fields.

The tested budget is 1/2/4/8/16/24/32/48/64/96/128/192/256/384/512/700;
the last numerically feasible count is additionally evaluated. Placement is
frozen before readings or stopping metrics are used. The study evaluates every
requested feasible prefix, with no sampling until a favorable field occurs.

## Git Bash execution and outputs

From the repository root in an environment with the project installed:

```bash
python -m subsurface_uq.experiments.covariance_reconstruction \
  --config configs/covariance_reconstruction.study.yaml \
  --output run_output/covariance_reconstruction_20261025
```

Use a fresh output directory on repetition; existing results are never overwritten.
Set `PYTHONPATH=src` if using the source tree before package installation.

For each RUN and design:

- `locations.png`: placement progression, colored by selection order.
- `fields.png`: reference, a fixed conditional draw, its reference difference and
  posterior SD at increasing measurement counts; shared scales across counts.
- `error_curves.png`: reference-error and conditional-spread quantiles, Monte Carlo
  intervals and kriging-mean bias.
- `measurements.csv`, `design.json`: frozen selection sequence and assumptions.
- `n-NNNN/summary.json`: scalar diagnostics for each amplitude.
- `n-NNNN/fields_sigma-*.npy`: two fixed **log10** conditional realizations;
  `permeability_sigma-*.npy`: their positive physical counterparts in m².
- `n-NNNN/errors_sigma-*.npz`: all draws' RMS and spatial maximum errors.
- `stopping_counts.json/.csv`: first-tested count tables, separately by RUN/layout.

The manifest records input, configuration, executed source and output hashes,
seeds and numerical backend checks. Large data stay local outside Git. This study
queries no LGCNN; subsequent temperature comparisons require native model inputs.

Tests compare ordinary means/variances against GSTools, check ensemble covariance
including mean uncertainty, exact constraints, amplitude-independent locations,
unknown-mean handling, numerical stopping and end-to-end artifact creation.
