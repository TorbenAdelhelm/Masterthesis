# Release25 synthetic Perlin forward-UQ baseline

This experiment propagates newly generated permeability realizations through the
published pretrained release25 LGCNN while keeping pressure and heat-pump
locations fixed from one prepared `pki` run. It is an **unconditional synthetic
input-UQ baseline**, not yet borehole-conditioned geostatistical uncertainty.

For the mathematical definitions of the sampler, Monte Carlo estimators,
streaming statistics and risk maps, see [`methodology.md`](methodology.md).

## Historical provenance

The sampler follows the historical `perlin_v2` permeability branch from
`JuliaPelzer/Dataset-generation-with-Pflotran`:

```text
commit: 8549bbd9e22d2bc75ce2038c1a0397359e45c971
path:   scripts/create_varying_field.py
```

The released DaRUS configuration for `dataset_giant_100hp_varyK` records:

- grid: `2560 x 2560 x 1` cells;
- domain: `12800 x 12800 x 5 m`;
- permeability case: `perlin_v2`;
- Perlin frequency: `[18, 18]`;
- permeability bounds: `1.0193679918450561e-11` to
  `5.09683995922528e-09 m^2`;
- fixed pressure gradient: `-0.003`;
- `vary_perm: true`, `vary_pressure: false`, `only_vary_distribution: true`.

The corresponding PFLOTRAN input fixes the initial groundwater temperature at
`10.6 °C` and injection temperature at `15.6 °C`. Therefore the default
temperature-change statistic is

$$
\Delta T(x)=T(x)-10.6\ ^\circ\mathrm C,
$$

not `T-10.0`. The thesis task description's approximately `10 °C` value is a
general example; `10.6 °C` is the exact value for this historical synthetic
setup.

## Perlin permeability transformation

If `P(x)` is the raw two-dimensional `noise.pnoise2` realization, define

$$
U(x)=\frac{P(x)-\min P}{\max P-\min P}.
$$

The historical generator maps this field linearly in base-10 logarithmic
permeability space,

$$
\ell(x)=\log_{10} k_{\min}
+U(x)\left(\log_{10} k_{\max}-\log_{10} k_{\min}\right),
$$

and returns

$$
K(x)=10^{\ell(x)}.
$$

The original generator draws one random three-component base offset and shifts
successive samples by integer values in the first coordinate. Its historical
`np.random.seed(...)` line was commented out. Consequently, `seed_id: 2907` in
the released settings is not sufficient to claim bitwise reconstruction of the
original three training fields. The thesis sampler intentionally makes `seed`
effective for *new* UQ runs while preserving the historical field formula.

## Run

From the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

A five-realization smoke experiment is:

```bash
python -m subsurface_uq.experiments.release25_perlin \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --n-samples 5 \
  --seed 2907 \
  --device cpu \
  --output run_output/release25_perlin_mc_5.npz \
  --plots-dir run_output/release25_perlin_mc_5_plots
```

`--cnn2-dir` is a retained legacy name: the directory is the second CNN of the
LGCNN, i.e. **Step 3 / CNN3**.

The default exceedance thresholds are `0.1 °C` and `1.0 °C` relative to the
`10.6 °C` background. They can be changed with
`--exceedance-thresholds`, and the background can be overridden for another
dataset with `--background-temperature`.

## Streaming statistics

For `N` propagated temperature realizations, the main spatial estimators are

$$
\hat\mu_T(x)=\frac1N\sum_{m=1}^{N}T^{(m)}(x)
$$

and, for the normal Monte Carlo choice `ddof=1`,

$$
\hat\sigma_T^2(x)=\frac1{N-1}
\sum_{m=1}^{N}\left(T^{(m)}(x)-\hat\mu_T(x)\right)^2.
$$

Mean, variance, standard deviation, minimum, maximum and exceedance counts are
accumulated online. `--store-all` is therefore unnecessary unless individual
realizations are needed for another analysis. A one-sample deterministic
diagnostic automatically uses `ddof=0`; `N=1, ddof=1` is treated as undefined.

## UQ plots

A new Perlin run with `--plots-dir` creates:

- mean temperature;
- temperature standard deviation;
- sample range `max-min`;
- mean temperature with standard-deviation contours;
- mean `Delta T` relative to the configured background;
- one empirical exceedance-probability map per configured threshold.

The empirical exceedance estimator is

$$
\widehat P_\tau(x)
=\frac1N\sum_{m=1}^{N}
\mathbf 1[T^{(m)}(x)-T_{bg}\ge\tau].
$$

Existing archives can be plotted without rerunning the LGCNN:

```bash
python -m subsurface_uq.visualization.cli \
  --input run_output/release25_perlin_mc_5.npz \
  --output-dir run_output/release25_perlin_mc_5_plots
```

Older archives created before the background was recorded can still be plotted
with an explicit `--background-temperature` value. For the historical synthetic
dataset use `10.6`.

## Reproducibility metadata

Every new run writes a versioned NPZ archive and a neighboring
`*.metadata.json`. In addition to the sampler parameters and realized sample
count, release25 experiment metadata records where available:

- effective and requested `ddof`;
- fixed run id and local asset paths;
- Masterthesis and release25 Git SHAs;
- SHA-256 hashes of CNN1 and CNN3 checkpoints;
- Python/platform and selected dependency versions;
- historical generator repository, path and commit;
- published model DOI `10.18419/DARUS-5080`.

The result schema is explicitly versioned so later changes to the archive layout
can be distinguished programmatically.

## Interpretation and next step

The present experiment estimates

$$
K^{(m)}\sim P_{\mathrm{Perlin}},\qquad
T^{(m)}=F(K^{(m)};p_0,i_0).
$$

It answers how the pretrained deterministic LGCNN responds to the historical
synthetic permeability variability while pressure and heat-pump locations are
fixed. It must not be described as uncertainty conditioned on borehole data.
The future conditional geostatistical sampler will replace only the
`PermeabilitySampler`; propagation, accumulators, statistics and visualizations
are intended to remain unchanged.
