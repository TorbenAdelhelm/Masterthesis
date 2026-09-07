# Release25 synthetic Perlin forward-UQ baseline

This experiment propagates newly generated permeability realizations through the
published, pretrained release25 LGCNN while keeping pressure and heat-pump
locations fixed from one prepared `pki` run.

## Provenance

The sampler follows the historical `perlin_v2` permeability branch from
`JuliaPelzer/Dataset-generation-with-Pflotran`,
`scripts/create_varying_field.py`. The released DaRUS configuration for
`dataset_giant_100hp_varyK` records:

- grid: `2560 x 2560 x 1` cells;
- domain: `12800 x 12800 x 5 m`;
- permeability case: `perlin_v2`;
- Perlin frequency: `[18, 18]`;
- permeability bounds: `1.0193679918450561e-11` to
  `5.09683995922528e-09 m^2`;
- fixed pressure gradient: `-0.003`;
- `vary_perm: true`, `vary_pressure: false`, `only_vary_distribution: true`.

The historical algorithm evaluates 2-D `noise.pnoise2`, min-max normalizes each
realization, maps it linearly into `log10(k_min)..log10(k_max)`, then applies
`10**x`. Successive samples use the same random base offset and shift the x
coordinate by integer `base` values.

The released settings also record `seed_id: 2907`, but the corresponding
historical `np.random.seed(...)` line was commented out. Therefore the original
three training fields are not assumed to be bitwise reproducible from the
settings alone. The thesis sampler intentionally makes the seed effective so
new Monte Carlo experiments are reproducible while retaining the historical
spatial-generation formula.

## Run

From the repository root, after installing the package:

```bash
python -m pip install -e ".[test]"
```

run, for example, a 20-realization CPU experiment:

```bash
python -m subsurface_uq.experiments.release25_perlin \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --n-samples 20 \
  --seed 2907 \
  --device cpu \
  --output run_output/release25_perlin_mc.npz
```

The installed console-script equivalent is `subsurface-uq-release25-perlin`.

## Reproducibility metadata

Every Perlin-UQ run writes both:

- the requested NPZ result, containing `count`, `mean`, `variance`, `std`,
  `minimum`, `maximum`, optional stored samples, and `metadata_json`;
- a neighboring `*.metadata.json` sidecar.

The metadata includes at least `seed`, `frequency`, `k_min`, `k_max`, realized
`sample_count`, shape, domain size, base offset, fixed run id, release25 paths,
streamline solver setting, and provenance notes.

## Interpretation

This is a synthetic prior/input-distribution UQ baseline, not yet
borehole-conditioned geostatistical uncertainty. Later, the Perlin sampler can
be replaced by a conditional GRF/kriging sampler while retaining the same
surrogate, propagation, statistics and QoI interfaces.
