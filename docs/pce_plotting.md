# PCE plotting

The current PCE plotting support is intentionally small and run-oriented. It does not add a sweep manager or a plotting framework; it only visualizes one saved PCE result archive in a reproducible way.

## Generate plots during a PCE run

Add `--plots-dir` to the existing release25 Perlin PCE command:

```bash
python -m subsurface_uq.experiments.release25_perlin_pce \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --degree 3 \
  --n-train 30 \
  --n-validation 50 \
  --train-seed 2907 \
  --validation-seed 3907 \
  --device cpu \
  --streamline-mode bounded \
  --streamline-max-nfev 100000 \
  --output run_output/pce/perlin_p3_n30_seed2907.npz \
  --plots-dir run_output/pce/perlin_p3_n30_seed2907_plots
```

The plot directory receives exactly three PNG files:

- validation parity: expensive LGCNN QoI versus PCE prediction;
- validation residuals: `PCE - LGCNN` in validation-sample order;
- fitted PCE coefficients: one bar for every retained total-degree basis term.

The input `.npz` remains the scientific result; plots can always be regenerated from it.

## Plot an existing archive

```bash
python -m subsurface_uq.visualization.pce_cli \
  --input run_output/pce/perlin_p3_n30_seed2907.npz \
  --output-dir run_output/pce/perlin_p3_n30_seed2907_plots
```

The installed equivalent is `subsurface-uq-plot-pce`.

An optional `--prefix` only changes plot filenames. No scientific data or metrics are recomputed.

## Scope

This implementation deliberately stops at single-run diagnostics. Cross-run degree/training-budget convergence plots should be added only after the first sweep has produced a stable archive naming/aggregation convention. This keeps the current proof-of-concept small and avoids introducing a premature experiment-management layer.
