# Bounded vs release25 streamline equivalence experiment

## Purpose

The bounded streamline mode fixes a reproducible pathological `solve_ivp` trajectory by
terminating trajectories at the valid grid boundary and preventing extrapolated velocity
evaluations during adaptive Runge-Kutta boundary-crossing stages. Before using bounded mode
for production UQ, its numerical effect should be measured on ordinary cases rather than
assumed negligible.

This experiment performs a paired comparison on exactly the same permeability realization:

```text
xi -> Perlin K -> CNN1 -> release25 Step 2 -> CNN3 -> T_release25
                  |\
                  | `-> bounded Step 2   -> CNN3 -> T_bounded
                  |
                  `-> identical CNN1 velocity field
```

The external Heat-Plume-Prediction checkout is not modified.

## Default comparison set

The default command reconstructs the same four-point Latin-hypercube training design used in
the seed-2907 PCE smoke test and compares only one-based indices `1 2 3`. These three samples
were observed to be non-pathological under the original release25 solver. Index 4 is excluded
by default because it is the known release25 pathological case and would defeat the purpose of
a small equivalence check by potentially hanging the reference calculation.

The selected stochastic coordinates and mapped Perlin offsets are saved in the result archive,
so the comparison is fully reproducible.

## Run

From the repository root:

```bash
python -m subsurface_uq.experiments.release25_streamline_equivalence \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --design-seed 2907 \
  --design-size 4 \
  --sample-indices 1 2 3 \
  --device cpu \
  --streamline-method RK45 \
  --streamline-max-nfev 100000 \
  --streamline-diagnostics \
  --output run_output/release25_streamline_equivalence.npz
```

The equivalent installed command is
`subsurface-uq-release25-streamline-equivalence`.

## Quantities compared

For each paired realization the experiment records differences for:

- CNN1 physical velocity field, as a sanity check; this should be numerically identical because
  Step 2 has not yet been applied;
- center streamline raster;
- combined outer (`+10` and `-10`) streamline raster;
- final CNN3 physical temperature field.

For each quantity it records maximum absolute error, mean absolute error, RMSE, and relative
L2 error. It also records wall-clock time for the complete release25 and bounded LGCNN
predictions.

The result archive contains only coordinates, offsets, timings, and comparison metrics, not the
full 2560x2560 fields, so the experiment remains lightweight in storage. Reproducibility and
model provenance are written to the JSON sidecar.

## Interpretation

No numerical tolerance is hard-coded. The purpose of the experiment is to measure the change
first. A thesis-level justification should report the observed streamline-raster and final
physical-temperature differences and decide whether they are negligible relative to the
surrogate/UQ scales of interest.

A useful outcome would be near-zero CNN1 velocity differences, small streamline-raster
differences confined to domain-exit behavior, and correspondingly negligible temperature
changes. If the final temperature differences are material, bounded mode must be treated as a
modified numerical model rather than a transparent robustness fix.
