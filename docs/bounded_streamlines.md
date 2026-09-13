# Bounded release25 streamline mode

## Motivation

The published release25 Step-2 implementation integrates every heat-pump streamline with
`scipy.integrate.solve_ivp` and constructs its `RegularGridInterpolator` objects with
`bounds_error=False, fill_value=None`. Consequently, a trajectory that leaves the valid
spatial grid can continue to be integrated using extrapolated velocities. The original code
removes out-of-domain trajectory points only after the ODE solve has finished.

During the Perlin PCE smoke test this behavior produced a reproducible pathological RK45
trajectory: the first three LGCNN evaluations completed normally, while a later realization
spent an excessive amount of time inside `solve_ivp -> RK45 -> RegularGridInterpolator`.

## Design

The external Heat-Plume-Prediction checkout is not modified. The Masterthesis package adds a
local drop-in `make_streamlines` replacement and selects it only when explicitly requested.

Two modes are available:

- `release25`: exact existing adapter behavior and the default for backwards compatibility;
- `bounded`: release25-compatible integration with terminal domain-boundary events and a
  per-streamline RHS-evaluation watchdog.

The bounded mode deliberately preserves the other Step-2 choices: velocity interpolation,
axis convention, heat-pump starting points, center/+10/-10 passes, grid resolution factor,
`solve_ivp` method, `t_end=27.5`, `t_steps=10000`, and faded streamline rasterization.

The numerical difference is that a trajectory terminates once it exits
`[0,H-1] x [0,W-1]`, rather than continuing through extrapolated velocities and discarding
those points afterwards.

## Recommended smoke test

```bash
python -m subsurface_uq.experiments.release25_perlin_pce \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --degree 1 \
  --n-train 4 \
  --n-validation 3 \
  --batch-size 1 \
  --train-seed 2907 \
  --validation-seed 2908 \
  --device cpu \
  --streamline-mode bounded \
  --streamline-max-nfev 100000 \
  --streamline-diagnostics \
  --output run_output/release25_perlin_pce_bounded_smoke.npz
```

`--streamline-max-nfev 0` disables the watchdog. The default is `100000` RHS evaluations per
individual streamline. `--streamline-slow-seconds` controls the threshold for printing a
slow-trajectory diagnostic and defaults to 2 seconds.

The same options are available in `subsurface_uq.experiments.release25_perlin` for the Monte
Carlo baseline.

## Reproducibility and thesis use

The selected mode, solver, watchdog budget, and diagnostics setting are written to experiment
metadata. Reference comparisons against the published implementation should use
`--streamline-mode release25`. UQ production runs may use `bounded` after checking that
ordinary, non-pathological cases yield negligible differences in streamline rasters and final
temperature fields.

A watchdog failure raises `StreamlineIntegrationError` with the sample number, streamline
pass, heat-pump index, start point, last integration time, and last state. It does not silently
replace or truncate a failed result.
