# Release25 + DaRUS runtime

The UQ package does not vendor the original LGCNN implementation, pretrained
weights, or scientific datasets. They stay external and are referenced by path.
PFLOTRAN is **not** required for inference.

## Required external assets

1. Heat-Plume-Prediction release25 code
   - upstream: https://github.com/JuliaPelzer/Heat-Plume-Prediction/tree/AllIn1/LGCNN/release25
   - this thesis has been checked against the equivalent `release25` branch in
     `TorbenAdelhelm/Heat-Plume-Prediction`.
2. Published synthetic-permeability pretrained models
   - DaRUS DOI: https://doi.org/10.18419/DARUS-5080
   - extract `LGCNN_step1_randomK.zip` and `LGCNN_step3_randomK.zip`.
3. A prepared `inputs_pki` dataset produced by Heat-Plume-Prediction
   preprocessing. Prepared datasets based on the same synthetic raw data are
   available from DaRUS (for example DOI `10.18419/DARUS-4467`). A prepared
   dataset directory must contain `info.yaml` and `Inputs/<RUN_ID>` tensors.

A convenient local layout is:

```text
external/
  Heat-Plume-Prediction/       # checkout at release25
models/
  LGCNN_step1_randomK/         # extracted archive
  LGCNN_step3_randomK/         # extracted archive
data/
  prepared_pki/
    info.yaml
    Inputs/
      RUN_0
      RUN_1
      ...
```

All three top-level directories above are local assets and should not be
committed.

## Deterministic smoke test

With no ensemble option the CLI propagates the original permeability of the
fixed prepared datapoint exactly once:

```bash
subsurface-uq-release25 \
  --release25-repo external/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_0 \
  --device cpu \
  --output run_output/release25_single.npz
```

Step 2 uses the original release25 `make_streamlines` implementation and its
SciPy ODE integration. On the full 100-heat-pump setting this is substantially
slower than the two CNN forward passes.

## Empirical Monte Carlo from prepared DaRUS permeability fields

Several prepared `pki` datapoints can be used as the initial empirical
permeability ensemble while pressure and heat-pump positions stay fixed to one
chosen datapoint:

```bash
subsurface-uq-release25 \
  --release25-repo external/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_0 \
  --permeability-run-ids RUN_0,RUN_1,RUN_2 \
  --device cuda \
  --output run_output/release25_empirical_mc.npz
```

Alternatively use `--ensemble-file` for an `[N,H,W]` permeability tensor/array
stored as `.npy`, `.npz`, `.pt`, or `.pth`.

## Model-loading contract

`load_standard_release25_model` reconstructs the published standard `model.pt`
from `info.yaml` and `HPS_options.yaml`. It follows release25's standard training
rule of taking the first value from each HPS `values` list. It deliberately does
not guess the architecture of an arbitrary Optuna trial checkpoint.

The runtime applies:

```text
physical p,k,i
  -> model-folder input normalization
  -> Step-1 CNN
  -> reverse velocity normalization
  -> original release25 Step-2 make_streamlines (center and +/-10-cell outer)
  -> Step-3 six-channel construction [i, vx, vy, s, k, s_outer]
  -> Step-3 input normalization
  -> Step-3 CNN
  -> reverse temperature normalization
  -> physical T
```

The Monte Carlo layer then computes streaming mean, variance, standard
deviation, minimum, and maximum without retaining all temperature fields unless
`--store-all` is requested.

## Validation status

Normal CI validates model reconstruction against a tiny release25-compatible
fixture and exercises the complete three-step adapter with a deterministic fake
Step-2 rasterizer. Large DaRUS weights/data are intentionally not downloaded in
CI. Therefore a local deterministic smoke run with the published assets remains
required before claiming numerical reproduction of the paper's pretrained
model output.
