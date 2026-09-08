# Release25 + DaRUS runtime

The UQ package does not vendor the original LGCNN implementation, pretrained
weights, or scientific datasets. They remain external and are referenced by
path. PFLOTRAN is **not** required for surrogate inference or forward Monte
Carlo UQ.

For the mathematical composition of the runtime and the UQ estimators, see
[`methodology.md`](methodology.md).

## Required external assets

1. Heat-Plume-Prediction release25 code
   - upstream: `JuliaPelzer/Heat-Plume-Prediction`, branch
     `AllIn1/LGCNN/release25`;
   - the thesis integration has also been checked against the equivalent
     `release25` branch in `TorbenAdelhelm/Heat-Plume-Prediction`.
2. Published synthetic-permeability pretrained models
   - DaRUS DOI: `10.18419/DARUS-5080`;
   - extract `LGCNN_step1_randomK.zip` and `LGCNN_step3_randomK.zip`.
3. A prepared `inputs_pki` dataset produced by Heat-Plume-Prediction
   preprocessing
   - the directory must contain `info.yaml` and `Inputs/<RUN_ID>` tensors;
   - prepared datasets are available through the LGCNN DaRUS releases. The
     exact local prepared dataset provenance should be recorded separately
     rather than inferred from its directory name.

A convenient local layout is:

```text
external/
  release25-repo/
    Heat-Plume-Prediction/
models/
  LGCNN_step1_randomK/
  LGCNN_step3_randomK/
data/
  prepared_pki/
    info.yaml
    Inputs/
      RUN_1.pt
      ...
```

These are local scientific assets and are intentionally ignored by Git.

## Deterministic runtime contract

For one physical permeability field `K`, fixed pressure `p` and fixed heat-pump
material field `i`, the adapter executes

$$
\mathbf v = F_1(p,K,i),
$$

$$
\mathbf s = S(i,\mathbf v),
$$

$$
T = F_3(K,i,\mathbf v,\mathbf s).
$$

Operationally this is

```text
physical p,k,i
  -> CNN1-folder input normalization
  -> Step-1 CNN
  -> reverse velocity normalization
  -> original release25 make_streamlines
       * central
       * +10-cell offset
       * -10-cell offset
  -> Step-3 raw channels [i, vx, vy, s, k, s_outer]
  -> CNN3-folder input normalization
  -> Step-3 CNN
  -> reverse temperature normalization
  -> physical T
```

The outer feature is the sum of the `+10` and `-10` faded streamline rasters,
matching the release25 preprocessing path.

The runtime reconstructs standard released model checkpoints from each model
folder's `model.pt`, `info.yaml` and `HPS_options.yaml`. It follows the release25
standard-training convention of taking the first value from HPS `values` lists.
It deliberately does not guess arbitrary Optuna-trial architectures.

The current Python API and CLI retain the name `cnn2` / `--cnn2-dir` for
backwards compatibility. This path actually denotes the **second CNN of the
LGCNN, which is Step 3 / CNN3**.

## Normalization

The adapter reproduces the released `NormalizeTransform` rules. For ordinary
rescaling from physical interval `[a,b]` to `[0,1]`,

$$
z=\frac{x-a}{b-a}.
$$

For standardization,

$$
z=\frac{x-\mu}{\sigma}.
$$

The inverse transformations are applied to network outputs before velocities or
temperatures are passed to physics/UQ components.

## Deterministic smoke test

With no ensemble option the empirical CLI propagates the original permeability
of the fixed prepared datapoint exactly once. The CLI automatically uses
`ddof=0` for this one-sample diagnostic because a sample variance with
`N=1, ddof=1` is undefined.

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --device cpu \
  --output run_output/release25_single.npz
```

Step 2 uses the original release25 SciPy ODE integration and is normally the
slowest part of a full-domain 100-heat-pump inference.

## Empirical Monte Carlo

Several prepared permeability fields can form an empirical input ensemble while
pressure and heat-pump locations remain fixed:

```bash
python -m subsurface_uq.experiments.release25_empirical \
  --release25-repo external/release25-repo/Heat-Plume-Prediction \
  --cnn1-dir models/LGCNN_step1_randomK \
  --cnn2-dir models/LGCNN_step3_randomK \
  --prepared-pki-dir data/prepared_pki \
  --fixed-run-id RUN_1 \
  --permeability-run-ids RUN_1,RUN_2,RUN_4 \
  --device cpu \
  --output run_output/release25_empirical_mc.npz
```

Alternatively `--ensemble-file` accepts an `[N,H,W]` permeability ensemble from
`.npy`, `.npz`, `.pt`, or `.pth`.

This empirical path is useful as a pipeline baseline, but using a small number
of existing fields is not a substitute for a scientifically specified
conditional random-field model.

## Temperature-reference validation

The validation CLI loads a prepared temperature label, reverses its stored
normalization, and compares it against one physical model prediction. Because
`UNetNoPad2` reduces the spatial dimensions, the default alignment center-crops
the larger reference to the model output.

```bash
python -m subsurface_uq.validation.cli \
  --prediction run_output/release25_single.npz \
  --reference-dir data/prepared_pki_temperature \
  --run-id RUN_1 \
  --output run_output/release25_RUN_1_validation.npz \
  --plots-dir run_output/release25_RUN_1_plots
```

Reported quantities include MAE, MSE, RMSE, maximum absolute error, mean bias,
absolute-error percentiles and spatial fractions above configured diagnostic
levels used by the current implementation.

The local development setup has successfully executed the real DaRUS pretrained
assets and a physical RUN_1 reference comparison. This establishes practical
integration with those assets. It is still distinct from the stronger claim of
bitwise/numerical equivalence to an independently executed original release25
pipeline; that stronger equivalence should only be claimed after a direct
original-runtime comparison.

## Validation layers

It is useful to distinguish three levels:

1. **CI/unit integration**: lightweight fixtures check normalization, model
   reconstruction contracts, propagation, statistics and plots without
   downloading large DaRUS assets.
2. **Real-asset smoke/reference validation**: local released models and prepared
   data are executed and compared to available reference temperature labels.
3. **Original-runtime equivalence**: the same input is independently executed
   through the original release25 workflow and the resulting intermediate/final
   fields are compared numerically.

The repository currently covers levels 1 and 2. Level 3 remains a separate,
stronger reproducibility test.

## Reproducibility metadata

New release25 experiment results record a versioned result schema and, when the
local paths permit it:

- Masterthesis Git SHA;
- external release25 Git SHA;
- SHA-256 digests of CNN1 and CNN3 checkpoints;
- Python/platform and selected package versions;
- fixed run id, model/data paths, streamline method and `ddof`;
- published model DOI.

These records are intended to make later thesis figures traceable to the exact
code and model artifacts that produced them.
