# Source-alignment notes

The thesis task description, the base paper, released model metadata and the
historical data-generation code are not identical descriptions of every detail.
This file records the discrepancies that matter for the implementation so they
are not silently reconciled.

## Step-3/CNN3 inputs

The thesis task description presents the conceptual final stage as a CNN using
`p, k, i, v, s`. The base paper's formal Step-3 equation omits pressure and
writes the temperature CNN as a function of `k, i, v, s`.

The executable random-permeability release25 model is more specific: its
prepared Step-3 input contains six channels

```text
[i, vx, vy, s, k, s_outer]
```

and no pressure channel. The current runtime follows the released model metadata
and code because those define the pretrained checkpoint's executable contract.
This should be described in the thesis as an implementation-level specialization
of the conceptual task diagram rather than as evidence that the task description
is wrong.

## Background temperature

The task description proposes threshold maps relative to a background of
approximately `10 °C`. The historical synthetic PFLOTRAN input used for the
random-permeability dataset specifies

```text
initial temperature   = 10.6 °C
injection temperature = 15.6 °C
```

which is consistent with the base paper's stated `5 °C` injection-temperature
difference. Consequently, release25 synthetic UQ uses `T_bg = 10.6 °C` by
default, while keeping the background configurable for other datasets.

## Perlin seed semantics

The released settings contain `seed_id: 2907`, but the historical
`np.random.seed(settings["general"]["seed_id"])` statement in
`create_varying_field.py` is commented out. Therefore the settings file alone
does not justify a claim that the original three Perlin fields can be regenerated
bitwise from seed 2907.

The thesis sampler uses seed 2907 as an explicit default for reproducible *new*
Monte Carlo experiments and records the generated base offset in every run.

## Streamline ODE method

Descriptions of the synthetic experiments may refer to a particular stiff ODE
method, while the executable release25 helper ultimately delegates to SciPy
`solve_ivp` and the released runtime path defaults to `RK45` unless another
method is explicitly supplied. The thesis wrapper exposes `RK45`, `RK23`, and
`Radau` and records the selected method in run metadata.

For reproducibility claims about a specific experiment, the recorded runtime
argument should be used rather than inferring the method from a general paper
description.

## Validation levels

A successful run through the thesis adapter with the published checkpoints is
not the same claim as exact numerical equivalence to an independently executed
original release25 workflow. The repository therefore distinguishes:

1. lightweight CI contract tests;
2. real-asset execution/reference validation;
3. direct original-runtime numerical equivalence.

The first two have been exercised. The third remains a stronger validation step.
