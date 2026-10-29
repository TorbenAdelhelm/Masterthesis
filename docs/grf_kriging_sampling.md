# Gaussian random field and kriging sampling

This module is the first Gaussian-random-field input model for the thesis UQ
pipeline. It deliberately stays at the permeability-input layer; it does not yet
add another release25 experiment CLI or a second PCE implementation.

## Prior field

The existing `KLLogGaussianPermeabilityMap` represents

\[
Y(x)=\log_{10}K(x)
=\mu_Y+\sum_{j=1}^{m}\sqrt{\lambda_j}\,\phi_j(x)\,\xi_j,
\qquad \xi_j\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1).
\]

The covariance is a separable product of one-dimensional Matérn-3/2
correlations. The implementation factorizes the two spatial axes, so it never
constructs the full `(H*W) x (H*W)` covariance matrix. The physical field is
`K=10**Y`; no clipping is applied.

The current covariance is therefore a **product/separable Matérn-3/2 model**, not
a radial isotropic two-dimensional Matérn covariance. This is a computational
choice that must remain explicit in the thesis.

## Conditioning / kriging

`ConditionalKLLogGaussianPermeabilityMap` conditions the retained KL prior on
sparse measurements at permeability-grid cells. Let

\[
Y=\mu+B\xi,\qquad \xi\sim\mathcal N(0,I_m),
\]

and let observations satisfy

\[
d=HY+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,R).
\]

With `A = H B`, conditioning is performed in KL-coordinate space:

\[
\mu_{\xi\mid d}
=A^T(AA^T+R)^{-1}(d-H\mu),
\]

\[
\Sigma_{\xi\mid d}
=I-A^T(AA^T+R)^{-1}A.
\]

The posterior covariance is eigendecomposed/factorized as

\[
\Sigma_{\xi\mid d}=LL^T,
\]

and the map exposed to the rest of the software is

\[
\eta\sim\mathcal N(0,I_r),\qquad
\xi=\mu_{\xi\mid d}+L\eta.
\]

Thus the stochastic coordinates remain independent standard Gaussians after
conditioning. This is important because the same map can be used directly by
`GaussianCoordinatePermeabilitySampler` for Monte Carlo and later by a Hermite
PCE design.

This formulation is the known-mean Gaussian/simple-kriging analogue in the
**truncated KL model**. It is not ordinary kriging with an unknown estimated
mean. With zero observation noise, the implementation rejects exact data that
cannot be represented by the retained KL subspace instead of silently replacing
them by a least-squares projection.

## Minimal usage

```python
import numpy as np

from subsurface_uq.sampling import (
    ConditionalKLLogGaussianPermeabilityMap,
    GaussianCoordinatePermeabilitySampler,
    KLLogGaussianPermeabilityMap,
)

prior = KLLogGaussianPermeabilityMap(
    shape=(256, 256),
    domain_size_m=(1280.0, 1280.0),
    mean_log10_k=-9.64,
    std_log10_k=0.35,
    length_scale_m=(250.0, 400.0),
    n_modes=20,
)

# Integer (row, col) cells of the permeability grid.
cells = np.array([[40, 70], [150, 110], [210, 190]])
measured_k = np.array([2.0e-10, 5.0e-10, 1.4e-10])

conditional = ConditionalKLLogGaussianPermeabilityMap.from_permeability_observations(
    prior=prior,
    observation_indices=cells,
    observation_k=measured_k,
    observation_std_log10_k=0.0,
)

sampler = GaussianCoordinatePermeabilitySampler(
    field_map=conditional,
    n_samples=100,
    batch_size=4,
    seed=2907,
)

for permeability_batch in sampler:
    # permeability_batch has shape [B,H,W] and can be passed to the existing
    # Monte Carlo propagation path.
    pass
```

The numerical values above illustrate the API only. `mean_log10_k`,
`std_log10_k`, correlation lengths, and KL dimension must be calibrated or
justified before they are used as thesis results.

## Current scope and next integration step

The implementation currently conditions observations at integer model-grid
cells. This keeps the first version explicit and avoids hiding an interpolation
choice. Borehole coordinates that do not coincide with grid cells should be
mapped or interpolated in a separately documented preprocessing step before
production use.

The next integration step should be a small GRF/conditional-GRF experiment that
feeds this map through the frozen LGCNN with bounded streamlines, first as Monte
Carlo. Only after that input-law compatibility check should the same Gaussian
coordinates be used for Hermite PCE training.
