# Stochastic-coordinate and KL permeability infrastructure

This document specifies the finite-dimensional stochastic permeability map that
is implemented independently of the release25 LGCNN experiments. It is the
intended common input representation for Monte Carlo propagation and later
non-intrusive Polynomial Chaos Expansion (PCE).

## 1. Separation of stochastic coordinates and permeability sampling

The central object is an explicit map

$$
G_m:\mathbb R^m\to\mathbb R^{H\times W},
\qquad
\boldsymbol\xi\mapsto K.
$$

`StochasticPermeabilityMap` represents this contract. It deliberately does not
choose how the coordinates are sampled. This is important because the same map
must support, for example,

- independent Gaussian Monte Carlo coordinates;
- a future PCE regression design;
- a future conditional Gaussian coordinate transformation.

`GaussianCoordinatePermeabilitySampler` is only an adapter for the existing
`PermeabilitySampler` interface. It draws

$$
\boldsymbol\xi\sim\mathcal N(0,I_m)
$$

and then evaluates the configured `StochasticPermeabilityMap`. Consequently,
`MonteCarloRunner` remains unchanged and continues to receive only physical
permeability batches.

## 2. Log-Gaussian random field

The initial implementation models base-10 log-permeability

$$
Y(\mathbf x)=\log_{10}K(\mathbf x)
$$

as a Gaussian random field. The finite-dimensional approximation is

$$
Y_m(\mathbf x,\boldsymbol\xi)
=
\mu_Y
+
\sum_{r=1}^{m}
\sqrt{\lambda_r}\,\phi_r(\mathbf x)\xi_r,
\qquad
\xi_r\overset{\mathrm{iid}}{\sim}\mathcal N(0,1),
$$

followed by

$$
K_m(\mathbf x,\boldsymbol\xi)=10^{Y_m(\mathbf x,\boldsymbol\xi)}.
$$

No clipping is applied by the map. Permeability-range compatibility with the
pretrained LGCNN must therefore be checked explicitly when scientific GRF
parameters are selected.

## 3. Separable Matérn-3/2 covariance

A direct covariance matrix on an `H x W` grid would contain `(H*W)^2` entries
and is infeasible at the target spatial resolutions. The implemented baseline
therefore uses the separable covariance

$$
C((y,x),(y',x'))
=
\sigma_Y^2
\rho_{3/2}\!\left(\frac{|y-y'|}{\ell_y}\right)
\rho_{3/2}\!\left(\frac{|x-x'|}{\ell_x}\right),
$$

where

$$
\rho_{3/2}(r)
=
(1+\sqrt{3}r)\exp(-\sqrt{3}r).
$$

On the discrete grid this gives

$$
C_{2D}=\sigma_Y^2(C_y\otimes C_x).
$$

The current implementation uses array-aligned conventions

```text
shape          = (H, W)
domain_size_m  = (L_y, L_x)
length_scale_m = (ell_y, ell_x)
```

with covariance locations at cell centers.

The separable product covariance is a deliberate computational baseline. It is
not identical to an isotropic radial 2-D Matérn covariance. A more general
covariance model can be introduced later if the geostatistical calibration
requires it.

## 4. Factorized discrete KL expansion

Let

$$
C_y u_i = \lambda_i^y u_i,
\qquad
C_x v_j = \lambda_j^x v_j.
$$

The 2-D discrete covariance eigenpairs are then

$$
\lambda_{ij}
=
\sigma_Y^2\lambda_i^y\lambda_j^x,
$$

and

$$
\phi_{ij}=u_i\otimes v_j.
$$

Therefore the implementation diagonalizes only the `H x H` and `W x W`
one-dimensional correlation matrices. It never constructs the full
`(H*W) x (H*W)` covariance matrix and it does not store all 2-D eigenfunctions.
The selected 2-D modes are represented by their one-dimensional eigenvector
indices.

This is a discrete KL representation of the permeability values at cell
centers. Its purpose is to reproduce the requested discrete covariance while
providing independent Gaussian coordinates for the UQ layer.

## 5. KL truncation

The selected 2-D eigenvalues are sorted in decreasing order. Two truncation
modes are supported.

With fixed dimension `m`, the leading `m` modes are retained directly.
Alternatively, for requested energy fraction `eta`, the smallest `m` is chosen
such that

$$
\frac{\sum_{r=1}^{m}\lambda_r}
{\sum_{r=1}^{HW}\lambda_r}
\ge \eta.
$$

The default implementation parameter is `eta=0.95`, but this is only a software
default. The final scientific stochastic dimension must be selected through a
combination of KL-energy diagnostics and downstream LGCNN/QoI convergence.

## 6. Relationship to PCE

The eventual non-intrusive PCE will use the same independent coordinates

$$
\boldsymbol\xi=(\xi_1,\ldots,\xi_m),
\qquad
\xi_j\sim\mathcal N(0,1),
$$

as polynomial inputs. In particular, the PCE training path is intended to be

$$
\boldsymbol\xi^{(n)}
\xrightarrow{G_m}
K^{(n)}
\xrightarrow{F_{\mathrm{LGCNN}}}
T^{(n)}
\xrightarrow{q}
Q^{(n)}.
$$

The Gaussian-coordinate Monte Carlo adapter is not used to define that PCE
experimental design; the PCE layer will retain the coordinates explicitly and
call `G_m` directly.

## 7. Validation implemented in this stage

The KL infrastructure is tested without the release25 models or DaRUS assets.
For small grids, the tests construct all KL modes and verify that their basis
reconstructs

$$
\sigma_Y^2(C_y\otimes C_x)
$$

to numerical precision. Additional tests check

- symmetry and unit diagonal of the Matérn correlation matrices;
- zero-coordinate mapping to `10**mean_log10_k`;
- energy-threshold mode selection;
- single and batched coordinate shapes;
- strictly positive permeability output;
- deterministic repeatability of seeded Gaussian coordinate sampling;
- compatibility of the Gaussian adapter with the existing
  `PermeabilitySampler` protocol.

This stage intentionally does **not** validate whether any particular choice of
`mean_log10_k`, `std_log10_k` or Matérn length scale is scientifically
appropriate for the pretrained LGCNN. Parameter calibration and
input-distribution compatibility are separate next steps.

## 8. Deferred extensions

The following are intentionally outside this implementation stage:

- calibration of `mu_Y`, `sigma_Y`, `ell_y`, and `ell_x`;
- borehole conditioning and conditional KL coordinates;
- connection to a release25 experiment CLI;
- PCE basis construction and regression;
- POD/PCE field reduction;
- component-wise CNN1/streamline/CNN3 PCE models.
