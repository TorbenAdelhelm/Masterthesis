# Perlin stochastic-coordinate model

This document specifies the stochastic-coordinate interpretation used for the
first PCE-oriented input experiment. The purpose is to stay close to the
historical permeability generator used for the released synthetic LGCNN data
while exposing explicit low-dimensional random variables.

## Historical generator

The reference implementation is

```text
JuliaPelzer/Dataset-generation-with-Pflotran
scripts/create_varying_field.py
commit 8549bbd9e22d2bc75ce2038c1a0397359e45c971
```

For `case == "perlin_v2"`, the dataset generator draws once

```python
base_offset = np.random.rand(3) * 4242
```

and generates field `base` with

```python
offset = base_offset + [base, 0, 0]
```

For a **2-D** simulation, the actual Perlin call uses only `offset[0]` and
`offset[1]`; the third component is only relevant to the 3-D branch. Therefore
the continuous random starting location affecting a 2-D permeability field is
two-dimensional.

The original finite dataset did not independently redraw this offset for each
field. All fields generated in one call shared one random `base_offset`, while
successive integer `base` values shifted the x coordinate. The historical
NumPy seeding call was commented out, so the exact original random offset cannot
be reconstructed from the recorded seed alone.

## Stochastic-coordinate law

For the PCE proof-of-concept we define independent standardized coordinates

$$
\xi_x,\xi_y\stackrel{\mathrm{iid}}{\sim}\mathcal U(-1,1).
$$

Each coordinate is mapped affinely to a Perlin starting offset. With the
release25-compatible defaults,

$$
o_x = 2121(\xi_x+1),\qquad
o_y = 2121(\xi_y+1),
$$

so that the random offsets span `[0, 4242]` (the upper endpoint has probability
zero under the continuous law). An optional deterministic x shift `b` reproduces
the historical role of one chosen integer `base`:

$$
o_x^{(b)} = o_x + b.
$$

The resulting map is

$$
G_{\mathrm{Perlin}}:\ [-1,1]^2\to\mathbb R_+^{H\times W},
\qquad
(\xi_x,\xi_y)\mapsto K.
$$

The physical field uses exactly the existing `historical_perlin_v2_field`
transformation: evaluate `noise.pnoise2`, perform field-wise min-max scaling,
map the result affinely into `[log10(k_min), log10(k_max)]`, and exponentiate
with base 10.

## Interpretation

This is an **iid continuous stochastic law derived from the same Perlin
generator family**, not a claim that the exact joint distribution of the finite
historical training set has been reconstructed. The difference is deliberate:
classical non-intrusive PCE requires explicit random coordinates, while the
historical dataset used one shared random offset followed by deterministic
integer translations.

This distinction should remain explicit in thesis text and experiment metadata.
The Perlin coordinate experiment is primarily a method-validation baseline: it
keeps the spatial generator and permeability transformation close to the LGCNN
training regime while giving the UQ layer a low-dimensional coordinate vector.

## PCE implication

Because the standardized coordinates are independent uniform variables, the
natural orthogonal polynomial family is Legendre. The first PCE experiment can
therefore use

$$
\Psi_{\boldsymbol\alpha}(\boldsymbol\xi)
=
\prod_{j=1}^2 P_{\alpha_j}(\xi_j)
$$

with suitable orthonormal scaling. The stochastic dimension is only two, so
full total-degree bases of moderate order remain small:

$$
P=\binom{m+p}{p}=\binom{2+p}{p}.
$$

For example, degrees 2, 3, 4, 5, and 6 require 6, 10, 15, 21, and 28 basis
terms respectively.

The field-wise min/max normalization and the Perlin lattice structure can still
reduce smoothness of the map from coordinates to downstream LGCNN QoIs. PCE
convergence must therefore be measured empirically rather than assumed from the
low dimension alone.

## Software mapping

The implementation separates the probability design from the physical field map:

```text
UniformCoordinatePermeabilitySampler
        -> xi ~ U(-1,1)^2
        -> PerlinCoordinatePermeabilityMap
        -> historical_perlin_v2_field
        -> K
```

For PCE training the sampler adapter is optional: a deterministic or randomized
experimental design can pass its own coordinate matrix directly to
`PerlinCoordinatePermeabilityMap.map_coordinates`. This preserves the same
`xi -> K` map for Monte Carlo and PCE.

No release25 LGCNN experiment CLI is connected by this change. That integration
belongs to the next roadmap step, after the coordinate map has passed its
independent unit tests.
