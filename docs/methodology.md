# Mathematical and software methodology

This document describes the mathematical objects implemented by the current
forward-UQ baseline and maps them to the package interfaces. It is intended to
stay close to the executable code; future methods should be added here when they
become part of the implementation.

## 1. Deterministic LGCNN map

For the present input-UQ experiments, pressure `p` and heat-pump locations `i`
are fixed and permeability `K` is uncertain. The published release25 LGCNN is
used as a deterministic map

$$
T = F(K; p,i).
$$

Its three stages follow the LGCNN decomposition from the base paper:

$$
\mathbf v = F_1(p,K,i),
$$

$$
\mathbf s = S(i,\mathbf v),
$$

$$
T = F_3(K,i,\mathbf v,\mathbf s).
$$

`F_1` and `F_3` are pretrained CNNs and `S` is the original non-learned
release25 streamline routine. In the implementation, the Step-3 input channels
are ordered as

```text
[i, vx, vy, s, k, s_outer]
```

where `s_outer` is the sum of the two faded streamline rasters generated from
starting positions shifted by `+10` and `-10` cells. The wrapper reverse-
normalizes CNN1 velocities before the streamline solver and reverse-normalizes
the final CNN3 temperature to physical units.

The command-line argument `--cnn2-dir` is retained for backwards compatibility,
but it points to the *second CNN in the LGCNN*, i.e. Step 3 / CNN3.

Reference implementation:
`JuliaPelzer/Heat-Plume-Prediction`, branch `AllIn1/LGCNN/release25`.

## 2. Random permeability and forward uncertainty propagation

Let the permeability field be a random field `K(x)`. A Monte Carlo experiment
generates or loads realizations

$$
K^{(1)},\ldots,K^{(N)}
$$

and propagates each realization through the deterministic LGCNN:

$$
T^{(m)}(x) = F\!\left(K^{(m)};p,i\right),\qquad m=1,\ldots,N.
$$

The software separation is therefore

```text
PermeabilitySampler
        -> TemperatureSurrogate
        -> MonteCarloRunner
        -> TemperatureAccumulator(s)
```

`PermeabilitySampler` controls the input probability model. `TemperatureSurrogate`
only maps a permeability field to a temperature field. `MonteCarloRunner` does
not need to know whether a realization came from an empirical ensemble, Perlin
noise, or a future borehole-conditioned Gaussian random field.

## 3. Monte Carlo field statistics

For every spatial cell `x`, the implemented sample mean is

$$
\hat\mu_T(x)=\frac{1}{N}\sum_{m=1}^{N}T^{(m)}(x).
$$

The variance with degrees-of-freedom parameter `d` (`ddof`) is

$$
\hat\sigma_T^2(x)
=\frac{1}{N-d}\sum_{m=1}^{N}
\left(T^{(m)}(x)-\hat\mu_T(x)\right)^2,
$$

and

$$
\hat\sigma_T(x)=\sqrt{\hat\sigma_T^2(x)}.
$$

The default statistical choice for an actual Monte Carlo ensemble is `ddof=1`.
For a single deterministic smoke-test realization, the experiment CLIs use
`ddof=0`; a sample variance with `N=1, ddof=1` is rejected as undefined rather
than being reported as zero.

The sample extrema and range are

$$
T_{\min}(x)=\min_m T^{(m)}(x),\qquad
T_{\max}(x)=\max_m T^{(m)}(x),
$$

$$
R_T(x)=T_{\max}(x)-T_{\min}(x).
$$

Range and standard deviation can look spatially similar for small `N` because
both respond to spread in the same ensemble, but they are not equivalent:
range depends only on the two extrema, while the variance uses all samples.

### Streaming Welford merge

The fields are not retained unless `--store-all` is requested. For two partial
batches `A` and `B` with counts `n_A,n_B`, means `mu_A,mu_B`, and centered sums
of squares `M2_A,M2_B`, the implementation merges them using

$$
\delta=\mu_B-\mu_A,
$$

$$
\mu=\mu_A+\delta\frac{n_B}{n_A+n_B},
$$

$$
M2=M2_A+M2_B+\delta^2\frac{n_A n_B}{n_A+n_B}.
$$

This produces the same mean and variance as direct accumulation while requiring
memory proportional to the output field size rather than `N` times the output
field size.

## 4. Extensible temperature accumulators and QoIs

The propagation core accepts arbitrary `TemperatureAccumulator` objects. Each
accumulator receives a physical temperature batch `[B,H,W]` through

```python
accumulator.update(temperatures)
```

and returns its result through

```python
accumulator.finalize()
```

The built-in field-statistics and exceedance-probability implementations use
this interface. Future monitoring-point temperatures, plume geometry, or other
QoIs can therefore be added without modifying the Monte Carlo propagation loop.

This design directly supports the thesis requirement that QoIs remain
extensible and separable from the forward propagation interface.

## 5. Temperature-change and exceedance probability

For a background groundwater temperature `T_bg`, define

$$
\Delta T^{(m)}(x)=T^{(m)}(x)-T_{\mathrm{bg}}.
$$

For threshold `tau`, the empirical spatial exceedance probability is

$$
\widehat P_\tau(x)
=\frac{1}{N}\sum_{m=1}^{N}
\mathbf 1\!\left[\Delta T^{(m)}(x)\ge\tau\right].
$$

The counters are updated online and do not require storage of the individual
temperature fields.

The thesis task description mentions risk maps for deviations such as `0.1 °C`
and `1 °C` from an approximately `10 °C` background. For the specific historical
release25 synthetic dataset, the PFLOTRAN input fixes the initial groundwater
temperature to exactly `10.6 °C` and the injection temperature to `15.6 °C`.
Therefore the Perlin baseline defaults to

$$
T_{\mathrm{bg}}=10.6\ ^\circ\mathrm C
$$

and evaluates the default thresholds `tau = 0.1 °C` and `1.0 °C`. The background
remains configurable for other datasets.

## 6. Historical Perlin permeability law

The current synthetic prior baseline reproduces the `perlin_v2` branch in
`JuliaPelzer/Dataset-generation-with-Pflotran/scripts/create_varying_field.py`.
The historical generator commit recorded by the experiment is

```text
8549bbd9e22d2bc75ce2038c1a0397359e45c971
```

For the released square synthetic domain, the DaRUS settings contain

```text
shape       = 2560 x 2560
size        = 12800 m x 12800 m
frequency   = (18, 18)
k_min       = 1.0193679918450561e-11 m^2
k_max       = 5.09683995922528e-09 m^2
seed_id     = 2907
case        = perlin_v2
```

Let `P(x)` denote one raw `pnoise2` realization and let

$$
P_{\min}=\min_x P(x),\qquad P_{\max}=\max_x P(x).
$$

The realization is normalized field-wise:

$$
U(x)=\frac{P(x)-P_{\min}}{P_{\max}-P_{\min}}.
$$

The historical code performs the range mapping in base-10 logarithmic
permeability space:

$$
\ell(x)=\log_{10}k_{\min}
+U(x)\left(\log_{10}k_{\max}-\log_{10}k_{\min}\right),
$$

followed by

$$
K(x)=10^{\ell(x)}.
$$

Thus `K` is bounded by the configured permeability extrema but is *not* a
log-normal Gaussian random field. It is a transformed Perlin field and should be
interpreted as a synthetic prior/input-distribution baseline.

The historical script generated a random three-component base offset and used
successive integer shifts in the first coordinate for different fields. Its
`np.random.seed(settings["general"]["seed_id"])` line was commented out, so the
three original training fields are not claimed to be bitwise recoverable from
`seed_id=2907` alone. The thesis sampler deliberately makes the UQ seed effective
so new experiments are reproducible while preserving the historical spatial
formula.

## 7. Input normalization used by release25

The adapter reconstructs the transformations stored in each model's `info.yaml`.
For ordinary rescaling from physical interval `[a,b]` to `[0,1]`,

$$
z=\frac{x-a}{b-a}.
$$

For standardization,

$$
z=\frac{x-\mu}{\sigma}.
$$

The reverse transformations are applied to predicted velocity and temperature
fields before they are interpreted physically. The implementation follows the
release25 `NormalizeTransform` contract rather than introducing a new
normalization convention.

## 8. Alignment of valid-convolution outputs

The published `UNetNoPad2` models use valid/no-padding convolutions, so the
predicted spatial field is smaller than the original prepared input/reference
field. For validation, the larger reference is center-cropped to the model
output shape. If an input field has shape `(H,W)` and the target has `(h,w)`, the
crop starts at

$$
y_0=\left\lfloor\frac{H-h}{2}\right\rfloor,\qquad
x_0=\left\lfloor\frac{W-w}{2}\right\rfloor.
$$

The plotted kilometre axes are local coordinates of this aligned output crop;
they should not be interpreted as absolute coordinates in the original larger
domain unless the original crop offset is explicitly added.

## 9. Reproducibility records

Every new Monte Carlo archive uses a versioned result schema and stores the run
metadata both inside the NPZ file and in a neighboring JSON sidecar. Perlin and
empirical release25 experiments additionally record, when available:

- realized sample count and `ddof`;
- seed, frequency and permeability bounds for Perlin runs;
- base offset and sample-index start;
- fixed run identifier and model/data paths;
- Masterthesis and release25 Git SHAs;
- SHA-256 digests of the CNN1 and CNN3 checkpoints;
- Python/platform and selected package versions;
- historical Perlin generator repository, path and commit;
- the published model DOI used by the release25 setup.

A missing Git SHA means the referenced directory was not a Git checkout at run
time; it does not prevent execution.

## 10. What this baseline does and does not represent

The Perlin experiment estimates uncertainty under a synthetic *unconditional*
input law:

$$
K^{(m)}\sim P_{\mathrm{Perlin}}.
$$

It does not yet estimate uncertainty conditioned on fixed borehole observations.
The planned geostatistical phase will instead target a conditional law such as

$$
K^{(m)}\sim p\!\left(K\mid K(x_j)=k_j,\ j=1,\ldots,n_b\right),
$$

while keeping the same surrogate, Monte Carlo propagation, accumulator and
visualization interfaces. The exact stochastic model (Gaussian/log-Gaussian,
covariance/variogram family, conditioning method, etc.) will be selected and
justified from the literature before implementation.
