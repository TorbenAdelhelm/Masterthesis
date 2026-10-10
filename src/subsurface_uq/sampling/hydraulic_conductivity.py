"""Unit-safe conversion between intrinsic permeability and hydraulic conductivity.

The LGCNN and PFLOTRAN data use intrinsic permeability ``k`` in m², while some
hydrogeological prior assumptions are expressed as hydraulic conductivity
``K_h`` in m/s.  The two quantities must never be compared directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FluidProperties:
    """Fluid properties used for ``k <-> K_h`` conversion.

    ``K_h = k * rho * g / mu`` and ``k = K_h * mu / (rho * g)``.
    Defaults correspond to an explicit SI reference convention and are stored in
    generated artifacts so the conversion is reproducible rather than implicit.
    """

    density_kg_m3: float = 1000.0
    dynamic_viscosity_pa_s: float = 1.0e-3
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        values = (
            self.density_kg_m3,
            self.dynamic_viscosity_pa_s,
            self.gravity_m_s2,
        )
        if not all(np.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("fluid properties must be finite and positive")

    @property
    def metadata(self) -> dict[str, float | str]:
        return {
            **asdict(self),
            "conversion": "K_h = k * rho * g / mu",
            "permeability_unit": "m^2",
            "hydraulic_conductivity_unit": "m/s",
        }


def _positive_finite(values: Array | float, *, name: str) -> Array:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def permeability_to_hydraulic_conductivity(
    permeability_m2: Array | float,
    *,
    fluid: FluidProperties = FluidProperties(),
) -> Array:
    """Convert intrinsic permeability ``k [m²]`` to ``K_h [m/s]``."""

    k = _positive_finite(permeability_m2, name="permeability_m2")
    return k * fluid.density_kg_m3 * fluid.gravity_m_s2 / fluid.dynamic_viscosity_pa_s


def hydraulic_conductivity_to_permeability(
    hydraulic_conductivity_m_s: Array | float,
    *,
    fluid: FluidProperties = FluidProperties(),
) -> Array:
    """Convert hydraulic conductivity ``K_h [m/s]`` to intrinsic ``k [m²]``."""

    kh = _positive_finite(hydraulic_conductivity_m_s, name="hydraulic_conductivity_m_s")
    return kh * fluid.dynamic_viscosity_pa_s / (fluid.density_kg_m3 * fluid.gravity_m_s2)
