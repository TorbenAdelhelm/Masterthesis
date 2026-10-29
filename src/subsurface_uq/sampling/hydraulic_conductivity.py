"""Unit-safe conversion between intrinsic permeability and hydraulic conductivity.

The LGCNN and PFLOTRAN data use intrinsic permeability ``k`` in m², while some
hydrogeological prior assumptions are expressed as hydraulic conductivity
``K_h`` in m/s. The historical real-K generator used ``k = K_h / 7.5e6``;
physical-SI conversion uses ``k = K_h * mu / (rho*g)``. These conventions are
close but not identical and must never be mixed silently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

Array = np.ndarray
HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR = 7.5e6
SUPPORTED_CONVENTIONS = {"physical", "historical-training"}


@dataclass(frozen=True)
class FluidProperties:
    """Fluid properties used for physical-SI ``k <-> K_h`` conversion."""

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


def hydraulic_conversion_metadata(
    convention: str,
    *,
    fluid: FluidProperties = FluidProperties(),
) -> dict[str, object]:
    """Describe exactly how a modelling ``K_h`` value maps to LGCNN ``k``."""

    convention = str(convention)
    if convention not in SUPPORTED_CONVENTIONS:
        raise ValueError(f"unsupported permeability convention {convention!r}")
    if convention == "historical-training":
        return {
            "convention": convention,
            "permeability_unit": "m^2",
            "hydraulic_conductivity_unit": "m/s",
            "forward": "K_h = k * 7.5e6",
            "inverse": "k = K_h / 7.5e6",
            "divisor_s_inv": HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
            "interpretation": (
                "Exact algebraic conversion used by the historical realistic-data generator; "
                "it is retained for compatibility with the pretrained real-K LGCNN."
            ),
        }
    return {
        "convention": convention,
        **fluid.metadata,
        "interpretation": "Physical SI conversion using explicitly recorded fluid properties.",
    }


def permeability_to_hydraulic_conductivity(
    permeability_m2: Array | float,
    *,
    convention: str = "physical",
    fluid: FluidProperties = FluidProperties(),
) -> Array:
    """Convert intrinsic permeability ``k [m²]`` to ``K_h [m/s]``.

    ``historical-training`` recovers the hydraulic-conductivity values consistent
    with the real-K training generator. ``physical`` applies Darcy's SI relation.
    """

    k = _positive_finite(permeability_m2, name="permeability_m2")
    convention = str(convention)
    if convention == "historical-training":
        return k * HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR
    if convention == "physical":
        return k * fluid.density_kg_m3 * fluid.gravity_m_s2 / fluid.dynamic_viscosity_pa_s
    raise ValueError(f"unsupported permeability convention {convention!r}")


def hydraulic_conductivity_to_permeability(
    hydraulic_conductivity_m_s: Array | float,
    *,
    convention: str = "physical",
    fluid: FluidProperties = FluidProperties(),
) -> Array:
    """Convert hydraulic conductivity ``K_h [m/s]`` to intrinsic ``k [m²]``."""

    kh = _positive_finite(hydraulic_conductivity_m_s, name="hydraulic_conductivity_m_s")
    convention = str(convention)
    if convention == "historical-training":
        return kh / HISTORICAL_TRAINING_HYDRAULIC_TO_PERMEABILITY_DIVISOR
    if convention == "physical":
        return kh * fluid.dynamic_viscosity_pa_s / (fluid.density_kg_m3 * fluid.gravity_m_s2)
    raise ValueError(f"unsupported permeability convention {convention!r}")
