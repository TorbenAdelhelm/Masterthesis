"""Validation utilities for deterministic and UQ surrogate outputs."""

from .diagnostics import (
    TemperatureDiagnostics,
    save_temperature_diagnostic_plots,
    summarize_temperature_errors,
)
from .measurements import (
    SpatialMeasurementCVResult,
    spatial_block_cross_validate_measurements,
    spatial_block_fold_assignment,
)
from .overlays import (
    build_release25_validation_overlay_context,
    save_validation_overlay_plots,
)
from .temperature import (
    TemperatureComparison,
    center_crop_2d,
    compare_temperature_fields,
    load_prediction_field,
    load_prepared_temperature_label,
    save_temperature_comparison,
)

__all__ = [
    "spatial_block_fold_assignment",
    "spatial_block_cross_validate_measurements",
    "SpatialMeasurementCVResult",
    "TemperatureComparison",
    "TemperatureDiagnostics",
    "build_release25_validation_overlay_context",
    "center_crop_2d",
    "compare_temperature_fields",
    "load_prediction_field",
    "load_prepared_temperature_label",
    "save_temperature_comparison",
    "save_temperature_diagnostic_plots",
    "save_validation_overlay_plots",
    "summarize_temperature_errors",
]
