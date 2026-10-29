"""Visualization helpers for surrogate and UQ outputs."""

from .pce import plot_pce_archive
from .permeability import plot_permeability_diagnostics
from .realistic_permeability import (
    plot_georeference_alignment,
    plot_generated_measurement_overlay,
    plot_generated_permeability_comparison,
    plot_heldout_metric_comparison,
    plot_heldout_reconstruction,
    plot_length_scale_comparison,
    plot_measurement_cv_comparison,
    plot_new_domain_summary,
    plot_nugget_fraction_comparison,
    plot_variogram_fits,
    save_permeability_field_png,
)
from .release25 import save_release25_output_plots, save_release25_overlay_plots
from .uq import (
    MonteCarloPlotData,
    load_monte_carlo_plot_data,
    plot_monte_carlo_archive,
    save_monte_carlo_uq_plots,
)

__all__ = [
    "plot_new_domain_summary",
    "plot_georeference_alignment",
    "plot_generated_measurement_overlay",
    "plot_generated_permeability_comparison",
    "MonteCarloPlotData",
    "load_monte_carlo_plot_data",
    "plot_monte_carlo_archive",
    "plot_pce_archive",
    "plot_heldout_metric_comparison",
    "plot_heldout_reconstruction",
    "plot_length_scale_comparison",
    "plot_measurement_cv_comparison",
    "plot_nugget_fraction_comparison",
    "plot_permeability_diagnostics",
    "plot_variogram_fits",
    "save_permeability_field_png",
    "save_monte_carlo_uq_plots",
    "save_release25_output_plots",
    "save_release25_overlay_plots",
]
