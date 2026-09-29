import numpy as np

from subsurface_uq.sampling import NewLGCNNDomain
from subsurface_uq.visualization.realistic_permeability import (
    plot_generated_permeability_comparison,
)


def test_generated_permeability_comparison_png_contains_reference_panels(tmp_path):
    generated = np.asarray(
        [
            [1.0e-10, 1.2e-10, 1.4e-10, 1.6e-10],
            [1.1e-10, 1.3e-10, 1.5e-10, 1.7e-10],
            [1.2e-10, 1.4e-10, 1.6e-10, 1.8e-10],
            [1.3e-10, 1.5e-10, 1.7e-10, 1.9e-10],
        ],
        dtype=np.float64,
    )
    reference = np.log10(generated * 0.95)
    training = generated * 1.05
    domain = NewLGCNNDomain(
        west_edge_m=1000.0,
        south_edge_m=2000.0,
        cell_size_m=5.0,
        nx=4,
        ny=4,
        selection_method="test",
        conditioning_measurement_count=2,
    )
    destination = tmp_path / "comparison.png"

    result = plot_generated_permeability_comparison(
        generated_permeability_m2=generated,
        conditioned_reference_log10_k=reference,
        training_reference_permeability_m2=training,
        training_reference_label="RUN_1",
        domain=domain,
        measurement_x_m=np.asarray([1002.5, 1012.5]),
        measurement_y_m=np.asarray([2002.5, 2012.5]),
        measurement_log10_k=np.log10(np.asarray([1.1e-10, 1.6e-10])),
        shared_log10_limits=(-10.2, -9.6),
        destination=destination,
    )

    assert result == destination.resolve()
    assert destination.is_file()
    assert destination.stat().st_size > 0
