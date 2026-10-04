from __future__ import annotations

import numpy as np

from subsurface_uq.sampling.historical_realistic import (
    HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
    historical_hydraulic_conductivity_to_training_permeability,
    historical_interpolate_window,
    historical_rotated_source_indices,
    reconstruct_historical_training_permeability,
)
from subsurface_uq.sampling.training_provenance import RealisticRunMetadata


def _metadata(angle: float = 0.0) -> RealisticRunMetadata:
    return RealisticRunMetadata(
        run_name="RUN_0",
        original_resolution_m=20.0,
        rotation_angle_deg=angle,
        start_position_m=(200.0, 400.0),
        source_path="synthetic",
    )


def test_historical_conversion_matches_source_code_divisor():
    kh = np.asarray([7.5e-4, 1.5e-3])
    k = historical_hydraulic_conductivity_to_training_permeability(kh)
    np.testing.assert_allclose(
        k,
        kh / HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
    )


def test_historical_rotated_indices_reproduce_plus_90_degree_convention():
    # start=(10,20) source cells; angle=0 means source code rotates the mesh by +90°.
    x_idx, y_idx = historical_rotated_source_indices(
        _metadata(angle=0.0),
        window_shape_source_cells=(2, 3),
    )
    np.testing.assert_array_equal(
        x_idx,
        np.asarray([[10, 10, 10], [9, 9, 9]]),
    )
    np.testing.assert_array_equal(
        y_idx,
        np.asarray([[20, 21, 22], [20, 21, 22]]),
    )


def test_historical_interpolation_uses_destination_cell_centres():
    # Source is affine in the two interpolation coordinates, so interpolation is exact.
    a0, a1 = np.mgrid[0:3, 0:3]
    source = 2.0 + 0.1 * (a0 * 20.0) + 0.2 * (a1 * 20.0)
    result = historical_interpolate_window(
        source,
        original_resolution_m=20.0,
        destination_resolution_m=10.0,
        destination_shape_yx=(4, 4),
    )
    y, x = np.mgrid[0:4, 0:4]
    # Historical interpolator is queried as (x_center, y_center) against source axes 0/1.
    expected = 2.0 + 0.1 * ((x + 0.5) * 10.0) + 0.2 * ((y + 0.5) * 10.0)
    np.testing.assert_allclose(result, expected)


def test_full_historical_reconstruction_returns_requested_shape():
    parent = np.full((80, 80), 7.5e-4, dtype=np.float64)
    result = reconstruct_historical_training_permeability(
        parent,
        _metadata(angle=0.0),
        destination_shape_yx=(8, 8),
        destination_resolution_m=5.0,
    )
    assert result.reconstructed_permeability_m2.shape == (8, 8)
    np.testing.assert_allclose(result.reconstructed_permeability_m2, 1.0e-10)
