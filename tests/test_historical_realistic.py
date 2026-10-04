from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from subsurface_uq.experiments.historical_realistic_reconstruction import build_parser
from subsurface_uq.sampling.historical_realistic import (
    HISTORICAL_HYDRAULIC_TO_PERMEABILITY_DIVISOR,
    HISTORICAL_PARENT_CRS,
    HISTORICAL_PARENT_SHA256,
    HISTORICAL_PARENT_SHAPE_YX,
    historical_hydraulic_conductivity_to_training_permeability,
    historical_interpolate_window,
    historical_rotated_source_indices,
    load_parent_hydraulic_conductivity_tif,
    reconstruct_historical_training_permeability,
    sample_parent_hydraulic_conductivity_at_projected_points,
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


def test_historical_parent_identity_constants_match_supplied_source():
    assert HISTORICAL_PARENT_CRS == "EPSG:5678"
    assert HISTORICAL_PARENT_SHAPE_YX == (3797, 3583)
    assert HISTORICAL_PARENT_SHA256 == (
        "6d50f2c6f9b96e3136fe77ad177ef4f70cc2d4cfc632f33ff655e59bdcc096ee"
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


def test_parent_tif_loader_masks_nodata_and_records_metadata(tmp_path: Path):
    path = tmp_path / "parent.tif"
    nodata = -9999.0
    values = np.asarray([[1.0e-3, nodata, 2.0e-3], [3.0e-3, 4.0e-3, 5.0e-3]], dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=1,
        dtype="float32",
        crs="EPSG:5678",
        transform=from_origin(4_431_882.5536, 5_375_953.1595, 20.0, 20.0),
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)

    loaded, metadata = load_parent_hydraulic_conductivity_tif(path)

    assert loaded.shape == (2, 3)
    assert np.isnan(loaded[0, 1])
    assert metadata["crs"] == "EPSG:5678"
    assert metadata["resolution_m"] == [20.0, 20.0]
    assert metadata["shape_yx"] == [2, 3]
    assert len(metadata["sha256"]) == 64


def test_parent_sampling_uses_pixel_centres_and_north_up_affine():
    row, col = np.mgrid[0:3, 0:4]
    parent = 1.0 + row + 10.0 * col
    metadata = {
        "affine_transform": [20.0, 0.0, 100.0, 0.0, -20.0, 200.0]
    }
    # Exact centers of (row=0,col=0) and (row=2,col=3).
    sampled, valid = sample_parent_hydraulic_conductivity_at_projected_points(
        parent,
        metadata,
        np.asarray([110.0, 170.0]),
        np.asarray([190.0, 150.0]),
        method="linear",
    )
    assert np.all(valid)
    np.testing.assert_allclose(sampled, np.asarray([1.0, 33.0]))


def test_historical_reconstruction_cli_checksum_is_opt_in():
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--parent-hydraulic-conductivity-tif",
            "parent.tif",
            "--output-dir",
            "out",
        ]
    )
    assert args.require_historical_parent_metadata is True
    assert args.require_exact_parent_checksum is False
