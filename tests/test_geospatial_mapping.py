import numpy as np
from scipy.interpolate import RegularGridInterpolator

from subsurface_uq.sampling.geospatial import (
    GEOREFERENCE_SWEEP_REPRESENTATIONS,
    LGCNNDomainGeoreference,
    ReferencePermeabilitySurface,
    infer_lgcnn_domain_georeference,
    load_reference_permeability_surface,
    orient_raw_field,
    geographic_to_raw_field,
    summarize_georeference_sweep,
)


def test_reference_surface_loader_selects_top_layer(tmp_path):
    path = tmp_path / "reference.csv"
    rows = []
    for x in (1000.0, 1100.0):
        for y in (2000.0, 2100.0):
            for z in (10.0, 20.0, 30.0):
                base = 1.0e-3 + 1.0e-6 * (x - 1000.0) + 2.0e-6 * (y - 2000.0)
                p10 = base * (1.0 + 0.01 * z)
                p50 = base * (1.0 + 0.02 * z)
                p90 = base * (1.0 + 0.03 * z)
                rows.append(f"{x} {y} {z} {p10} {p50} {p90}\n")
    path.write_text("".join(rows), encoding="utf-8")

    surface = load_reference_permeability_surface(
        path, column="K_P50", z_mode="top"
    )

    assert surface.shape == (2, 2)
    np.testing.assert_allclose(surface.x_m, [1000.0, 1100.0])
    np.testing.assert_allclose(surface.y_m, [2000.0, 2100.0])
    expected00 = 1.0e-3 * (1.0 + 0.02 * 30.0)
    np.testing.assert_allclose(surface.values[0, 0], expected00)
    assert np.all(surface.active_mask)


def test_infer_lgcnn_georeference_recovers_crop_orientation_and_origin():
    rng = np.random.default_rng(17)
    ny_ref, nx_ref = 34, 40
    x = 4450000.0 + np.arange(nx_ref) * 100.0
    y = 5320000.0 + np.arange(ny_ref) * 100.0

    # A smooth but spatially unique reference pattern avoids mirror/transpose ties.
    noise = rng.normal(size=(ny_ref, nx_ref))
    padded = np.pad(noise, 1, mode="edge")
    smooth = sum(
        padded[dy : dy + ny_ref, dx : dx + nx_ref]
        for dy in range(3)
        for dx in range(3)
    ) / 9.0
    yy, xx = np.meshgrid(np.arange(ny_ref), np.arange(nx_ref), indexing="ij")
    reference_log = -2.5 + 0.08 * smooth + 0.003 * xx + 0.002 * yy
    reference = ReferencePermeabilitySurface(
        x_m=x,
        y_m=y,
        values=10.0**reference_log,
        active_mask=np.ones_like(reference_log, dtype=bool),
        value_name="K_P50",
        z_mode="top",
        z_value_m=None,
    )

    coarse_ix = 8
    coarse_iy = 7
    raw_cell = 5.0
    nx_raw = 12 * 20
    ny_raw = 10 * 20
    first_x = x[coarse_ix] - 47.5
    first_y = y[coarse_iy] - 47.5
    raw_x = first_x + np.arange(nx_raw) * raw_cell
    raw_y = first_y + np.arange(ny_raw) * raw_cell
    gy, gx = np.meshgrid(raw_y, raw_x, indexing="ij")
    interp = RegularGridInterpolator(
        (y, x), reference_log, bounds_error=True
    )
    geo_log = interp(np.column_stack((gy.ravel(), gx.ravel()))).reshape(ny_raw, nx_raw)
    shift = -6.991
    geo_field = 10.0 ** (geo_log + shift)

    # release-style raw axes are deliberately transposed; inference must recover it.
    raw_source = geo_field.T
    mapping = infer_lgcnn_domain_georeference(
        raw_source,
        reference,
        run_name="RUN_1",
        raw_cell_size_m=raw_cell,
        coarse_anchor_stride=2,
        coarse_keep_per_transform=2,
        refine_radius_m=20.0,
        refine_step_m=5.0,
        refine_sample_stride_cells=20,
        min_reference_coverage=0.95,
        min_correlation=0.995,
        max_centered_rmse_log10=0.01,
    )

    assert mapping.transform == "transpose"
    np.testing.assert_allclose(mapping.first_cell_center_x_m, first_x, atol=5.0)
    np.testing.assert_allclose(mapping.first_cell_center_y_m, first_y, atol=5.0)
    np.testing.assert_allclose(
        mapping.log10_unit_shift_raw_minus_reference, shift, atol=0.01
    )
    assert mapping.correlation > 0.999
    assert mapping.centered_rmse_log10 < 0.005
    assert mapping.validated
    assert mapping.nx == nx_raw
    assert mapping.ny == ny_raw

    row, col = mapping.xy_to_fractional_indices(
        np.asarray([first_x + 10.0]), np.asarray([first_y + 15.0])
    )
    np.testing.assert_allclose(row, [3.0])
    np.testing.assert_allclose(col, [2.0])


def test_orientation_transforms_round_trip_basic_shapes():
    raw = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(orient_raw_field(raw, "identity"), raw)
    np.testing.assert_array_equal(orient_raw_field(raw, "transpose"), raw.T)
    np.testing.assert_array_equal(orient_raw_field(raw, "flip_x"), raw[:, ::-1])
    np.testing.assert_array_equal(orient_raw_field(raw, "flip_y"), raw[::-1, :])


def test_geographic_to_raw_is_inverse_for_all_transforms():
    raw = np.arange(30).reshape(5, 6)
    for transform in (
        "identity",
        "flip_y",
        "flip_x",
        "flip_xy",
        "transpose",
        "transpose_flip_y",
        "transpose_flip_x",
        "transpose_flip_xy",
    ):
        geographic = orient_raw_field(raw, transform)
        reconstructed = geographic_to_raw_field(geographic, transform)
        np.testing.assert_array_equal(reconstructed, raw)


def test_georeference_edges_are_cell_edge_coordinates():
    mapping = LGCNNDomainGeoreference(
        run_name="RUN_1",
        transform="identity",
        crs_assumption="test",
        first_cell_center_x_m=102.5,
        first_cell_center_y_m=202.5,
        cell_size_m=5.0,
        nx=4,
        ny=3,
        reference_column="K_P50",
        reference_z_mode="top",
        reference_z_m=None,
        log10_unit_shift_raw_minus_reference=0.0,
        unit_shift_interpretation="same",
        centered_rmse_log10=0.0,
        correlation=1.0,
        reference_coverage_fraction=1.0,
        matched_points=12,
        coarse_origin_ix=0,
        coarse_origin_iy=0,
        validated=True,
    )
    assert mapping.west_edge_m == 100.0
    assert mapping.south_edge_m == 200.0
    assert mapping.east_edge_m == 120.0
    assert mapping.north_edge_m == 215.0
    assert mapping.contains_xy(np.asarray([100.0, 119.9]), np.asarray([200.0, 214.9])).all()


def test_georeference_sweep_requires_cross_run_consistency():
    rows = []
    runs = ("RUN_1", "RUN_2", "RUN_3")
    for column, z_mode in GEOREFERENCE_SWEEP_REPRESENTATIONS:
        for index, run in enumerate(runs):
            good = (column, z_mode) == ("K_P50", "log_geomean")
            rows.append(
                {
                    "run_name": run,
                    "reference_column": column,
                    "reference_z_mode": z_mode,
                    "transform": "transpose" if good else ("identity" if index < 2 else "flip_x"),
                    "correlation": 0.96 if good else 0.25,
                    "centered_rmse_log10": 0.08 if good else 0.9,
                    "reference_coverage_fraction": 0.95,
                    "log10_unit_shift_raw_minus_reference": (
                        -6.99 + 0.02 * index if good else -5.5 + 0.4 * index
                    ),
                    "unit_shift_interpretation": (
                        "reference_values_behave_like_hydraulic_conductivity_m_per_s"
                        if good
                        else "unit_relation_ambiguous"
                    ),
                    "validated": good,
                    "error": None,
                }
            )

    summary = summarize_georeference_sweep(
        rows,
        run_names=runs,
        max_unit_shift_spread_log10=0.15,
    )

    assert summary["consistent_defensible_mapping_exists"]
    assert summary["defensible_representation_count"] == 1
    selected = summary["selected_representation"]
    assert selected["reference_column"] == "K_P50"
    assert selected["reference_z_mode"] == "log_geomean"
    assert selected["consistent_transform"] == "transpose"
    assert selected["unit_shift_spread_log10"] < 0.15
    selected_rows = [row for row in summary["rows"] if row["selected_representation_row"]]
    assert len(selected_rows) == 3
    assert all(row["consistent_defensible_mapping_exists"] for row in selected_rows)


def test_georeference_sweep_rejects_individually_good_but_inconsistent_transforms():
    runs = ("RUN_1", "RUN_2")
    rows = []
    for column, z_mode in GEOREFERENCE_SWEEP_REPRESENTATIONS:
        for index, run in enumerate(runs):
            target = (column, z_mode) == ("K_P10", "top")
            rows.append(
                {
                    "run_name": run,
                    "reference_column": column,
                    "reference_z_mode": z_mode,
                    "transform": ("identity" if index == 0 else "flip_y") if target else "identity",
                    "correlation": 0.95 if target else 0.2,
                    "centered_rmse_log10": 0.10 if target else 1.0,
                    "reference_coverage_fraction": 0.95,
                    "log10_unit_shift_raw_minus_reference": -6.99,
                    "unit_shift_interpretation": "reference_values_behave_like_hydraulic_conductivity_m_per_s",
                    "validated": target,
                    "error": None,
                }
            )

    summary = summarize_georeference_sweep(rows, run_names=runs)

    assert not summary["consistent_defensible_mapping_exists"]
    assert summary["defensible_representation_count"] == 0
    assert summary["selected_representation"] is None
