import csv
from pathlib import Path

import numpy as np
import openpyxl

from subsurface_uq.sampling import (
    aggregate_measurements_by_reference_cell,
    calibrate_point_covariance_candidates,
    exact_simple_kriging_grid_from_points,
    exact_simple_kriging_predict_points,
    hydraulic_conductivity_to_intrinsic_permeability,
    load_munich_hydraulic_conductivity_measurements,
    load_reference_horizontal_grid,
)
from subsurface_uq.validation.measurements import (
    spatial_block_cross_validate_measurements,
    spatial_block_fold_assignment,
)


def _write_reference_grid(path: Path) -> None:
    active = [(1000.0, 2000.0), (1100.0, 2000.0), (1200.0, 2000.0),
              (1000.0, 2100.0), (1200.0, 2100.0),
              (1000.0, 2200.0), (1100.0, 2200.0), (1200.0, 2200.0)]
    with path.open("w", encoding="utf-8") as handle:
        for x, y in active:
            for z in (0.0, 1.0):
                handle.write(f"{x} {y} {z} 1e-4 2e-4 3e-4\n")


def _write_measurement_workbook(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "kf_werte_180223"
    sheet.append([
        "Ob_Id", "Rechtswert", "Hochwert", "BEARBEITER", "KF Wert",
        "Landkreis", "Strategrap", "GW_Zustand",
    ])
    sheet.append(["A", 1002.0, 2003.0, "TUM", 1.0e-3, "München", "q", "ungespannt"])
    sheet.append([None, 1098.0, 2001.0, "TUM", 2.0e-3, "München", "q", "ungespannt"])
    sheet.append(["C", 1199.0, 2198.0, "TUM", 4.0e-3, "München", "q", "ungespannt"])
    sheet.append(["bad_zero", 1000.0, 2200.0, "TUM", 0.0, "München", "q", "ungespannt"])
    sheet.append(["bad_state", 1000.0, 2200.0, "TUM", 3.0e-3, "München", "q", "gespannt"])
    sheet.append(["inactive", 1100.0, 2100.0, "TUM", 5.0e-3, "München", "q", "ungespannt"])
    sheet.append(["outside", 9999.0, 9999.0, "TUM", 6.0e-3, "München", "q", "ungespannt"])
    sheet.append(["formula", 1000.0, 2000.0, "TUM", "=GEOMEAN(#REF!,A1)", "München", "q", "ungespannt"])
    workbook.save(path)


def test_real_measurement_loader_filters_and_maps_active_footprint(tmp_path):
    grid_path = tmp_path / "grid.csv"
    workbook_path = tmp_path / "measurements.xlsx"
    _write_reference_grid(grid_path)
    _write_measurement_workbook(workbook_path)

    grid = load_reference_horizontal_grid(grid_path, expected_cell_size_m=100.0)
    measurements, metadata = load_munich_hydraulic_conductivity_measurements(
        workbook_path, reference_grid=grid
    )

    assert grid.shape == (3, 3)
    assert np.count_nonzero(grid.active_mask) == 8
    assert len(measurements) == 3
    assert metadata["non_positive_k"] == 1
    assert metadata["groundwater_state_filtered"] == 1
    assert metadata["outside_active_footprint"] == 2
    assert metadata["non_numeric_or_missing_k"] == 1
    assert measurements.measurement_id.tolist()[1].startswith("excel_row_")
    np.testing.assert_array_equal(measurements.ix, [0, 1, 2])
    np.testing.assert_array_equal(measurements.iy, [0, 0, 2])


def test_cell_aggregation_uses_geometric_mean(tmp_path):
    grid_path = tmp_path / "grid.csv"
    workbook_path = tmp_path / "measurements.xlsx"
    _write_reference_grid(grid_path)
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "kf_werte_180223"
    sheet.append(["Ob_Id", "Rechtswert", "Hochwert", "BEARBEITER", "KF Wert", "Landkreis", "Strategrap", "GW_Zustand"])
    sheet.append(["A", 1001.0, 2001.0, "TUM", 1.0e-3, "München", "q", "ungespannt"])
    sheet.append(["B", 1002.0, 2002.0, "TUM", 4.0e-3, "München", "q", "ungespannt"])
    workbook.save(workbook_path)
    grid = load_reference_horizontal_grid(grid_path)
    measurements, _ = load_munich_hydraulic_conductivity_measurements(workbook_path, reference_grid=grid)
    rows = aggregate_measurements_by_reference_cell(measurements)
    assert len(rows) == 1
    assert rows[0]["measurement_count"] == 2
    np.testing.assert_allclose(rows[0]["geometric_mean_hydraulic_conductivity_m_s"], 2.0e-3)


def test_intrinsic_permeability_conversion_is_constant_log_shift():
    conductivity = np.asarray([1.0e-4, 1.0e-3, 1.0e-2])
    permeability = hydraulic_conductivity_to_intrinsic_permeability(conductivity)
    shifts = np.log10(permeability) - np.log10(conductivity)
    np.testing.assert_allclose(shifts, np.full(3, shifts[0]), rtol=0.0, atol=1e-12)


def test_point_variogram_calibration_and_continuous_exact_kriging():
    x, y = np.meshgrid(np.arange(6) * 500.0, np.arange(6) * 500.0)
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    log_values = -2.2 + 0.15 * np.sin(x.ravel() / 900.0) + 0.1 * np.cos(y.ravel() / 1100.0)
    conductivity = 10.0 ** log_values
    results, variograms, counts = calibrate_point_covariance_candidates(
        coordinates,
        conductivity,
        lag_bin_m=500.0,
        max_lag_m=2500.0,
        angle_tolerance_deg=22.5,
        min_pairs_per_bin=2,
    )
    assert {item.covariance_model for item in results} == {
        "matern32", "exponential", "radial_exponential"
    }
    assert all(len(variograms[key][0]) >= 3 for key in ("x", "y", "diag"))
    assert all(np.all(counts[key] >= 2) for key in counts)

    selected = results[0]
    obs = coordinates[[0, 7, 14, 21, 28, 35]]
    obs_values = log_values[[0, 7, 14, 21, 28, 35]]
    posterior = exact_simple_kriging_predict_points(
        observation_coordinates_xy_m=obs,
        observation_log10_k=obs_values,
        query_coordinates_xy_m=obs,
        covariance_model=selected.covariance_model,
        mean_log10_k=selected.mean_log10_k,
        std_log10_k=selected.std_log10_k,
        length_scale_y_m=selected.length_scale_y_m,
        length_scale_x_m=selected.length_scale_x_m,
    )
    np.testing.assert_allclose(posterior.mean_log10_k, obs_values, atol=1e-7)
    np.testing.assert_allclose(posterior.std_log10_k, 0.0, atol=1e-5)


def test_spatial_block_assignment_keeps_blocks_together():
    coordinates = np.asarray([
        [0.0, 0.0], [100.0, 100.0], [1100.0, 0.0], [1200.0, 100.0],
        [0.0, 1100.0], [100.0, 1200.0], [1100.0, 1100.0], [1200.0, 1200.0],
    ])
    assignment = spatial_block_fold_assignment(
        coordinates, n_folds=2, block_size_m=1000.0, seed=7
    )
    assert assignment[0] == assignment[1]
    assert assignment[2] == assignment[3]
    assert assignment[4] == assignment[5]
    assert assignment[6] == assignment[7]
    assert set(assignment.tolist()) == {0, 1}


def test_exact_grid_from_continuous_points_respects_active_mask():
    observations = np.asarray([[0.0, 0.0], [200.0, 200.0], [0.0, 200.0]])
    values = np.asarray([-3.0, -2.5, -2.8])
    mask = np.asarray([[True, True, False], [True, True, True], [False, True, True]])
    result = exact_simple_kriging_grid_from_points(
        observation_coordinates_xy_m=observations,
        observation_log10_k=values,
        x_min_m=0.0,
        y_min_m=0.0,
        nx=3,
        ny=3,
        cell_size_x_m=100.0,
        cell_size_y_m=100.0,
        covariance_model="radial_exponential",
        mean_log10_k=-2.8,
        std_log10_k=0.3,
        length_scale_y_m=300.0,
        length_scale_x_m=400.0,
        active_mask=mask,
    )
    assert np.isnan(result.mean_log10_k[0, 2])
    assert np.isnan(result.std_log10_k[2, 0])
    assert np.isfinite(result.mean_log10_k[1, 1])


def test_spatial_block_cv_predicts_every_measurement_once():
    x, y = np.meshgrid(np.arange(8) * 500.0, np.arange(8) * 500.0)
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    log_values = (
        -2.3
        + 0.12 * np.sin(x.ravel() / 1000.0)
        + 0.09 * np.cos(y.ravel() / 1200.0)
    )
    conductivity = 10.0 ** log_values
    result = spatial_block_cross_validate_measurements(
        coordinates,
        conductivity,
        models=("radial_exponential",),
        n_folds=4,
        block_size_m=1000.0,
        fold_seed=11,
        lag_bin_m=500.0,
        max_lag_m=2500.0,
        angle_tolerance_deg=22.5,
        min_pairs_per_bin=2,
    )
    assert result.summary_rows[0]["n_predictions"] == coordinates.shape[0]
    assert len(result.fold_rows) == 4
    assert np.isfinite(result.summary_rows[0]["rmse_log10_k"])
    assert np.isfinite(result.summary_rows[0]["gaussian_nlpd"])
    assert 0.0 <= result.summary_rows[0]["coverage_90"] <= 1.0
