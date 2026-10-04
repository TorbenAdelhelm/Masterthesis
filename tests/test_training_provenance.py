from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from subsurface_uq.experiments.training_provenance import build_parser
from subsurface_uq.sampling.geospatial import LGCNNDomainGeoreference
from subsurface_uq.sampling.training_provenance import (
    discover_realistic_runs,
    load_realistic_run_metadata,
    measurement_training_consistency,
    sample_training_field_at_projected_points,
    summarize_metadata_georeference_relation,
)


def _georef(run_name: str = "RUN_1") -> LGCNNDomainGeoreference:
    return LGCNNDomainGeoreference(
        run_name=run_name,
        transform="identity",
        crs_assumption="test",
        first_cell_center_x_m=100.0,
        first_cell_center_y_m=200.0,
        cell_size_m=5.0,
        nx=4,
        ny=3,
        reference_column="K_P50",
        reference_z_mode="top",
        reference_z_m=None,
        log10_unit_shift_raw_minus_reference=0.0,
        unit_shift_interpretation="test",
        centered_rmse_log10=0.0,
        correlation=1.0,
        reference_coverage_fraction=1.0,
        matched_points=12,
        coarse_origin_ix=0,
        coarse_origin_iy=0,
        validated=True,
    )


def test_load_and_discover_realistic_run_metadata(tmp_path: Path):
    for number in (3, 1, 2):
        run = tmp_path / f"RUN_{number}"
        run.mkdir()
        (run / "pflotran.h5").touch()
        (run / "realistic_params.yaml").write_text(
            yaml.safe_dump(
                {
                    "orig resolution [m]": 20,
                    "rotation angle [°]": 99.0 + number,
                    "start position [m]": [55000 + number, 43000 + number],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    assert discover_realistic_runs(tmp_path) == ("RUN_1", "RUN_2", "RUN_3")
    metadata = load_realistic_run_metadata(tmp_path, "RUN_2")
    assert metadata.original_resolution_m == 20.0
    assert metadata.rotation_angle_deg == 101.0
    assert metadata.start_position_m == (55002.0, 43002.0)


def test_training_field_sampling_is_bilinear_in_log_space():
    # log10(k) is an affine plane, so bilinear interpolation is exact.
    row, col = np.mgrid[0:3, 0:4]
    log_field = -10.0 + 0.1 * row + 0.05 * col
    field = 10.0**log_field
    georef = _georef()
    x = np.asarray([102.5, 107.5])
    y = np.asarray([202.5, 207.5])

    sampled, inside = sample_training_field_at_projected_points(field, georef, x, y)

    assert np.all(inside)
    expected_log = np.asarray([-9.925, -9.775])
    np.testing.assert_allclose(np.log10(sampled), expected_log, atol=1.0e-12)


def test_measurement_training_consistency_recovers_exact_samples():
    row, col = np.mgrid[0:3, 0:4]
    field = 10.0 ** (-10.0 + 0.1 * row + 0.05 * col)
    georef = _georef()
    x = np.asarray([100.0, 105.0, 110.0])
    y = np.asarray([200.0, 205.0, 210.0])
    measured, inside = sample_training_field_at_projected_points(field, georef, x, y)
    assert np.all(inside)

    metrics, _, valid = measurement_training_consistency(
        field,
        georef,
        x,
        y,
        measured,
    )

    assert np.all(valid)
    assert metrics["measurement_count_in_run"] == 3
    assert metrics["rmse_log10"] < 1.0e-12
    assert metrics["mae_log10"] < 1.0e-12
    assert metrics["fraction_within_0_05_log10"] == 1.0


def test_metadata_relation_reports_but_does_not_claim_rotation_semantics(tmp_path: Path):
    metadata = []
    georefs = {}
    for index, run_name in enumerate(("RUN_1", "RUN_2", "RUN_3")):
        run = tmp_path / run_name
        run.mkdir()
        start = (1000.0 + 10.0 * index, 2000.0 + 20.0 * index)
        (run / "realistic_params.yaml").write_text(
            yaml.safe_dump(
                {
                    "orig resolution [m]": 20,
                    "rotation angle [°]": 90.0 + index,
                    "start position [m]": list(start),
                }
            ),
            encoding="utf-8",
        )
        metadata.append(load_realistic_run_metadata(tmp_path, run_name))
        base = _georef(run_name)
        georefs[run_name] = LGCNNDomainGeoreference(
            **{
                **base.__dict__,
                "first_cell_center_x_m": start[0] + 4_000_000.0,
                "first_cell_center_y_m": start[1] + 5_000_000.0,
            }
        )

    result = summarize_metadata_georeference_relation(
        metadata,
        georefs,
        tolerance_m=1.0,
    )

    # A pure translation test cannot distinguish which fixed point of an equally
    # sized crop the historical start position referred to. It can only establish
    # that the run-to-run relation is translation-consistent.
    assert result["consistent_candidate_count"] == 3
    assert result["unique_simple_translation_candidate"] is False
    assert result["selected_simple_translation_candidate"] is None
    assert result["rotation_convention_resolved"] is False


def test_training_provenance_cli_defaults_to_reference_sweep():
    args = build_parser().parse_args(
        [
            "--dataset-root",
            "dataset",
            "--measurements",
            "measurements.xlsx",
            "--reference-grid",
            "reference.csv",
            "--output-dir",
            "out",
        ]
    )
    assert args.reference_mode == "sweep"
    assert args.metadata_filename == "realistic_params.yaml"
    assert args.cell_size_m == 5.0
