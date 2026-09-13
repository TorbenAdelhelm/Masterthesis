import json

import numpy as np

from subsurface_uq.visualization.pce import plot_pce_archive
from subsurface_uq.visualization.pce_cli import build_parser


def _write_archive(path):
    metadata = {"diagnostics": {"rmse": 0.01}, "pce": {"degree": 2}}
    np.savez_compressed(
        path,
        validation_qoi=np.array([0.10, 0.12, 0.09], dtype=float),
        validation_prediction=np.array([0.101, 0.118, 0.092], dtype=float),
        multi_indices=np.array(
            [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]],
            dtype=int,
        ),
        coefficients=np.array([0.11, 0.01, -0.02, 0.003, 0.002, -0.001]),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def test_plot_pce_archive_creates_compact_diagnostic_set(tmp_path):
    archive = tmp_path / "pce_result.npz"
    output_dir = tmp_path / "plots"
    _write_archive(archive)

    paths = plot_pce_archive(archive, output_dir)

    assert set(paths) == {"parity", "residuals", "coefficients"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_pce_plot_cli_parser_is_minimal_and_explicit():
    args = build_parser().parse_args(
        [
            "--input",
            "run_output/pce_result.npz",
            "--output-dir",
            "run_output/pce_plots",
            "--prefix",
            "degree3_n30",
        ]
    )

    assert args.input == "run_output/pce_result.npz"
    assert args.output_dir == "run_output/pce_plots"
    assert args.prefix == "degree3_n30"
