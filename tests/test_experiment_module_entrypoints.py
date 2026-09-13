from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module, expected_option",
    [
        ("subsurface_uq.experiments.release25_perlin", "--n-samples"),
        ("subsurface_uq.experiments.release25_perlin_pce", "--n-train"),
        (
            "subsurface_uq.experiments.release25_streamline_equivalence",
            "--sample-indices",
        ),
    ],
)
def test_experiment_module_help_is_executable(module: str, expected_option: str):
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--release25-repo" in completed.stdout
    assert "--cnn1-dir" in completed.stdout
    assert "--cnn2-dir" in completed.stdout
    assert "--prepared-pki-dir" in completed.stdout
    assert "--fixed-run-id" in completed.stdout
    assert expected_option in completed.stdout