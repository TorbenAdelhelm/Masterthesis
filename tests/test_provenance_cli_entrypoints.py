from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def _run_module_help(module: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_historical_reconstruction_module_help_is_invokable():
    output = _run_module_help(
        "subsurface_uq.experiments.historical_realistic_reconstruction"
    )
    assert "--dataset-root" in output
    assert "--parent-hydraulic-conductivity-tif" in output
    assert "--require-exact-parent-checksum" in output
    assert "--output-dir" in output


def test_parent_measurement_audit_module_help_is_invokable():
    output = _run_module_help("subsurface_uq.experiments.parent_measurement_audit")
    assert "--parent-hydraulic-conductivity-tif" in output
    assert "--measurements" in output
    assert "--output-dir" in output


def test_pyproject_registers_provenance_console_scripts():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert (
        'subsurface-uq-historical-realistic-reconstruction = '
        '"subsurface_uq.experiments.historical_realistic_reconstruction:main"'
        in pyproject
    )
    assert (
        'subsurface-uq-parent-measurement-audit = '
        '"subsurface_uq.experiments.parent_measurement_audit:main"'
        in pyproject
    )
