from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys

import numpy as np
import pytest

from subsurface_uq.sampling import (
    DiagnosticPermeabilitySampler,
    PermeabilityDiagnostics,
)
from subsurface_uq.visualization import plot_permeability_diagnostics


@dataclass
class _ArraySampler:
    batches: list[np.ndarray]

    def __len__(self) -> int:
        return sum(batch.shape[0] for batch in self.batches)

    def __iter__(self):
        yield from self.batches


def test_diagnostic_sampler_yields_original_batches_and_streams_log_statistics():
    fields = np.asarray(
        [
            [[1.0e-10, 1.0e-9], [1.0e-8, 1.0e-7]],
            [[1.0e-9, 1.0e-8], [1.0e-7, 1.0e-6]],
            [[1.0e-8, 1.0e-7], [1.0e-6, 1.0e-5]],
        ],
        dtype=np.float64,
    )
    batches = [fields[:2], fields[2:]]
    diagnostics = PermeabilityDiagnostics(preview_count=2, ddof=1)
    wrapped = DiagnosticPermeabilitySampler(_ArraySampler(batches), diagnostics)

    yielded = list(wrapped)
    assert yielded[0] is batches[0]
    assert yielded[1] is batches[1]

    result = diagnostics.finalize()
    expected = np.log10(fields)
    np.testing.assert_allclose(result.mean_log10_k, np.mean(expected, axis=0), rtol=1e-6)
    np.testing.assert_allclose(result.variance_log10_k, np.var(expected, axis=0, ddof=1), rtol=1e-6)
    np.testing.assert_allclose(result.std_log10_k, np.std(expected, axis=0, ddof=1), rtol=1e-6)
    assert result.count == 3
    assert len(result.preview_log10_k) == 2
    np.testing.assert_allclose(result.preview_log10_k[0], expected[0])
    np.testing.assert_allclose(result.preview_log10_k[1], expected[1])


def test_preview_count_larger_than_sample_count_keeps_only_available_fields():
    fields = np.full((2, 3, 4), 2.0e-10, dtype=np.float32)
    diagnostics = PermeabilityDiagnostics(preview_count=5, ddof=1)
    diagnostics.update(fields)

    result = diagnostics.finalize()
    assert result.count == 2
    assert len(result.preview_log10_k) == 2


def test_non_positive_permeability_is_rejected_before_log10():
    diagnostics = PermeabilityDiagnostics(preview_count=0, ddof=0)
    with pytest.raises(ValueError, match="strictly positive"):
        diagnostics.update(np.asarray([[[1.0e-10, 0.0]]]))


def test_permeability_plot_helper_creates_summary_previews_and_accepts_observations(tmp_path):
    fields = np.asarray(
        [
            [[1.0e-10, 2.0e-10], [3.0e-10, 4.0e-10]],
            [[2.0e-10, 3.0e-10], [4.0e-10, 5.0e-10]],
        ],
        dtype=np.float64,
    )
    diagnostics = PermeabilityDiagnostics(preview_count=2, ddof=1)
    diagnostics.update(fields)
    result = diagnostics.finalize()

    paths = plot_permeability_diagnostics(
        result,
        tmp_path,
        cell_size_m=5.0,
        observation_indices=np.asarray([[0, 1], [1, 0]], dtype=int),
    )

    assert set(paths) == {
        "log10k_mean",
        "log10k_std",
        "log10k_sample_001",
        "log10k_sample_002",
    }
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_release25_grf_help_exposes_permeability_plot_options():
    completed = subprocess.run(
        [sys.executable, "-m", "subsurface_uq.experiments.release25_grf_mc", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--permeability-plots-dir" in completed.stdout
    assert "--permeability-preview-count" in completed.stdout
