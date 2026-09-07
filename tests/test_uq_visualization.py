from __future__ import annotations

import json

import numpy as np

from subsurface_uq.io import save_monte_carlo_result
from subsurface_uq.propagation import MonteCarloResult
from subsurface_uq.visualization import (
    load_monte_carlo_plot_data,
    plot_monte_carlo_archive,
)


def _result() -> MonteCarloResult:
    mean = np.linspace(10.0, 12.0, 20, dtype=np.float32).reshape(4, 5)
    std = np.linspace(0.0, 0.4, 20, dtype=np.float32).reshape(4, 5)
    minimum = mean - 0.5
    maximum = mean + 0.75
    probabilities = np.stack(
        [
            np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(4, 5),
            np.linspace(1.0, 0.0, 20, dtype=np.float32).reshape(4, 5),
        ]
    )
    return MonteCarloResult(
        count=5,
        mean=mean,
        variance=std * std,
        std=std,
        minimum=minimum,
        maximum=maximum,
        samples=None,
        exceedance_thresholds=(0.1, 1.0),
        exceedance_probabilities=probabilities,
        background_temperature=10.0,
    )


def test_load_and_plot_monte_carlo_archive_with_exceedance_maps(tmp_path) -> None:
    archive = tmp_path / "perlin_mc.npz"
    save_monte_carlo_result(
        _result(),
        archive,
        metadata={"seed": 2907, "cell_size_m": 5.0},
    )

    loaded = load_monte_carlo_plot_data(archive)
    assert loaded.count == 5
    assert loaded.field_shape == (4, 5)
    assert loaded.exceedance_thresholds == (0.1, 1.0)
    np.testing.assert_allclose(loaded.temperature_range, 1.25)

    paths = plot_monte_carlo_archive(archive, tmp_path / "plots")
    expected = {
        "temperature_mean",
        "temperature_std",
        "temperature_range",
        "temperature_mean_std_overlay",
        "delta_temperature_mean",
        "probability_deltaT_ge_0p1",
        "probability_deltaT_ge_1",
    }
    assert set(paths) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())


def test_plotter_supports_older_archive_without_exceedance_fields(tmp_path) -> None:
    archive = tmp_path / "legacy_mc.npz"
    metadata = json.dumps({"sample_count": 3})
    mean = np.ones((3, 4), dtype=np.float32) * 11.0
    np.savez_compressed(
        archive,
        count=np.asarray(3, dtype=np.int64),
        mean=mean,
        std=np.ones_like(mean) * 0.2,
        minimum=mean - 0.3,
        maximum=mean + 0.4,
        metadata_json=np.asarray(metadata),
    )

    paths = plot_monte_carlo_archive(
        archive,
        tmp_path / "legacy_plots",
        background_temperature=10.0,
    )
    assert set(paths) == {
        "temperature_mean",
        "temperature_std",
        "temperature_range",
        "temperature_mean_std_overlay",
        "delta_temperature_mean",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths.values())
