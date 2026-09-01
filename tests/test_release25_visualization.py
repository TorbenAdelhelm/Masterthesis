from __future__ import annotations

import torch

from subsurface_uq.visualization import (
    save_release25_output_plots,
    save_release25_overlay_plots,
)


def _outputs(h: int = 8, w: int = 10) -> dict[str, torch.Tensor]:
    center = torch.zeros((h, w), dtype=torch.float32)
    center[2:7, 4] = torch.linspace(0.1, 1.0, 5)
    outer = torch.zeros((h, w), dtype=torch.float32)
    outer[2:7, 3] = 0.4
    outer[2:7, 5] = 0.5
    return {
        "velocity": torch.stack(
            [
                torch.linspace(-2.0, 2.0, h * w, dtype=torch.float32).reshape(h, w),
                torch.full((h, w), 1.0, dtype=torch.float32),
            ],
            dim=0,
        ),
        "streamlines": torch.stack([center, outer], dim=0),
        "temperature": torch.linspace(10.0, 12.0, h * w, dtype=torch.float32).reshape(1, h, w),
    }


def test_save_release25_output_plots(tmp_path) -> None:
    saved = save_release25_output_plots(
        _outputs(),
        tmp_path,
        prefix="RUN_1",
        cell_size_m=5.0,
    )

    assert set(saved) == {
        "streamline_center",
        "streamline_outer",
        "temperature",
        "velocity_x",
        "velocity_y",
        "velocity_magnitude",
    }
    for path in saved.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_save_release25_output_plots_can_skip_velocity(tmp_path) -> None:
    outputs = {
        "streamlines": torch.zeros((2, 4, 5), dtype=torch.float32),
        "temperature": torch.ones((1, 4, 5), dtype=torch.float32),
    }

    saved = save_release25_output_plots(
        outputs,
        tmp_path,
        prefix="RUN_2",
        include_velocity=False,
    )

    assert set(saved) == {"streamline_center", "streamline_outer", "temperature"}


def test_save_release25_overlay_plots_marks_heat_pumps_and_streamlines(tmp_path) -> None:
    h, w = 8, 10
    material = torch.ones((h + 4, w + 4), dtype=torch.float32)
    # Two disconnected heat-pump components survive the centered crop.
    material[3:5, 4:6] = 2.0
    material[7:9, 9:11] = 2.0

    saved = save_release25_overlay_plots(
        _outputs(h, w),
        material,
        tmp_path,
        prefix="RUN_1",
        cell_size_m=5.0,
        streamline_threshold=0.05,
    )

    assert set(saved) == {
        "temperature_streamlines",
        "temperature_heatpumps",
        "temperature_streamlines_heatpumps",
    }
    for path in saved.values():
        assert path.is_file()
        assert path.stat().st_size > 0
