from __future__ import annotations

import torch

from subsurface_uq.visualization import save_release25_output_plots


def test_save_release25_output_plots(tmp_path) -> None:
    h, w = 8, 10
    outputs = {
        "velocity": torch.stack(
            [
                torch.full((h, w), 2.0, dtype=torch.float32),
                torch.full((h, w), -1.0, dtype=torch.float32),
            ],
            dim=0,
        ),
        "streamlines": torch.stack(
            [
                torch.linspace(0.0, 1.0, h * w, dtype=torch.float32).reshape(h, w),
                torch.linspace(1.0, 0.0, h * w, dtype=torch.float32).reshape(h, w),
            ],
            dim=0,
        ),
        "temperature": torch.full((1, h, w), 11.5, dtype=torch.float32),
    }

    saved = save_release25_output_plots(outputs, tmp_path, prefix="RUN_1")

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
