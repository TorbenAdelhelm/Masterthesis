from __future__ import annotations

import numpy as np

from subsurface_uq.validation.overlays import save_validation_overlay_plots


def test_save_validation_overlay_plots(tmp_path) -> None:
    h, w = 12, 16
    prediction = np.full((h, w), 11.5, dtype=np.float32)
    reference = np.full((h, w), 11.2, dtype=np.float32)
    absolute_error = np.abs(prediction - reference)

    streamline = np.zeros((h, w), dtype=np.float32)
    streamline[:, 7] = 0.2

    heat_pump_centers = np.asarray(
        [
            [2.0, 3.0],
            [9.0, 8.0],
        ],
        dtype=np.float64,
    )

    saved = save_validation_overlay_plots(
        prediction=prediction,
        reference=reference,
        absolute_error=absolute_error,
        overlay_context={
            "central_streamline": streamline,
            "heat_pump_centers": heat_pump_centers,
        },
        directory=tmp_path,
        prefix="release25_RUN_1",
        cell_size_m=5.0,
        streamline_threshold=0.05,
    )

    assert set(saved) == {
        "prediction_overlay",
        "reference_overlay",
        "absolute_error_overlay",
    }
    for path in saved.values():
        assert path.is_file()
        assert path.stat().st_size > 0
