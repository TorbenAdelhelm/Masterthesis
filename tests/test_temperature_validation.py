from __future__ import annotations

import numpy as np
import torch
import yaml

from subsurface_uq.validation import (
    center_crop_2d,
    compare_temperature_fields,
    load_prediction_field,
    load_prepared_temperature_label,
    save_temperature_comparison,
)


def test_center_crop_2d_matches_release25_centering() -> None:
    field = np.arange(36, dtype=np.float32).reshape(6, 6)
    cropped = center_crop_2d(field, (4, 4))
    np.testing.assert_array_equal(cropped, field[1:5, 1:5])


def test_compare_temperature_fields_center_crops_reference() -> None:
    reference = np.arange(36, dtype=np.float64).reshape(6, 6)
    aligned = reference[1:5, 1:5]
    prediction = aligned + 2.0

    result = compare_temperature_fields(prediction, reference)

    assert result.shape == (4, 4)
    assert result.mae == 2.0
    assert result.mse == 4.0
    assert result.rmse == 2.0
    assert result.max_absolute_error == 2.0
    assert result.bias == 2.0
    np.testing.assert_allclose(result.reference, aligned)


def test_load_prepared_temperature_label_reverse_standardizes(tmp_path) -> None:
    dataset = tmp_path / "temperature"
    labels = dataset / "Labels"
    labels.mkdir(parents=True)
    info = {
        "Labels": {
            "Temperature [C]": {
                "index": 0,
                "norm": "Standardize",
                "mean": 10.0,
                "std": 2.0,
                "min": 6.0,
                "max": 14.0,
            }
        }
    }
    with (dataset / "info.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(info, handle)
    normalized = torch.tensor([[[0.0, 0.5], [-1.0, 1.0]]], dtype=torch.float32)
    torch.save(normalized, labels / "RUN_1.pt")

    physical = load_prepared_temperature_label(dataset, "RUN_1")

    expected = np.array([[10.0, 11.0], [8.0, 12.0]], dtype=np.float32)
    np.testing.assert_allclose(physical, expected)


def test_prediction_loading_and_comparison_save(tmp_path) -> None:
    prediction_path = tmp_path / "prediction.npz"
    np.savez_compressed(prediction_path, mean=np.full((3, 4), 11.5, dtype=np.float32))

    prediction = load_prediction_field(prediction_path)
    result = compare_temperature_fields(prediction, np.full((3, 4), 11.0))
    destination = save_temperature_comparison(
        result,
        tmp_path / "validation.npz",
        run_id="RUN_1",
    )

    with np.load(destination) as archive:
        assert archive["run_id"].item() == "RUN_1"
        assert float(archive["mae"]) == 0.5
        assert archive["prediction"].shape == (3, 4)
        assert archive["reference"].shape == (3, 4)
