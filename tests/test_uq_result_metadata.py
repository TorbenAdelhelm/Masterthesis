from __future__ import annotations

import json

import numpy as np

from subsurface_uq.io import save_monte_carlo_result
from subsurface_uq.propagation import MonteCarloResult


def test_save_monte_carlo_result_persists_required_metadata(tmp_path) -> None:
    field = np.ones((3, 4), dtype=np.float32)
    result = MonteCarloResult(
        count=5,
        mean=field,
        variance=field * 2,
        std=field * 3,
        minimum=field * 4,
        maximum=field * 5,
        samples=None,
    )
    destination = tmp_path / "uq_result.npz"
    metadata = {
        "seed": 2907,
        "frequency": [18.0, 18.0],
        "k_min": 1.0193679918450561e-11,
        "k_max": 5.09683995922528e-09,
        # Intentionally inconsistent to verify the writer uses result.count.
        "sample_count": 999,
    }

    archive_path, metadata_path = save_monte_carlo_result(
        result,
        destination,
        metadata=metadata,
    )

    assert archive_path == destination.resolve()
    assert metadata_path.is_file()

    sidecar = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert sidecar["seed"] == 2907
    assert sidecar["frequency"] == [18.0, 18.0]
    assert sidecar["k_min"] == metadata["k_min"]
    assert sidecar["k_max"] == metadata["k_max"]
    assert sidecar["sample_count"] == 5
    assert sidecar["result_schema_version"] == 2

    with np.load(archive_path, allow_pickle=False) as archive:
        embedded = json.loads(str(archive["metadata_json"]))
        assert int(archive["count"]) == 5
        assert int(archive["schema_version"]) == 2
        assert embedded == sidecar
