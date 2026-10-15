import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from subsurface_uq.sampling.reference_coherence import (
    reference_ensemble_coherence, reference_probe_law_diagnostics,
)
from subsurface_uq.sampling.reference_field import build_reference_field_maps
from subsurface_uq.visualization.reference_validation import write_reference_validation_report


def _report(*, count=8, constant=False, method="RQMC"):
    rows, columns = np.indices((8, 10))
    reference = np.full((8, 10), 1e-10) if constant else 10. ** (-10 + .08 * rows + .03 * columns)
    model, _ = build_reference_field_maps(
        reference, cell_size_m=20, residual_std_log10_k=.08,
        length_scale_m=(50, 80), n_modes=8, center="arithmetic-mean",
    )
    coordinates = np.random.default_rng(45).normal(size=(count, model.dimension))
    logs = model.map_log10_coordinates(coordinates)
    generated = 10. ** logs
    probes = np.array([[0, 0], [3, 4], [7, 9]])
    coherence = reference_ensemble_coherence(reference, generated, cell_size_m=20, max_lag_cells=3)
    marginals = []
    for row in coherence["per_field"]:
        features = row["field_features"]
        marginals.append({"sample_index": row["sample_index"],
                          "log10_mean": features["mean_log10_k"],
                          "log10_std": features["std_log10_k"],
                          "range_m2": (10. ** np.asarray(row["generated_log10_range"])).tolist(),
                          "quantile_probabilities": [.5],
                          "quantiles_m2": [10. ** features["q50_log10_k"]]})
    return {
        "schema_version": 1,
        "source": {"reference_run": "RUN_2", "center": "arithmetic-mean", "reference_sha256": "a" * 64},
        "support": {"native_cell_size_m": 20, "diagnostic_cell_size_m": 20,
                    "diagnostic_shape": [8, 10], "geometric_coarsening_factor": 1},
        "native_marginals": {"reference": {"pixel_count": 80}, "generated_per_field": marginals},
        "native_agreement": coherence["per_field"],
        "coherence": coherence,
        "probe_law": reference_probe_law_diagnostics(model, logs[:, probes[:, 0], probes[:, 1]],
                                                       probes, sampling_method=method),
        "assessment": {"status": "reference_review_required", "sample_filtering": None},
    }


@pytest.mark.parametrize("method", ["MC", "RQMC"])
def test_report_renders_real_backend_diagnostics_without_mutating_metrics(tmp_path, method):
    report = _report(method=method)
    original = copy.deepcopy(report)
    outputs = write_reference_validation_report(report, tmp_path)
    assert outputs == {
        "distribution_png": "validation_distribution.png",
        "correlation_png": "validation_correlation.png",
        "spectra_png": "validation_spectra.png",
        "probe_law_png": "validation_probe_law.png",
        "stability_png": "validation_stability.png",
        "markdown": "validation_report.md",
    }
    for key, relative in outputs.items():
        path = tmp_path / relative
        assert path.is_file()
        if key.endswith("png"):
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            rendered = plt.imread(path)
            assert min(rendered.shape[:2]) > (600 if key == "stability_png" else 900)
            assert float(rendered.std()) > .05
    markdown = (tmp_path / outputs["markdown"]).read_text(encoding="utf-8")
    for phrase in ("not confidence intervals", "do not reject or filter", "training support separately",
                   "scrambled Sobol points are not iid", "retained Gaussian/KL law", "RUN_1, RUN_2 and RUN_3",
                   "native cells (no coarsening)", "reference_review_required", "validation.json", "a" * 64):
        assert phrase in markdown
    assert all(f"]({name})" in markdown for key, name in outputs.items() if key.endswith("png"))
    assert json.dumps(report, sort_keys=True, allow_nan=False) == json.dumps(original, sort_keys=True, allow_nan=False)
    assert not plt.get_fignums()


@pytest.mark.parametrize("include_probe", [False, True])
def test_report_handles_constant_reference_single_field_and_optional_probe_law(tmp_path, include_probe):
    report = _report(count=1, constant=True)
    if not include_probe:
        report["probe_law"] = None
    outputs = write_reference_validation_report(report, tmp_path)
    assert ("probe_law_png" in outputs) is include_probe
    markdown = (tmp_path / outputs["markdown"]).read_text(encoding="utf-8")
    assert ("No retained-law probe comparison was supplied" in markdown) is not include_probe
    assert "Generated field count: **1**" in markdown
    assert not plt.get_fignums()


def test_report_rejects_unsupported_schema_before_creating_files(tmp_path):
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError, match="schema_version"):
        write_reference_validation_report({"schema_version": 2}, destination)
    assert not destination.exists()
