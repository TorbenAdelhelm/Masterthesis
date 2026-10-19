import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip("gstools")

from subsurface_uq.io import sha256_file


def test_native_input_study_provenance_and_resume(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/reference_pilot/run_matched_kernel_study.py"
    spec = importlib.util.spec_from_file_location("matched_study", path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    y, x = np.mgrid[:24, :24]
    references = {}
    for run in ("RUN_1", "RUN_2", "RUN_3"):
        reference = tmp_path / f"{run}.npy"
        np.save(reference, (1.e-10 * 10**(.2*np.sin(y/5) + .1*np.cos(x/4))).astype(np.float32))
        fits = [{"family": family, "fit_max_lag_m": window,
                 "ell_row_m": 20. if window == 1280 else 25., "ell_col_m": 15. if window == 1280 else 18.}
                for family in ("gaussian", "matern32_radial", "exponential_radial") for window in (1280., 2560.)]
        references[run] = {"reference_path": str(reference), "reference_sha256": sha256_file(reference), "fits": fits}
    source = tmp_path / "fits.json"
    source.write_text(json.dumps({"references": references}))
    root = tmp_path / "study"
    study.prepare(root, source, 4, 4)
    manifest = study.load_study(root)
    cases = manifest["context"]["cases"]
    assert len(cases) == len({c["scenario_id"] for c in cases}) == 18
    sid = cases[0]["scenario_id"]
    study.input_stage(root, [sid])
    study.input_stage(root, [sid])  # Only completed, checksum-verified outputs reused.
    directory = root / "inputs" / sid
    report = study.read(directory / "validation.json")
    assert len(report["records"]) == 4
    assert {r["batch"] for r in report["records"]} == {0, 1}
    assert report["sample_filtering"] is None
    assert len({r["physical_field_sha256"] for r in report["records"]}) == 4
    assert np.isfinite(report["probe_law"]["covariance_relative_frobenius_error"])
    with (directory / "sample-000.coarse.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="completed scenario output changed"):
        study.input_stage(root, [sid])
