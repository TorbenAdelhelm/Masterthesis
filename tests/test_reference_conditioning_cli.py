import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("gstools")

from subsurface_uq.experiments.reference_conditioning import load_config, main


def _config(tmp_path, **updates):
    row, col = np.mgrid[:8,:7]
    reference = np.power(10.,-10+.15*np.sin(row/2)+.1*np.cos(col/2)).astype(np.float32)
    np.save(tmp_path/"reference.npy",reference)
    value={"references":{"RUN_1":"reference.npy"},
           "covariance":{"family":"exponential_radial","ell_row_m":2.,"ell_col_m":1.5,"sigma_R":.08},
           "cell_size_m":1.,"location_priors":{"uniform":{"kind":"uniform"},"variance":{"kind":"variance"}},
           "n_measurements":3,"n_designs":2,"n_samples":3,"seed":17,"workers":1,
           "diagnostic_factor":1,"max_lag_cells":2,"retained_sample_indices":[0]}
    value.update(updates)
    path=tmp_path/"study.yaml"
    path.write_text(yaml.safe_dump(value),encoding="utf-8")
    return path,reference


def test_cli_keeps_designs_separate_and_reproduces_frozen_readings(tmp_path):
    path,reference=_config(tmp_path)
    output=tmp_path/"output"
    assert main(["--config",str(path),"--output-dir",str(output)])==0
    manifest=json.loads((output/"manifest.json").read_text())
    assert manifest["status"]=="complete"
    assert manifest["synthetic_observations"] is True
    assert manifest["reference_pooling"] is None
    assert len(manifest["implementation_sha256"])==3
    summary=json.loads((output/"summary.json").read_text())["references"]
    assert set(summary)=={"RUN_1"}
    assert set(summary["RUN_1"])=={"uniform","variance"}
    for prior in summary["RUN_1"].values():
        assert prior["design_count"]==2
        assert prior["between_design_mean_variance_log10"]==pytest.approx(0,abs=1e-20)
        assert prior["design_mixture_mean_variance_log10"]==pytest.approx(prior["within_design_mean_variance_log10"])
    designs=sorted((output/"RUN_1").glob("*/*/design.json"))
    assert len(designs)==4
    for file in designs:
        record=json.loads(file.read_text())
        assert len(record["records"])==3
        assert record["sample_filtering"] is None
        assert (file.parent/"conditional_fields.png").stat().st_size>1000
        diagnostic=json.loads((file.parent/"reference_diagnostics.json").read_text())
        assert diagnostic["generated_field_count"]==3
        assert diagnostic["sample_filtering"] is None
        assert (file.parent/"sample-0000.npy").is_file()
        assert not (file.parent/"sample-0001.npy").exists()
        assert np.load(file.parent/"sample-0000.npy").shape==reference.shape
        np.testing.assert_allclose(np.load(file.parent/"posterior_log10_std.npy")**2,
                                   np.load(file.parent/"posterior_log10_variance.npy"),rtol=2e-7,atol=1e-12)
    rerun=tmp_path/"rerun"
    main(["--config",str(path),"--output-dir",str(rerun)])
    for original in designs:
        repeated=rerun/original.relative_to(output)
        left,right=(json.loads(p.read_text()) for p in (original,repeated))
        assert left["design_id"]==right["design_id"]
        assert left["records"][0]["sha256"]==right["records"][0]["sha256"]
        assert (original.parent/"measurements.csv").read_bytes()==(repeated.parent/"measurements.csv").read_bytes()
    with pytest.raises(ValueError,match="already exists"):
        main(["--config",str(path),"--output-dir",str(output)])


def test_noisy_reconstruction_reports_nonzero_between_design_component(tmp_path):
    path,_=_config(tmp_path,mean_mode="global_log_mean",measurement_mode="noisy",
                   observation_std_log10_k=.03,location_priors={"uniform":{"kind":"uniform"}},n_samples=2)
    output=tmp_path/"output"
    main(["--config",str(path),"--output-dir",str(output)])
    summary=json.loads((output/"summary.json").read_text())["references"]["RUN_1"]["uniform"]
    assert summary["between_design_mean_variance_log10"]>0
    assert summary["design_mixture_mean_variance_log10"]==pytest.approx(
        summary["within_design_mean_variance_log10"]+summary["between_design_mean_variance_log10"])
    within=np.load(output/"RUN_1/uniform/within_design_variance_log10.npy")
    between=np.load(output/"RUN_1/uniform/between_design_variance_log10.npy")
    mixture=np.load(output/"RUN_1/uniform/design_mixture_variance_log10.npy")
    np.testing.assert_allclose(mixture,within+between,rtol=2e-7,atol=1e-10)


@pytest.mark.parametrize("updates,reason",[
    ({"unsupported_assumption":1},"unknown"),
    ({"measurement_mode":"exact","observation_std_log10_k":.1},"zero noise"),
    ({"measurement_mode":"noisy","observation_std_log10_k":0},"positive noise"),
    ({"permeability_units":"m/s"},"m2"),
    ({"references":{"../RUN_1":"reference.npy"}},"path separators"),
])
def test_cli_rejects_ambiguous_configuration_before_creating_outputs(tmp_path,updates,reason):
    path,_=_config(tmp_path,**updates)
    with pytest.raises(ValueError,match=reason):
        load_config(path)
