"""End-to-end density estimates against an independent analytic log-field model."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

pytest.importorskip("gstools")

from subsurface_uq.experiments.conditioning_temperature import load_config, run_study


def _configuration(tmp_path, **updates):
    row,col = np.indices((7,8))
    reference = 10.**(-10+.01*row+.02*col)
    np.save(tmp_path/"reference.npy",reference)
    base = {"references":{"RUN_1":"reference.npy"},
        "covariance":{"family":"exponential_radial","ell_row_m":8.,"ell_col_m":6.,"sigma_R":.1},
        "cell_size_m":5.,"location_priors":{"uniform":{"kind":"uniform"}},
        "n_measurements":3,"n_designs":2,"n_samples":4,"seed":17,"workers":1,
        "diagnostic_factor":1,"max_lag_cells":2,"retained_sample_indices":[0]}
    (tmp_path/"conditioning.yaml").write_text(yaml.safe_dump(base))
    (tmp_path/"model.yaml").write_text("test model context\n")
    value = {"conditioning_config":"conditioning.yaml","model_config":"model.yaml",
        "measurement_counts":[0,1,3],"sample_budgets":[2,4],"risk_thresholds_C":[.1,.3],
        "quantile_chunk_rows":2,"retain_temperature_ensemble":True}
    value.update(updates)
    source = tmp_path/"study.yaml"
    source.write_text(yaml.safe_dump(value))
    return source,reference


def _factory(reference, calls):
    def factory(*args, **kwargs):
        def predict(field):
            calls.append(np.array(field,copy=True))
            return (np.log10(field)+10.).astype(np.float32)
        return SimpleNamespace(input_shape=reference.shape,cell_size_m=5.,background_temperature=0.,
            receptors=((2,3),),mean_anomaly_roi=(0,7,0,8),
            metadata={"kind":"analytic_test_log_field_model"},predict_temperature=predict,
            normalization_diagnostics=lambda field:{"test_model":True})
    return factory


def test_nested_density_propagation_and_statistics_match_saved_model_outputs(tmp_path):
    source,reference = _configuration(tmp_path)
    calls=[]
    output=tmp_path/"output"
    result = run_study(source,output,model_factory=_factory(reference,calls))
    assert len(calls)==25  # one nominal + 2 designs * 3 measurement counts * 4 draws
    assert result["manifest"]["status"]=="complete"
    assert result["manifest"]["reference_pooling"] is None
    assert result["manifest"]["sample_filtering"] is None
    for design in range(2):
        parent=output/f"RUN_1/uniform/design-{design:03d}"
        with (parent/"maximum_measurements.csv").open() as handle:
            maximum=list(csv.DictReader(handle))
        for count in [0,1,3]:
            directory=parent/f"n-{count:04d}"
            record=json.loads((directory/"summary.json").read_text())
            temperatures=np.load(directory/"temperature_ensemble.npy")
            qoi=np.load(directory/"qoi_samples.npy")
            np.testing.assert_allclose(np.load(directory/"temperature_mean.npy"),temperatures.mean(axis=0),atol=1e-7)
            np.testing.assert_allclose(np.load(directory/"temperature_std.npy"),temperatures.std(axis=0,ddof=1),atol=1e-7)
            np.testing.assert_allclose(np.load(directory/"temperature_q95.npy"),np.quantile(temperatures,.95,axis=0),atol=1e-7)
            np.testing.assert_allclose(qoi[:,0],temperatures[:,2,3])
            assert record["qoi_checkpoints"]["4"]["receptor_001"]["variance"]==pytest.approx(qoi[:,0].var(ddof=1))
            assert (directory/"temperature_uncertainty.png").stat().st_size>1000
            assert (directory/"sample-0000.png").stat().st_size>1000
            if count:
                with (directory/"measurements.csv").open() as handle:
                    assert list(csv.DictReader(handle))==maximum[:count]
                saved=np.load(directory/"permeability-0000.npy")
                cells=np.array([[int(row["row"]),int(row["column"])] for row in maximum[:count]])
                np.testing.assert_allclose(saved[tuple(cells.T)],reference[tuple(cells.T)],rtol=1e-7)
        seeds=[json.loads((parent/f"n-{n:04d}/summary.json").read_text())["records"][0]["seed"] for n in [0,1,3]]
        assert len(set(seeds))==1
    for row in result["rows"]:
        group=[c for c in result["cases"] if c["n_measurements"]==row["n_measurements"]]
        if row["target"]=="receptor_001":
            values=[c["qoi_checkpoints"][str(row["n_samples"])]["receptor_001"]["variance"] for c in group]
            assert row["expected_conditional_variance"]["mean"]==pytest.approx(np.mean(values))
        if row["n_measurements"]==0:
            assert row["paired_variance_reduction_from_zero"]["mean"]==0
    with pytest.raises(ValueError,match="already exists"):
        run_study(source,output,model_factory=_factory(reference,[]))


def test_noisy_readings_are_frozen_across_counts_and_outputs_are_reproducible(tmp_path):
    source,reference=_configuration(tmp_path,retain_temperature_ensemble=False)
    base=yaml.safe_load((tmp_path/"conditioning.yaml").read_text())
    base.update(measurement_mode="noisy",observation_std_log10_k=.03,mean_mode="global_log_mean",n_designs=1)
    (tmp_path/"conditioning.yaml").write_text(yaml.safe_dump(base))
    for name in ("first","second"):
        run_study(source,tmp_path/name,model_factory=_factory(reference,[]))
    first=tmp_path/"first/RUN_1/uniform/design-000"
    second=tmp_path/"second/RUN_1/uniform/design-000"
    for count in [0,1,3]:
        left,right=first/f"n-{count:04d}",second/f"n-{count:04d}"
        np.testing.assert_array_equal(np.load(left/"qoi_samples.npy"),np.load(right/"qoi_samples.npy"))
        assert not (left/"temperature_ensemble.npy").exists()
    with (first/"maximum_measurements.csv").open() as handle:
        maximum=list(csv.DictReader(handle))
    with (first/"n-0001/measurements.csv").open() as handle:
        assert list(csv.DictReader(handle))==maximum[:1]
    summary=json.loads((tmp_path/"first/density_summary.json").read_text())
    assert all(row["expected_conditional_variance"]["design_estimate_sem"] is None for row in summary["rows"])


@pytest.mark.parametrize("updates,reason",[
    ({"measurement_counts":[1,3]},"include zero"),
    ({"measurement_counts":[0,3,1]},"increasing"),
    ({"measurement_counts":[0,1,2]},"largest measurement"),
    ({"sample_budgets":[2,3]},"largest sample"),
    ({"sample_budgets":[True,4]},"integer"),
    ({"risk_thresholds_C":[-.1]},"positive"),
    ({"threshold_comparison":"<"},"comparison"),
    ({"retain_temperature_ensemble":"false"},"boolean"),
    ({"unknown":1},"known"),
])
def test_configuration_rejects_conflicting_or_ambiguous_choices(tmp_path,updates,reason):
    source,_=_configuration(tmp_path,**updates)
    with pytest.raises(ValueError,match=reason):
        load_config(source)


def test_grid_mismatch_fails_before_output_creation(tmp_path):
    source,reference=_configuration(tmp_path)
    factory=_factory(reference,[])
    def wrong(*args,**kwargs):
        model=factory(*args,**kwargs)
        model.input_shape=(8,8)
        return model
    with pytest.raises(ValueError,match="grid/spacing"):
        run_study(source,tmp_path/"output",model_factory=wrong)
    assert not (tmp_path/"output").exists()
