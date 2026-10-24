import numpy as np
import pytest

from subsurface_uq.statistics.temperature_risk import receptor_risk, summarize_risk_store, validate_output_grid


def test_declared_threshold_boundary_and_undefined_conditional_severity():
    boundary = np.array([0.,.1,.2],dtype=np.float32)
    strict = receptor_risk(boundary,background=0.,threshold=.1,comparison=">")
    inclusive = receptor_risk(boundary,background=0.,threshold=.1,comparison=">=")
    assert strict["exceedance_count"]==1
    assert inclusive["exceedance_count"]==2
    assert strict["expected_excess_C"]==pytest.approx(.1/3)
    assert strict["conditional_excess_C"]==pytest.approx(.1)
    no_event = receptor_risk(np.zeros(4),background=0.,threshold=.1)
    assert no_event["conditional_excess_C"] is None
    assert no_event["probability_ci95"][1]>0
    assert no_event["expected_excess_C"]==0


def test_chunked_risk_matches_direct_physical_area_and_severity(tmp_path):
    fields = np.array([[[0,1],[2,3]],[[2,0],[4,1]],[[1,3],[0,2]]],dtype=np.float32)
    np.save(tmp_path/"temperatures.npy",fields)
    result = summarize_risk_store(tmp_path/"temperatures.npy",tmp_path/"risk",thresholds=[1.,2.],
        background=0.,receptors=[(0,1)],cell_size_m=5.,budgets=[2,3],comparison=">",chunk_rows=1)
    samples = np.load(tmp_path/"risk/risk_samples.npz")
    for i,threshold in enumerate([1.,2.]):
        mask = fields>threshold
        np.testing.assert_allclose(np.load(tmp_path/"risk"/result["thresholds"][i]["probability_map"]),mask.mean(axis=0))
        np.testing.assert_allclose(np.load(tmp_path/"risk"/result["thresholds"][i]["expected_excess_map"]),np.maximum(fields-threshold,0).mean(axis=0))
        np.testing.assert_array_equal(samples["area_m2"][:,i],mask.sum(axis=(1,2))*25)
        risk = result["thresholds"][i]["checkpoints"][-1]["receptors"]["receptor_001"]
        assert risk["probability"]==pytest.approx(mask[:,0,1].mean())
    with pytest.raises(ValueError,match="ROI"):
        validate_output_grid((2,2),[],(0,3,0,1))
