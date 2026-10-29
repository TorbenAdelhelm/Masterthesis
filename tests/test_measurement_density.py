import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm
import yaml

from subsurface_uq.io import sha256_file
from subsurface_uq.experiments.measurement_density import load_config, run
from subsurface_uq.sampling.measurement_density import coverage_locations, gaussian_crps, native_ensembles
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from subsurface_uq.sampling.covariance_reconstruction import OrdinaryKrigingSystem
from subsurface_uq.statistics.regional_temperature import empirical_tail_risk, regional_deviation_samples


@pytest.mark.parametrize("kind",["balanced","clustered"])
def test_locations_are_geometry_only_and_keep_nested_prefixes(kind):
    full,meta=coverage_locations((32,35),30,stride=2,seed=17,kind=kind)
    smaller,_=coverage_locations((32,35),10,stride=2,seed=17,kind=kind)
    # Weighted choice without replacement is prefix-stable for the same seed.
    np.testing.assert_array_equal(full[:10],smaller)
    assert len(np.unique(full,axis=0))==30
    assert not meta["reference_mean_used"] and not meta["reference_values_used"]


def test_normal_crps_matches_analytic_special_cases_and_numerical_expectations():
    assert gaussian_crps(0.,1.,0.)==pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    assert gaussian_crps(2.,0.,-1.)==3.
    draws=np.random.default_rng(4).normal(1.,2.,size=(2,200000))
    empirical=np.mean(np.abs(draws[0]-3.))-.5*np.mean(np.abs(draws[0]-draws[1]))
    assert gaussian_crps(1.,2.,3.)==pytest.approx(empirical,abs=.012)


def test_atom_aware_expected_shortfall_and_regional_maximum_order():
    risk=empirical_tail_risk([0.,0.,0.,10.],.5)
    assert risk["VaR"]==0 and risk["expected_shortfall"]==5.
    reference=np.zeros((3,3))
    fields=np.zeros((4,3,3));fields[0,0,0]=10;fields[1,1,1]=8
    regional=regional_deviation_samples(fields,reference,(0,3,0,3))
    np.testing.assert_array_equal(regional["maximum_absolute_deviation_C"],[10,8,0,0])
    with pytest.raises(ValueError):regional_deviation_samples(fields,reference,(0,4,0,3))


def test_native_block_conditioning_agrees_with_direct_ordinary_realizations(tmp_path):
    shape=(9,8);spec=GSToolsReferenceSpec("matern32_radial",9.,7.,sigma_R=.2)
    reference=10.**(-9.+np.arange(72).reshape(shape)/200.)
    indices=np.array([[0,0],[3,5],[8,7],[1,6]])
    prior=GSToolsReferenceFFTSampler(GSToolsReferenceSpec("matern32_radial",9.,7.,sigma_R=1.),shape,cell_size_m=5.,workers=1)
    draws=np.array([prior.sample_residual(s) for s in range(5)])
    native_ensembles(reference,spec,indices,[2,4],draws,tmp_path,cell_size_m=5.,chunk_size=13)
    queries=np.column_stack(np.unravel_index(np.arange(72),shape))
    for n in [2,4]:
        readings=np.log10(reference[tuple(indices[:n].T)])
        system=OrdinaryKrigingSystem(shape,spec,indices[:n],readings,cell_size_m=5.)
        mean,_,residual=system.unit_draws(system.features(queries),draws.reshape(5,-1).T,draws[:,indices[:n,0],indices[:n,1]].T)
        expected=np.power(10.,mean[:,None]+.2*residual).T.reshape(5,*shape).astype(np.float32)
        actual=np.load(tmp_path/f"n-{n:04d}/permeability_ensemble.npy")
        np.testing.assert_allclose(actual,expected,rtol=2.e-7)
    assert not list(tmp_path.glob("temporary*"))


class FakeModel:
    input_shape=(9,9)
    cell_size_m=5.
    background_temperature=10.6
    receptors=((2,2),(3,3))
    mean_anomaly_roi=(1,5,1,5)
    metadata={"files_sha256":{},"test_model":True}
    calls=0

    def predict_temperature(self,field):
        self.calls+=1
        log=np.log10(field[1:8,1:8].astype(float))+9.
        return (10.6+.2*log+.1*log**2).astype(np.float32)

    def normalization_diagnostics(self,field):
        return {"minimum":float(field.min()),"maximum":float(field.max()),"interpretation":"test"}


def _config(tmp_path):
    path=tmp_path/"reference.npy"
    reference=10.**(-9.+np.arange(81).reshape(9,9)/300.)
    np.save(path,reference)
    families=["gaussian","matern32_radial","exponential_radial"]
    fits={"references":{"RUN_1":{"reference_sha256":sha256_file(path),"fits":[
        {"family":f,"fit_max_lag_m":2560.,"ell_row_m":15.,"ell_col_m":12.} for f in families]}}}
    (tmp_path/"fits.json").write_text(json.dumps(fits))
    cfg={"references":{"RUN_1":"reference.npy"},"kernel_fits":"fits.json","fit_max_lag_m":2560.,"families":families,
        "sigma_log10":.2,"screen_sigma_log10":[.1,.2],"measurement_counts":[2,6],"coverage_scenarios":["balanced","clustered"],
        "cluster_scale_fraction":.1,"candidate_stride_cells":2,"screen_stride_cells":1,"cell_size_m":5.,"n_samples":4,
        "sample_budgets":[2,4],"seed":11,"workers":1,"chunk_size":23,"model_config":"unused.yaml","model_asset_root":".",
        "torch_threads":1,"risk_thresholds_C":[.1,1.],"deviation_limits_C":[.05,.1,.5],"quantile_chunk_rows":2}
    source=tmp_path/"config.yaml";source.write_text(yaml.safe_dump(cfg))
    return source,cfg


def test_complete_workflow_and_verified_prediction_resume(tmp_path):
    source,_=_config(tmp_path)
    model=FakeModel();factory=lambda *a,**kw:model
    output=tmp_path/"study"
    result=run(source,output,model_factory=factory)
    assert result["status"]=="complete"
    assert model.calls==17
    assert result["generated_evaluations"]==16
    assert not list(output.rglob("temporary*"))
    for coverage in ["balanced","clustered"]:
        for n in [2,6]:
            summary=json.loads((output/"RUN_1"/coverage/f"n-{n:04d}/temperature_summary.json").read_text())
            assert summary["input_summary"]["reference_mean_used"] is False
            assert summary["regional_risk"]["maximum_absolute_deviation"]["VaR_ci95"]["upper"] is None
    # Completed caches are validated and reused, not resampled/requeried.
    run(source,output,stage="temperatures",model_factory=factory)
    assert model.calls==17
    case=output/"RUN_1/balanced/n-0002/temperature_ensemble.npy"
    array=np.load(case,mmap_mode="r+");array[0,0,0]+=1;array.flush();del array
    with pytest.raises(ValueError,match="cache changed"):
        run(source,output,stage="temperatures",model_factory=factory)


@pytest.mark.parametrize("key,value",[("measurement_counts",[0,8]),("sigma_log10",0),("screen_stride_cells",3),
    ("n_samples",3),("families",["unknown"]),("coverage_scenarios",["unknown"])])
def test_invalid_configuration_fails_before_output_creation(tmp_path,key,value):
    source,cfg=_config(tmp_path);cfg[key]=value;source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(source)
