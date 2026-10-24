from pathlib import Path
import json
import numpy as np
import pytest
import yaml

pytest.importorskip("gstools")

from subsurface_uq.experiments.conditioning_deviation import cached_prior, conditional_blocks, measure_case, pairwise_rms, run_study
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec, GSToolsReferenceFFTSampler
from subsurface_uq.sampling.reference_conditioning import ConditionalReferenceFFTSampler, draw_reference_measurements


@pytest.mark.parametrize("family",["gaussian","matern32_radial","exponential_radial"])
@pytest.mark.parametrize("mode",["empty","exact","noisy"])
def test_blocked_cached_law_matches_scalar_seed_and_dense_moments(tmp_path,family,mode):
    row,col=np.indices((9,8))
    ref=10.**(-10+.02*row+.03*col)
    spec=GSToolsReferenceSpec(family,7.,5.,sigma_R=.15)
    prior=GSToolsReferenceFFTSampler(spec,ref.shape,cell_size_m=5.)
    observations=None if mode=="empty" else draw_reference_measurements(ref,covariance=spec,n_measurements=4,
        cell_size_m=5.,location_seed=21,observation_seed=31,measurement_mode=mode,
        observation_std_log10_k=.04 if mode=="noisy" else 0.)
    sampler=ConditionalReferenceFFTSampler(ref,spec,observations,cell_size_m=5.,prior_sampler=prior,
        mean_mode="global_log_mean" if mode=="noisy" else "reference_median",chunk_size=13)
    seeds=[51,63,79,88,92,101,199,209]
    cached=cached_prior(prior,seeds,tmp_path/"prior.npy")
    draws=np.empty((ref.size,len(seeds)))
    means,variances=np.empty(ref.size),np.empty(ref.size)
    for start,values,mean,variance in conditional_blocks(sampler,cached,seeds):
        draws[start:start+len(mean)]=values
        means[start:start+len(mean)]=mean
        variances[start:start+len(mean)]=variance
    expected=np.stack([sampler.sample_log10(seed) for seed in seeds])
    np.testing.assert_allclose(draws.T.reshape(expected.shape),expected,rtol=0,atol=1e-10)
    mean,variance=sampler.posterior_moments()
    np.testing.assert_allclose(means.reshape(ref.shape),mean,atol=1e-12)
    np.testing.assert_allclose(variances.reshape(ref.shape),variance,atol=1e-12)
    result=measure_case(sampler,cached,seeds,tmp_path/"measured",[0,1],[4,8])
    residuals=expected-np.log10(ref)
    pair=np.load(tmp_path/"measured/pairwise_log10_rms.npy")
    for a,b in [(0,1),(2,4),(4,7)]:
        assert pair[a,b]==pytest.approx(np.sqrt(np.mean((expected[a]-expected[b])**2)),abs=1e-10)
    observed=result["checkpoints"][-1]
    assert observed["reference_log10_rms"]==pytest.approx(np.sqrt(np.mean(residuals**2)))
    assert observed["relative_reference_rms_percent"]==pytest.approx(100*np.sqrt(np.mean((10.**residuals-1)**2)))
    assert result["analytic_reference_log10_rms"]**2==pytest.approx(np.mean(variance+(mean-np.log10(ref))**2))
    assert result["analytic_between_sample_log10_rms"]**2==pytest.approx(2*np.mean(variance))
    np.testing.assert_allclose(np.load(tmp_path/"measured/ensemble_log10_std.npy"),expected.std(axis=0,ddof=1),rtol=1e-6,atol=1e-7)
    if mode=="empty":
        assert result["analytic_reference_log10_rms"]==pytest.approx(.15)
        expected_relative=np.sqrt(np.exp(np.log(10.)**2*.15**2)*np.expm1(np.log(10.)**2*.15**2)+np.expm1(.5*np.log(10.)**2*.15**2)**2)
        assert result["analytic_relative_reference_rms_percent"]==pytest.approx(100*expected_relative)


def test_native_study_preserves_nested_observations_and_dependent_pair_identity(tmp_path):
    ref=10.**(-10+.03*np.arange(72).reshape(9,8))
    np.save(tmp_path/"ref.npy",ref)
    base={"references":{"RUN_1":"ref.npy"},"cell_size_m":5.,
        "covariance":{"family":"exponential_radial","ell_row_m":7.,"ell_col_m":5.,"sigma_R":.1},
        "location_priors":{"variance":{"kind":"variance"}},"n_measurements":4,"n_designs":1,"n_samples":8,
        "retained_sample_indices":[0,1],"diagnostic_factor":1,"max_lag_cells":2,"chunk_size":13}
    (tmp_path/"base.yaml").write_text(yaml.safe_dump(base))
    (tmp_path/"study.yaml").write_text(yaml.safe_dump({"conditioning_config":"base.yaml","measurement_counts":[0,1,4],"sample_budgets":[4,8]}))
    result=run_study(tmp_path/"study.yaml",tmp_path/"output")
    assert result["status"]=="complete" and result["generated_fields"]==24
    assert result["independent_prior_draws"]==8
    assert not list((tmp_path/"output").rglob("temporary_prior.npy"))
    assert (tmp_path/"output/RUN_1.variance.deviations.png").stat().st_size>1000
    for path in (tmp_path/"output").glob("RUN_1/variance/design-000/n-*"):
        pair=np.load(path/"pairwise_log10_rms.npy")
        empirical=np.load(path/"ensemble_log10_std.npy")
        assert np.mean(pair[np.triu_indices(8,1)]**2)==pytest.approx(2*np.mean(empirical.astype(float)**2),rel=1e-6)
        assert pair.shape==(8,8)
    with pytest.raises(ValueError,match="fresh"):
        run_study(tmp_path/"study.yaml",tmp_path/"output")
