from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from subsurface_uq.experiments.covariance_reconstruction import load_config, quantile_interval, run, stopping_table
from subsurface_uq.sampling.covariance_reconstruction import OrdinaryKrigingSystem, covariance_locations
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec


def _spec():
    return GSToolsReferenceSpec("exponential_radial", 3., 2., sigma_R=.2)


def test_location_design_is_value_free_amplitude_free_and_nested():
    first = covariance_locations((12, 13), _spec(), cell_size_m=1., n_measurements=15, seed=42)
    second = covariance_locations((12, 13), replace(_spec(), sigma_R=20.), cell_size_m=1., n_measurements=6, seed=42)
    np.testing.assert_array_equal(first.indices[:6], second.indices)
    np.testing.assert_allclose(first.probabilities[:6], second.probabilities)
    assert first.probabilities[0] == pytest.approx(1/(12*13))
    assert first.metadata["reference_mean_used"] is False
    assert first.metadata["reference_values_used"] is False
    offsets = first.indices[:, None]-first.indices[None, :]
    np.testing.assert_allclose(first.cholesky@first.cholesky.T,
        _spec().correlation(offsets[..., 0], offsets[..., 1]), atol=1.e-12)


def test_selected_probabilities_match_dense_ordinary_variances():
    shape = (7, 8)
    design = covariance_locations(shape, _spec(), cell_size_m=1., n_measurements=9, seed=4)
    queries = np.column_stack(np.unravel_index(np.arange(np.prod(shape)), shape))
    for n in range(1, 9):
        system = OrdinaryKrigingSystem(shape, _spec(), design.indices[:n], np.zeros(n), cell_size_m=1.)
        _, variance = system.moments(system.features(queries))
        variance[np.ravel_multi_index(design.indices[:n].T, shape)] = 0.
        chosen = np.ravel_multi_index(design.indices[n], shape)
        assert design.probabilities[n] == pytest.approx(variance[chosen]/variance.sum())
        assert design.preselection_variance[n] == pytest.approx(variance[chosen])


@pytest.mark.parametrize("family", ["gaussian", "matern32_radial", "exponential_radial"])
def test_ordinary_mean_and_variance_agree_with_gstools(family):
    gs = pytest.importorskip("gstools")
    spec = replace(_spec(), family=family)
    indices = np.array([[0, 0], [2, 4], [5, 1]])
    values = np.array([-9.4, -9., -9.2])
    queries = np.array([[0, 0], [1, 3], [4, 6]])
    system = OrdinaryKrigingSystem((7, 8), spec, indices, values, cell_size_m=1.)
    mean, unit_variance = system.moments(system.features(queries))
    krige = gs.krige.Ordinary(spec.gstools_model(), [indices[:, 1], indices[:, 0]], values,
                             exact=True, pseudo_inv=False)
    expected_mean, expected_variance = krige([queries[:, 1], queries[:, 0]], mesh_type="unstructured")
    np.testing.assert_allclose(mean, expected_mean, atol=2.e-12)
    np.testing.assert_allclose(unit_variance, expected_variance, atol=2.e-12)


def test_matheron_ordinary_draws_keep_mean_uncertainty_and_exact_pins():
    spec = _spec()
    points = np.array([[0, 0], [0, 3], [3, 0], [3, 3], [1, 2]])
    offsets = points[:, None]-points[None, :]
    correlation = spec.correlation(offsets[..., 0], offsets[..., 1])
    prior = np.random.default_rng(21).multivariate_normal(np.zeros(5), correlation, size=30000).T
    system = OrdinaryKrigingSystem((4, 4), spec, points[:2], [-9., -9.5], cell_size_m=1.)
    f = system.features(points)
    mean, variance, draws = system.unit_draws(f, prior, prior[:2])
    np.testing.assert_allclose(draws[:2], 0., atol=2.e-15)
    np.testing.assert_allclose(mean[:2], system.readings, atol=2.e-14)
    r = 1-f@system.h
    expected_cov = correlation-f@f.T+np.outer(r,r)/system.precision
    np.testing.assert_allclose(np.cov(draws), expected_cov, atol=.025, rtol=.04)
    np.testing.assert_allclose(variance, np.diag(expected_cov), atol=1.e-14)
    assert np.any(variance[2:] > 1-np.sum(f[2:]**2, axis=1)+.01)
    shifted = OrdinaryKrigingSystem((4, 4), spec, points[:2], system.readings+3., cell_size_m=1.)
    shifted_mean, shifted_variance = shifted.moments(f)
    np.testing.assert_allclose(shifted_mean, mean+3.)
    np.testing.assert_allclose(shifted_variance, variance)


def test_zero_observations_are_not_given_a_reference_mean():
    with pytest.raises(ValueError, match="at least one"):
        OrdinaryKrigingSystem((4,4), _spec(), np.empty((0,2),dtype=int), [], cell_size_m=1.)


def test_smooth_kernel_reports_numerical_budget_limit_without_nugget():
    spec = GSToolsReferenceSpec("gaussian", 1000., 1000.)
    design = covariance_locations((8,8),spec,cell_size_m=1.,n_measurements=50,seed=4,pivot_floor=1.e-7)
    assert len(design.indices) < 50
    assert design.metadata["stop_reason"] == "numerical_innovation_variance_floor"
    assert design.metadata["added_nugget"] is None


def test_quantile_interval_and_unreached_counts():
    interval = quantile_interval(np.arange(128),.95)
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
    assert quantile_interval(np.arange(4),.95)["upper"] is None
    cases=[]
    for n,q in [(1,.3),(4,.1),(8,.2)]:
        cases.append({"sigma_log10":.2,"n_measurements":n,"reference_rmse":{"estimate":q,"upper":q+.01},
            "conditional_spread":{"estimate":q,"upper":q+.01},"reference_rmse_markov_bound":2*q,"conditional_spread_markov_bound":2*q})
    table=stopping_table(cases,[.05,.15],.95)
    assert table[0]["reference_rmse_first_tested_count"] is None
    assert table[1]["reference_rmse_first_tested_count"] == 4
    assert table[1]["reference_rmse_markov_first_tested_count"] is None


def _configuration(tmp_path):
    ref=10.**(-9.+np.arange(56).reshape(7,8)/200.)
    np.save(tmp_path/"ref.npy",ref)
    cfg={"references":{"RUN_1":"ref.npy"},"covariance":{"family":"exponential_radial","ell_row_m":3.,"ell_col_m":2.},
        "covariance_by_reference":{},"native_cell_size_m":1.,"evaluation_stride_cells":1,"candidate_stride_cells":1,
        "measurement_counts":[1,3,6],"n_designs":1,"n_samples":12,"sigma_log10":[.1,.2],
        "error_limits_log10_rms":[.05,.5],"probability":.95,"seed":5,"workers":1,"chunk_size":30,"pivot_floor":1.e-7}
    path=tmp_path/"config.yaml";path.write_text(yaml.safe_dump(cfg))
    return path,cfg


def test_study_outputs_preserve_exact_data_and_do_not_pool_or_leave_caches(tmp_path):
    path,_=_configuration(tmp_path)
    output=tmp_path/"study"
    manifest=run(path,output)
    assert manifest["status"]=="completed"
    assert manifest["reference_mean_used"] is False
    assert manifest["scenario_weights"] is None
    assert not list(output.rglob("temporary*"))
    table=json.loads((output/"stopping_counts.json").read_text())
    assert len(table)==4
    dr=output/"RUN_1"/"design-000"
    readings=np.genfromtxt(dr/"measurements.csv",delimiter=",",names=True)
    r,c=readings["row"].astype(int),readings["column"].astype(int)
    generated=np.load(dr/"n-0006"/"fields_sigma-0.200.npy")
    np.testing.assert_allclose(generated[:,r,c],np.tile(readings["log10_K"],(2,1)),atol=1.e-10)
    with pytest.raises(ValueError,match="already exists"):
        run(path,output)


@pytest.mark.parametrize("key,value",[("measurement_counts",[0,4]),("sigma_log10",[0]),("mean_mode","reference_median"),
    ("n_samples",1),("probability",1.),("native_cell_size_m",0)])
def test_configuration_rejects_inconsistent_choices(tmp_path,key,value):
    path,cfg=_configuration(tmp_path);cfg[key]=value;path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        load_config(path)
