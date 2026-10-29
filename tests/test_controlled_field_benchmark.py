import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import cho_solve
import yaml

from subsurface_uq.experiments.controlled_field_benchmark import load_config, run, seed_for, _locations, _probes
from subsurface_uq.sampling.controlled_fields import KnownMeanConditioner, physical_permeability
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceFFTSampler, GSToolsReferenceSpec
from subsurface_uq.statistics.calibration import ensemble_truth_scores, calibration_summary


def test_known_mean_joint_posterior_matches_independent_gaussian_equations():
    shape=(6,7);spec=GSToolsReferenceSpec("matern32_radial",12.,9.,sigma_R=.3)
    data=np.array([[0,0],[3,4],[5,6]])
    queries=np.array([[1,2],[2,6],[5,1]])
    system=KnownMeanConditioner(shape,spec,data,mean_log10=-9.3,cell_size_m=5.)
    locations=np.concatenate((data,queries))
    offset=(locations[:,None]-locations[None,:])*5.
    c=spec.correlation(offset[...,0],offset[...,1]);chol=np.linalg.cholesky(c)
    rng=np.random.default_rng(4)
    readings=np.array([-9.1,-9.5,-9.2])
    f=system.features(queries)
    for n in [0,1,3]:
        mean,var=system.moments(f,readings[:n],count=n)
        weights=cho_solve((chol[:n,:n],True),c[:n,3:]).T if n else np.empty((3,0))
        expected_mean=-9.3+weights@(readings[:n]+9.3)
        expected_cov=.3**2*(c[3:,3:]-weights@c[:n,3:])
        np.testing.assert_allclose(mean,expected_mean,atol=1e-14)
        np.testing.assert_allclose(var,np.diag(expected_cov),atol=1e-14)
        priors=rng.standard_normal((24000,6))@chol.T
        coefficients=np.array([system.update_coefficients(readings[:n],p[:n],count=n) for p in priors])
        draws=-9.3+.3*priors[:,3:]+coefficients@f[:,:n].T
        # Gaussian sample-mean and Wishart sample-covariance standard errors.
        assert np.all(np.abs(draws.mean(axis=0)-mean) < 6*np.sqrt(np.diag(expected_cov)/len(draws)))
        covariance_sem=np.sqrt((expected_cov**2+np.outer(np.diag(expected_cov),np.diag(expected_cov)))/(len(draws)-1))
        assert np.all(np.abs(np.cov(draws.T)-expected_cov)<6*covariance_sem)


def test_fft_matheron_full_grid_preserves_exact_data_and_native_covariance():
    shape=(5,6);spec=GSToolsReferenceSpec("exponential_radial",12.,9.,sigma_R=.2)
    data=np.array([[0,0],[3,4]])
    system=KnownMeanConditioner(shape,spec,data,mean_log10=-9.3,cell_size_m=5.)
    sampler=GSToolsReferenceFFTSampler(replace(spec,sigma_R=1.),shape,cell_size_m=5.,workers=1)
    readings=np.array([-9.1,-9.4]);q=np.array([[1,2],[4,1]])
    values=[]
    for seed in range(1300):
        logs=system.conditional_log_field(sampler.sample_residual(seed),readings,chunk_size=7)
        np.testing.assert_allclose(logs[tuple(data.T)],readings,atol=2e-14)
        values.append(logs[tuple(q.T)])
    m,v=system.moments(system.features(q),readings)
    assert np.all(np.abs(np.mean(values,axis=0)-m)<6*np.sqrt(v/len(values)))
    assert np.all(np.abs(np.var(values,axis=0,ddof=1)-v)<6*v*np.sqrt(2/(len(values)-1)))


def test_randomized_ranks_handle_atoms_and_finite_interval_target():
    rng=np.random.default_rng(17)
    scores=[ensemble_truth_scores(10.,[10.]*8,rng=rng) for _ in range(9000)]
    hist=np.bincount([s['rank'] for s in scores],minlength=9)
    assert np.max(np.abs(hist-1000))<100
    summary=calibration_summary(scores)
    assert summary['attainable_rank_coverage']==pytest.approx(7/9)
    assert summary['rank_coverage']==pytest.approx(7/9,abs=.02)
    assert summary['mean_crps']==0.
    assert len(summary['rank_histogram'])==9
    x=ensemble_truth_scores(0.,[-1.,1.],rng=rng)
    assert x['crps']==.5


class TestModel:
    __test__=False
    input_shape=(9,9);cell_size_m=5.;background_temperature=10.6
    receptors=((2,2),(3,3));mean_anomaly_roi=(1,5,1,5)
    metadata={"files_sha256":{},"kind":"nonlinear_test_model"}
    calls=0

    def predict_temperature(self,field):
        self.calls+=1
        x=np.log10(field[1:8,1:8].astype(float))+9.3
        return (10.6+.2*x+.1*x*x).astype(np.float32)

    def normalization_diagnostics(self,field):
        return {"min":float(field.min()),"max":float(field.max())}


def config(tmp_path):
    cfg=yaml.safe_load(open('configs/controlled_field.pilot.yaml',encoding='utf-8'))
    cfg.update(shape=[9,9],measurement_counts=[0,2,6],candidate_stride_cells=2,diagnostic_stride_cells=2,
               n_truths=3,n_samples=4,sample_budgets=[2,4],validation_replicates=30,validation_probes=2,
               workers=1,torch_threads=1,chunk_size=17,quantile_chunk_rows=2,model_config='model.yaml',model_asset_root='.')
    cfg['scenarios']['matern32_known']['covariance'].update(ell_row_m=15.,ell_col_m=12.)
    (tmp_path/'model.yaml').write_text('test: true')
    source=tmp_path/'config.yaml';source.write_text(yaml.safe_dump(cfg))
    return source,cfg


def structural_config(tmp_path):
    from copy import deepcopy
    from subsurface_uq.io import sha256_file
    source,cfg=config(tmp_path)
    a=deepcopy(cfg['scenarios']['matern32_known']);b=deepcopy(a)
    b['mean_log10_k']+=.12;b['covariance']['sigma_R']*=2
    cfg['scenarios']={'A':a,'B':b}
    cfg.update(n_truths=2,measurement_counts=[0,6],coverage_scenarios=['balanced'],
        structural_profile_file='profiles.json',profile_pairing_group='paired',
        paired_embedding_shape=[32,32],sensitivity_reference_profile='A')
    np.save(tmp_path/'map.npy',np.ones((9,9),dtype=np.float32)*1e-9)
    artifact={'schema_version':1,'family':'matern32_radial','native_shape':[9,9],'cell_size_m':5.,
        'profiles':{k:{'law':v} for k,v in cfg['scenarios'].items()},
        'source_maps':{'map':{'path':'map.npy','sha256':sha256_file(tmp_path/'map.npy')}}}
    (tmp_path/'profiles.json').write_text(json.dumps(artifact))
    source.write_text(yaml.safe_dump(cfg));return source,cfg


def test_structural_profiles_share_driving_noise_and_geometry_without_pooling(tmp_path):
    source,cfg=structural_config(tmp_path);model=TestModel();out=tmp_path/'paired-study'
    np.testing.assert_array_equal(_locations(cfg,'A','balanced',0)[0],_locations(cfg,'B','balanced',0)[0])
    np.testing.assert_array_equal(_probes(cfg,'A'),_probes(cfg,'B'))
    manifest=run(source,out,model_factory=lambda *a,**kw:model)
    assert manifest['profile_probability_weights'] is None
    assert model.calls==2*2*(1+4*2)==36
    for trial in range(2):
        a=np.load(out/f'A/truth-{trial:04d}/truth_log10.npy')
        b=np.load(out/f'B/truth-{trial:04d}/truth_log10.npy')
        np.testing.assert_allclose(b-(-9.3+.12),2*(a+9.3),atol=2e-14)
        for design in ['prior','balanced-000/n-0006']:
            ra=json.loads((out/f'A/truth-{trial:04d}'/design/'prediction_records.json').read_text())
            rb=json.loads((out/f'B/truth-{trial:04d}'/design/'prediction_records.json').read_text())
            assert [r['seed'] for r in ra]==[r['seed'] for r in rb]
    sensitivity=json.loads((out/'profile_sensitivity.json').read_text())
    assert len(sensitivity['profile_groups'])==8
    assert len(sensitivity['paired_profile_contrasts'])==4
    contrast=sensitivity['paired_profile_contrasts'][0]
    groups=sensitivity['profile_groups']
    before=next(g for g in groups if g['profile']=='A' and g['n_measurements']==0 and g['n_samples']==2)
    after=next(g for g in groups if g['profile']=='B' and g['n_measurements']==0 and g['n_samples']==2)
    np.testing.assert_allclose(contrast['paired_temperature_variance_difference_C2']['outer_trial_values'],
        np.array(after['temperature_average_variance_C2']['outer_trial_values'])-before['temperature_average_variance_C2']['outer_trial_values'])
    assert set(manifest['report_sha256']) >= {'profile_sensitivity.json','profile_sensitivity.png'}


def test_structural_artifact_binds_laws_and_raw_source_integrity(tmp_path):
    source,cfg=structural_config(tmp_path);load_config(source)
    cfg['scenarios']['A']['mean_log10_k']+=.01;source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='exactly match'):load_config(source)
    cfg['scenarios']['A']['mean_log10_k']-=.01;source.write_text(yaml.safe_dump(cfg))
    np.save(tmp_path/'map.npy',np.zeros((9,9)))
    with pytest.raises(ValueError,match='source map.*changed'):load_config(source)


def test_profile_pairing_requires_common_embedding_and_complete_contract(tmp_path):
    source,cfg=structural_config(tmp_path);cfg['paired_embedding_shape']=[16,32]
    source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='paired embedding'):load_config(source)
    del cfg['paired_embedding_shape'];source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='every controlled-benchmark key'):load_config(source)


def test_complete_controlled_workflow_and_checked_resume(tmp_path):
    source,cfg=config(tmp_path);model=TestModel();factory=lambda *a,**kw:model
    out=tmp_path/'study'
    result=run(source,out,model_factory=factory)
    assert result['status']=='complete'
    assert model.calls==cfg['n_truths']*(1+cfg['n_samples']*5)==63
    summary=json.loads((out/'summary.json').read_text())
    assert len(summary['density'])==5
    assert all(r['n_independent_truths']==3 for r in summary['temperature_calibration'])
    for r in summary['input_probe_calibration']:
        assert sum(r['rank_histogram'])==3
    # Completed cases are verified and reused without any new query.
    run(source,out,stage='temperatures',model_factory=factory)
    run(source,out,stage='report',model_factory=factory)
    assert model.calls==63
    array=np.load(out/'matern32_known/truth-0000/prior/temperature_ensemble.npy',mmap_mode='r+')
    array[0,0,0]+=1;array.flush();del array
    with pytest.raises(ValueError,match='artifact changed|cache changed'):
        run(source,out,stage='temperatures',model_factory=factory)


def test_interrupted_model_queries_resume_same_draw_without_replacement(tmp_path):
    source,cfg=config(tmp_path);model=TestModel();counter=0
    class Interrupted(TestModel):
        def predict_temperature(self,field):
            nonlocal counter
            counter+=1
            if counter==5:
                raise RuntimeError('simulated interruption')
            return model.predict_temperature(field)
    out=tmp_path/'study'
    with pytest.raises(RuntimeError,match='interruption'):
        run(source,out,model_factory=lambda *a,**k:Interrupted())
    run(source,out,stage='temperatures',model_factory=lambda *a,**k:model)
    run(source,out,stage='report',model_factory=lambda *a,**k:model)
    assert model.calls==63
    records=json.loads((out/'matern32_known/truth-0000/prior/prediction_records.json').read_text())
    assert records[0]['seed']==seed_for(cfg['seed'],'matern32_known',0,'posterior',0)
    assert records[0]['seed']!=seed_for(cfg['seed'],'matern32_known',0,'hidden-truth')


def test_revealed_values_only_and_no_innovation_borrowed_from_hidden_field(tmp_path):
    spec=GSToolsReferenceSpec('matern32_radial',12.,15.,sigma_R=.2)
    data=np.array([[1,1],[4,5]])
    system=KnownMeanConditioner((7,8),spec,data,mean_log10=-9.3,cell_size_m=5.)
    rng=np.random.default_rng(4);truth=rng.normal(-9.3,.2,(7,8));changed=truth.copy()
    mask=np.ones(truth.shape,bool);mask[tuple(data.T)]=False;changed[mask]+=100
    prior=rng.normal(size=truth.shape)
    a=system.conditional_log_field(prior,truth[tuple(data.T)])
    b=system.conditional_log_field(prior,changed[tuple(data.T)])
    np.testing.assert_array_equal(a,b)
    assert not np.allclose(a[mask],truth[mask])
    with pytest.raises(ArithmeticError):physical_permeability([[-1000.]])


@pytest.mark.parametrize('key,value',[('measurement_counts',[2,6]),('n_samples',1),('n_truths',0),
    ('coverage_scenarios',['unknown']),('sample_budgets',[2,3]),('permeability_units','m/s')])
def test_config_errors_before_output_or_model_loading(tmp_path,key,value):
    source,cfg=config(tmp_path);cfg[key]=value;source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(source)


def test_source_and_configuration_changes_refuse_resume(tmp_path):
    source,cfg=config(tmp_path);out=tmp_path/'study'
    run(source,out,stage='plan')
    (tmp_path/'model.yaml').write_text('changed: true')
    with pytest.raises(ValueError,match='changed'):
        run(source,out,stage='validate')


def test_resume_refuses_a_different_execution_installation(tmp_path,monkeypatch):
    from subsurface_uq.experiments import controlled_field_benchmark as module
    source,cfg=config(tmp_path);out=tmp_path/'study'
    run(source,out,stage='plan')
    original=module._sources
    monkeypatch.setattr(module,'_sources',lambda *args:{**original(*args),'different_package_location':'unused'})
    with pytest.raises(ValueError,match='execution source or package installation changed'):
        run(source,out,stage='validate')
