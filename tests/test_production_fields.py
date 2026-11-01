import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
from scipy.integrate import quad
import yaml

from subsurface_uq.experiments import controlled_field_benchmark as b
from subsurface_uq.sampling.structural_profiles import fit_profile,area_matched_profile,grid_mean_correlation
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec
from subsurface_uq.sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from test_controlled_field_benchmark import config,TestModel


FAMILIES=['gaussian','matern32_radial','exponential_radial']


def production_config(tmp_path):
    source,cfg=config(tmp_path)
    law=next(iter(cfg['scenarios'].values()))
    cfg.update(n_truths=1,retain_input_ensemble=True,validation_samples=4,
               coverage_scenarios=['balanced'],n_designs=1)
    cfg['scenarios']={}
    for family in FAMILIES:
        value=deepcopy(law);value['covariance'].update(family=family,ell_row_m=8.,ell_col_m=10.)
        cfg['scenarios'][family]=value
    source.write_text(yaml.safe_dump(cfg));return source,cfg


@pytest.mark.parametrize('family',FAMILIES)
def test_integral_correlation_area_identity_and_family_specific_fits(family):
    unit=GSToolsReferenceSpec(family,1.,1.,sigma_R=1.)
    assert quad(lambda r:r*unit.correlation(r,0.),0,np.inf)[0]==pytest.approx(1.,abs=1e-10)
    rng=np.random.default_rng(3)
    logs=-9.+.1*rng.normal(size=(13,14))
    profile=fit_profile(logs,cell_size_m=5.,lags_cells=[1,2,4],length_bounds_m=[2.,100.],family=family)
    assert profile['law']['covariance']['family']==family
    matched=area_matched_profile(profile,'exponential_radial',shape=logs.shape,cell_size_m=5.,lags_cells=[1,2,4])
    spec=GSToolsReferenceSpec(**matched['law']['covariance'])
    q=grid_mean_correlation(spec,logs.shape,5.)
    assert spec.sigma_R**2*(1-q)==pytest.approx(logs.var(),abs=1e-14)
    assert matched['correlation_area_m2']==pytest.approx(2*np.pi*spec.ell_row_m*spec.ell_col_m)


def test_complete_production_matches_independent_draws_retains_and_browses_all(tmp_path):
    source,cfg=production_config(tmp_path);model=TestModel();out=tmp_path/'out'
    manifest=b.run(source,out,model_factory=lambda *a,**kw:model)
    assert manifest['status']=='complete' and model.calls==39
    assert manifest['geometry_cache_bytes']==0
    assert not list(out.rglob('features.npy')) and not list(out.rglob('prior_scratch.npy'))
    assert len(list((out/'browser').glob('*.png')))==3*3*6
    html=(out/'ensemble_browser.html').read_text(encoding='utf-8')
    assert '__STUDY_DATA__' not in html and all(f in html for f in FAMILIES)
    assert not list((out/'browser').glob('*.npz'))
    checks=json.loads((out/'dense_validation/summary.json').read_text())
    assert checks['dense_samples_per_trial']==4
    assert set(checks['native_fft_checks'])==set(FAMILIES)
    for name,law in cfg['scenarios'].items():
        root=out/name/'truth-0000';truth=np.load(root/'truth_log10.npy')
        indices,_=b._locations(cfg,name,'balanced',0)
        spec=GSToolsReferenceSpec(**law['covariance'])
        conditioner=KnownMeanConditioner(cfg['shape'],spec,indices,mean_log10=law['mean_log10_k'],cell_size_m=5.)
        sampler=b._prior(cfg,spec)
        for design,n,directory in b._cases(cfg,name,0,out):
            fields=np.load(directory/'permeability_samples.npy')
            temps=np.load(directory/'temperature_ensemble.npy')
            records=json.loads((directory/'prediction_records.json').read_text())
            assert fields.shape==(4,9,9) and temps.shape==(4,7,7)
            for i in range(4):
                seed=b.seed_for(cfg['seed'],b._stream(cfg,name),0,'posterior',i)
                expected=physical_permeability(conditioner.conditional_log_field(sampler.sample_residual(seed),truth[tuple(indices[:n].T)],count=n))
                np.testing.assert_allclose(fields[i],expected,rtol=1.3e-7,atol=0.)
                assert records[i]['input_sha256']==b.array_hash(fields[i])
                np.testing.assert_array_equal(temps[i],TestModel().predict_temperature(fields[i]))
    calls=model.calls
    b.run(source,out,'resume',model_factory=lambda *a,**kw:model)
    assert model.calls==calls


def test_restart_prediction_uses_retained_inputs_without_resampling(tmp_path,monkeypatch):
    source,cfg=production_config(tmp_path);out=tmp_path/'out';model=TestModel()
    b.run(source,out,'plan');b.run(source,out,'validate');b.run(source,out,'inputs')
    original=model.predict_temperature
    def interrupted(field):
        if model.calls==5:raise RuntimeError('interrupted')
        return original(field)
    model.predict_temperature=interrupted
    with pytest.raises(RuntimeError,match='interrupted'):
        b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)
    model.predict_temperature=original
    monkeypatch.setattr(b,'_prior',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('resampled')))
    b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)
    assert model.calls==39
    case=next(out.rglob('permeability_samples.npy'))
    values=np.load(case,mmap_mode='r+');values[0,0,0]*=2;values.flush();del values
    with pytest.raises(ValueError,match='artifact changed'):
        b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)


def test_restart_fft_scratch_keeps_global_sample_indices(tmp_path,monkeypatch):
    source,cfg=production_config(tmp_path);out=tmp_path/'out'
    b.run(source,out,'plan');b.run(source,out,'validate')
    original=b._prior
    class Interrupted:
        def __init__(self,sampler):self.sampler=sampler;self.metadata=sampler.metadata;self.calls=0
        def sample_residual(self,seed):
            self.calls+=1
            if self.calls==3:raise RuntimeError('FFT interrupted')
            return self.sampler.sample_residual(seed)
    monkeypatch.setattr(b,'_prior',lambda *a,**kw:Interrupted(original(*a,**kw)))
    with pytest.raises(RuntimeError,match='FFT interrupted'):
        b.run(source,out,'inputs')
    saved=json.loads(next(out.rglob('prior_records.json')).read_text())
    assert [r['sample_index'] for r in saved]==[0]
    monkeypatch.setattr(b,'_prior',original)
    result=b.run(source,out,'inputs')
    assert result['status']=='inputs_ready'
    assert not list(out.rglob('prior_scratch.npy'))
    for name in cfg['scenarios']:
        rows=json.loads((out/name/'truth-0000/prior_records.json').read_text())
        assert [r['seed'] for r in rows]==[b.seed_for(cfg['seed'],b._stream(cfg,name),0,'posterior',i) for i in range(4)]


def test_committed_study_has_all_reference_kernel_count_combinations():
    catalog=json.loads(Path('configs/structural_profiles.kernels.json').read_text())
    cfg=yaml.safe_load(Path('configs/controlled_field.kernels.500.yaml').read_text())
    assert catalog['schema_version']==2
    assert set(catalog['source_maps'])=={'RUN_0','RUN_2','RUN_3','RUN_4'}
    assert cfg['n_samples']==500 and cfg['measurement_counts']==[0,64,128,256]
    assert cfg['retain_input_ensemble'] and cfg['require_reference_id']
    assert cfg['scenarios']=={k:p['law'] for k,p in catalog['profiles'].items()}
    for ref in catalog['source_maps']:
        profiles=[p for p in catalog['profiles'].values() if p['source_map']==ref]
        assert {p['law']['covariance']['family'] for p in profiles}==set(FAMILIES)
        assert len({p['correlation_area_m2'] for p in profiles})==1
        assert all('independent_family_fit' in p for p in profiles)
        assert all(p['expected_grid_spatial_variance_log10']==pytest.approx(p['spatial_variance_log10']) for p in profiles)


def test_multifamily_fit_roundtrip_has_explicit_source_and_anchor(tmp_path):
    from subsurface_uq.experiments.structural_profiles import fit,verify_artifact
    np.save(tmp_path/'RUN_0.npy',10**(-9.+.1*np.random.default_rng(2).normal(size=(9,9))))
    cfg={'references':{'RUN_0':'RUN_0.npy'},'preparation_manifest':None,'cell_size_m':5.,
        'permeability_units':'m2','lags_cells':[1,2,4],'length_bounds_m':[2.,12.],
        'quadrature_order':16,'structural_samples':4,'diagnostic_stride_cells':1,
        'paired_embedding_shape':[32,32],'seed':31,'workers':1,
        'families':FAMILIES,'correlation_area_anchor':'gaussian'}
    source=tmp_path/'fit.yaml';source.write_text(yaml.safe_dump(cfg))
    output=tmp_path/'catalog/profiles.json'
    result=fit(source,output,tmp_path/'report')
    verified,_=verify_artifact(output,reference_id='RUN_0')
    assert result==verified and verified['schema_version']==2
    assert set(verified['profiles'])=={f'RUN_0_{family}' for family in FAMILIES}
    for p in verified['profiles'].values():
        assert p['source_map']=='RUN_0' and p['anchor_family']=='gaussian'
        assert p['independent_family_fit']['law']['covariance']['family']==p['law']['covariance']['family']
