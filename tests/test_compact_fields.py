from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import yaml
from threadpoolctl import threadpool_limits

from subsurface_uq.experiments import controlled_field_benchmark as b
from subsurface_uq.experiments.compact_fields import storage_estimate,_source_digest
from subsurface_uq.experiments.regenerate_input import regenerate
from subsurface_uq.sampling.conditional_density import condition_sample_batch,prefix_moments
from subsurface_uq.sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceSpec
from test_controlled_field_benchmark import TestModel
from test_production_fields import production_config,FAMILIES


def compact_config(tmp_path):
    source,cfg=production_config(tmp_path)
    cfg.update(n_samples=5,sample_budgets=[2,5],retain_input_ensemble=False)
    cfg['compact_storage']={'batch_size':2,'retained_input_samples':2,'preview_size':5,'temperature_shape':[7,7]}
    source.write_text(yaml.safe_dump(cfg));return source,cfg


@pytest.mark.parametrize('family',FAMILIES)
def test_batch_covariance_solve_matches_whitened_matheron_and_analytic_moments(tmp_path,family):
    shape=(8,9);spec=GSToolsReferenceSpec(family,9.,13.,sigma_R=.2,angle_deg=27.)
    data=np.array([[0,0],[7,8],[0,8],[7,0],[4,4],[2,6]])
    system=KnownMeanConditioner(shape,spec,data,mean_log10=-9.3,cell_size_m=5.)
    cfg={'shape':list(shape),'cell_size_m':5.,'workers':1}
    sampler=b._prior(cfg,spec);prior=np.stack([sampler.sample_residual(i) for i in range(3)])
    readings=np.linspace(-9.5,-9.1,len(data));counts=[0,2,6]
    directories={n:tmp_path/str(n) for n in counts}
    averages=prefix_moments(system,readings,counts,directories,chunk_size=11)
    condition_sample_batch(system,readings,prior,counts,directories,chunk_size=11)
    queries=np.column_stack(np.unravel_index(np.arange(np.prod(shape)),shape))
    f=system.features(queries)
    for n in counts:
        fields=np.load(directories[n]/'permeability_samples.npy')
        for i in range(len(prior)):
            expected=physical_permeability(system.conditional_log_field(prior[i],readings[:n],count=n,chunk_size=11))
            np.testing.assert_allclose(fields[i],expected,rtol=1.3e-7,atol=0.)
        mean,var=system.moments(f,readings[:n],count=n)
        np.testing.assert_allclose(np.load(directories[n]/'posterior_log10_mean.npy').ravel(),mean,atol=5e-7)
        np.testing.assert_allclose(np.load(directories[n]/'posterior_log10_std.npy').ravel()**2,var,atol=8e-9)
        assert averages[n]==pytest.approx(var.mean(),abs=1e-15)
    assert averages[0]>averages[2]>averages[6]


def test_compact_retains_outputs_selected_inputs_previews_native_views_and_statistics(tmp_path):
    source,cfg=compact_config(tmp_path);model=TestModel();out=tmp_path/'compact'
    result=b.run(source,out,model_factory=lambda *a,**kw:model)
    assert result['status']=='complete' and model.calls==48
    assert result['native_input_ensembles_retained'] is False
    assert not list(out.rglob('batch_scratch')) and not list(out.rglob('features.npy'))
    assert not list(out.rglob('permeability_samples.npy')) and not list(out.rglob('prior_scratch.npy'))
    for name in cfg['scenarios']:
        recipe=json.loads((out/name/'truth-0000/input_recipe.json').read_text())
        assert recipe['configuration']['n_samples']==5
        assert recipe['configuration']['compact_storage']['batch_size']==2
        for _,_,directory in b._cases(cfg,name,0,out):
            selected=np.load(directory/'permeability_selected.npy')
            previews=np.load(directory/'log_permeability_previews.npy')
            temperatures=np.load(directory/'temperature_ensemble.npy')
            rows=json.loads((directory/'prediction_records.json').read_text())
            assert selected.shape==(2,9,9) and previews.shape==(5,5,5) and temperatures.shape==(5,7,7)
            for i in range(2):
                assert b.array_hash(selected[i])==rows[i]['input_sha256']
                np.testing.assert_array_equal(temperatures[i],TestModel().predict_temperature(selected[i]))
                points=np.linspace(0,8,5).astype(int)
                expected=np.log10(selected[i].astype(float))[np.ix_(points,points)].astype(np.float32)
                np.testing.assert_array_equal(previews[i],expected)
            assert (directory/'temperature_q05.npy').exists() and (directory/'qoi_samples.npy').exists()
    images=list((out/'browser').glob('native-*.png'))
    assert len(images)==3*3*2*2
    assert {Image.open(p).size for p in images if '-input-' in p.name}=={(9,9)}
    assert {Image.open(p).size for p in images if '-output-' in p.name}=={(7,7)}
    html=(out/'ensemble_browser.html').read_text(encoding='utf-8')
    assert 'Open native-resolution image' in html and 'first 2 full-resolution inputs' in html
    assert 'native_views' in html
    calls=model.calls;b.run(source,out,'resume',model_factory=lambda *a,**kw:model)
    assert model.calls==calls


def test_compact_and_full_archive_have_same_seeded_mc_statistics(tmp_path):
    source,cfg=compact_config(tmp_path);compact=tmp_path/'compact'
    b.run(source,compact,model_factory=lambda *a,**kw:TestModel())
    full=deepcopy(cfg);full.pop('compact_storage');full['retain_input_ensemble']=True
    archive_cfg=tmp_path/'archive.yaml';archive_cfg.write_text(yaml.safe_dump(full));archive=tmp_path/'archive'
    b.run(archive_cfg,archive,model_factory=lambda *a,**kw:TestModel())
    for name in cfg['scenarios']:
        for _,_,directory in b._cases(cfg,name,0,compact):
            old=archive/directory.relative_to(compact)
            np.testing.assert_allclose(np.load(directory/'temperature_ensemble.npy'),np.load(old/'temperature_ensemble.npy'),atol=1e-6)
            np.testing.assert_allclose(np.load(directory/'qoi_samples.npy'),np.load(old/'qoi_samples.npy'),atol=2e-6)
            np.testing.assert_allclose(np.load(directory/'posterior_log10_std.npy'),np.load(old/'posterior_log10_std.npy'),atol=1e-7)
            np.testing.assert_allclose(np.load(directory/'permeability_selected.npy'),np.load(old/'permeability_samples.npy')[:2],rtol=1.3e-7,atol=0.)


def test_interrupted_prediction_resumes_global_batches_without_replacing_draws(tmp_path,monkeypatch):
    source,cfg=compact_config(tmp_path);out=tmp_path/'compact';model=TestModel();original=model.predict_temperature
    def interrupted(field):
        if model.calls==8:raise RuntimeError('interrupted model')
        return original(field)
    model.predict_temperature=interrupted
    with pytest.raises(RuntimeError,match='interrupted model'):
        b.run(source,out,model_factory=lambda *a,**kw:model)
    batch=next(out.rglob('batch_scratch'))
    assert json.loads((batch/'context.json').read_text())['start']==2
    model.predict_temperature=original
    prior=b._prior;seen=[]
    def tracked(*args,**kwargs):
        sampler=prior(*args,**kwargs);draw=sampler.sample_residual
        def sample(seed):seen.append(seed);return draw(seed)
        sampler.sample_residual=sample;return sampler
    monkeypatch.setattr(b,'_prior',tracked)
    b.run(source,out,'resume',model_factory=lambda *a,**kw:model)
    assert model.calls==48 and not list(out.rglob('batch_scratch'))
    first=next(iter(b.load_config(source)[1]['scenarios']))
    assert not set(seen)&{b.seed_for(cfg['seed'],first,0,'posterior',i) for i in range(4)}
    assert b.seed_for(cfg['seed'],first,0,'posterior',4) in seen


def test_interrupted_prior_generation_reuses_committed_batch_draws(tmp_path,monkeypatch):
    source,cfg=compact_config(tmp_path);out=tmp_path/'compact';model=TestModel()
    b.run(source,out,'plan');b.run(source,out,'validate');b.run(source,out,'inputs')
    first=next(iter(cfg['scenarios']));target=b.seed_for(cfg['seed'],first,0,'posterior',1)
    prior=b._prior
    def interrupted(*args,**kwargs):
        sampler=prior(*args,**kwargs);draw=sampler.sample_residual
        def sample(seed):
            if seed==target:raise RuntimeError('interrupted prior')
            return draw(seed)
        sampler.sample_residual=sample;return sampler
    monkeypatch.setattr(b,'_prior',interrupted)
    with pytest.raises(RuntimeError,match='interrupted prior'):
        b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)
    records=json.loads(next(out.rglob('batch_scratch/prior_records.json')).read_text())
    assert [r['sample_index'] for r in records]==[0]
    seen=[]
    def tracked(*args,**kwargs):
        sampler=prior(*args,**kwargs);draw=sampler.sample_residual
        def sample(seed):seen.append(seed);return draw(seed)
        sampler.sample_residual=sample;return sampler
    monkeypatch.setattr(b,'_prior',tracked)
    b.run(source,out,'resume',model_factory=lambda *a,**kw:model)
    assert b.seed_for(cfg['seed'],first,0,'posterior',0) not in seen
    assert model.calls==48 and not list(out.rglob('batch_scratch'))


def test_resume_rejects_a_changed_auto_selected_device_before_predicting(tmp_path):
    source,cfg=compact_config(tmp_path);out=tmp_path/'compact';model=TestModel()
    model.metadata={**model.metadata,'inference_device':{'requested':'auto','selected':'cpu'}}
    original=model.predict_temperature
    def interrupted(field):
        if model.calls==2:raise RuntimeError('interrupted on CPU')
        return original(field)
    model.predict_temperature=interrupted
    with pytest.raises(RuntimeError,match='interrupted on CPU'):
        b.run(source,out,model_factory=lambda *a,**kw:model)
    records={p:p.read_bytes() for p in out.rglob('prediction_records.json')}
    changed=TestModel()
    changed.metadata={**model.metadata,'inference_device':{'requested':'auto','selected':'cuda:0'}}
    with pytest.raises(ValueError,match='selected inference device'):
        b.run(source,out,'resume',model_factory=lambda *a,**kw:changed)
    assert changed.calls==0
    assert all(p.read_bytes()==data for p,data in records.items())
    model.predict_temperature=original
    b.run(source,out,'resume',model_factory=lambda *a,**kw:model)
    assert model.calls==48


@pytest.mark.parametrize('filename',['permeability_selected.npy','log_permeability_previews.npy'])
def test_compact_retained_corruption_refuses_resume(tmp_path,filename):
    source,cfg=compact_config(tmp_path);out=tmp_path/'compact';model=TestModel()
    b.run(source,out,model_factory=lambda *a,**kw:model)
    path=next(out.rglob(filename));a=np.load(path,mmap_mode='r+');a.flat[0]+=1;a.flush();del a
    with pytest.raises(ValueError,match='artifact changed'):
        b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)
    assert model.calls==48


def test_regeneration_of_unretained_and_last_batch_inputs_survives_moving_results(tmp_path):
    source,cfg=compact_config(tmp_path);out=tmp_path/'compact';model=TestModel()
    b.run(source,out,model_factory=lambda *a,**kw:model)
    moved=tmp_path/'moved';shutil.move(str(out),moved)
    source.unlink();(tmp_path/'model.yaml').unlink()
    for name in cfg['scenarios']:
        case=moved/name/'truth-0000/balanced-000/n-0006'
        rows=json.loads((case/'prediction_records.json').read_text())
        for i in [2,4]:
            target=tmp_path/f'{name}-{i}.npy'
            with threadpool_limits(limits=2,user_api='blas'):
                result=regenerate(moved,name,6,i,target)
            assert result['status']=='verified' and result['sample_index']==i
            assert b.array_hash(np.load(target))==rows[i]['input_sha256']
            assert json.loads(target.with_suffix('.npy.json').read_text())['native_array_sha256']==rows[i]['input_sha256']
    assert model.calls==48 and not list(tmp_path.glob('.regenerate-*'))


@pytest.mark.parametrize('key,value',[('batch_size',0),('retained_input_samples',6),
    ('preview_size',False),('temperature_shape',[10,10])])
def test_invalid_compact_storage_rejected_before_output(tmp_path,key,value):
    source,cfg=compact_config(tmp_path);cfg['compact_storage'][key]=value;source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):b.run(source,tmp_path/'out','plan')
    assert not (tmp_path/'out').exists()


def test_wrong_native_temperature_shape_fails_before_batch_predictions(tmp_path):
    source,cfg=compact_config(tmp_path);cfg['compact_storage']['temperature_shape']=[5,5]
    source.write_text(yaml.safe_dump(cfg));model=TestModel();out=tmp_path/'out'
    with pytest.raises(ValueError,match='temperature_shape'):
        b.run(source,out,model_factory=lambda *a,**kw:model)
    assert model.calls==1 and not list(out.rglob('batch_scratch'))


def test_compact_size_bound_for_the_actual_500_draw_study():
    cfg=yaml.safe_load(Path('configs/controlled_field.kernels.compact.500.yaml').read_text())
    estimate=storage_estimate(cfg,12)
    assert estimate['array_bytes_by_role']['temperature_ensembles']==12*500*1028**2*4
    assert estimate['array_bytes_by_role']['selected_permeability']==12*8*2560**2*4
    assert 29e9<estimate['retained_array_bytes']<31e9
    assert estimate['estimated_peak_upper_bytes']<35e9
    assert estimate['peak_batch_scratch_bytes']<3e9 and estimate['geometry_cache_bytes']==0
    archive=yaml.safe_load(Path('configs/controlled_field.kernels.500.yaml').read_text())
    assert cfg['scenarios']==archive['scenarios'] and cfg['seed']==archive['seed']


def test_multiple_designs_share_one_prior_and_regenerate_the_selected_layout(tmp_path):
    source,cfg=compact_config(tmp_path)
    cfg['coverage_scenarios']=['balanced','clustered'];source.write_text(yaml.safe_dump(cfg))
    model=TestModel();out=tmp_path/'out'
    b.run(source,out,model_factory=lambda *a,**kw:model)
    assert model.calls==78
    for name in cfg['scenarios']:
        rows=json.loads((out/name/'truth-0000/clustered-000/n-0006/prediction_records.json').read_text())
        target=tmp_path/f'{name}-clustered.npy'
        regenerate(out,name,6,4,target,design='clustered-000')
        assert b.array_hash(np.load(target))==rows[4]['input_sha256']


def test_compact_runtime_changes_refuse_resume(tmp_path,monkeypatch):
    from subsurface_uq.experiments import compact_fields
    source,cfg=compact_config(tmp_path);out=tmp_path/'out'
    b.run(source,out,'plan')
    monkeypatch.setattr(compact_fields,'execution_runtime',lambda:{'changed':'numerical-library-version'})
    with pytest.raises(ValueError,match='runtime environment changed'):
        b.run(source,out,'validate')


def test_changed_pending_batch_is_rejected_before_any_new_prediction(tmp_path):
    source,cfg=compact_config(tmp_path);model=TestModel();out=tmp_path/'out';original=model.predict_temperature
    def interrupted(field):
        if model.calls==3:raise RuntimeError('pause within first batch')
        return original(field)
    model.predict_temperature=interrupted
    with pytest.raises(RuntimeError):b.run(source,out,model_factory=lambda *a,**kw:model)
    file=next(out.rglob('batch_scratch/cases/**/permeability_samples.npy'))
    a=np.load(file,mmap_mode='r+');a.flat[0]*=2;a.flush();del a
    model.predict_temperature=original
    with pytest.raises(ValueError,match='artifact changed'):
        b.run(source,out,'temperatures',model_factory=lambda *a,**kw:model)
    assert model.calls==3


def test_compact_reference_selection_keeps_matching_context_and_portable_recipe(tmp_path):
    from test_reference_selection import selected_config
    source,cfg=selected_config(tmp_path)
    cfg['retain_input_ensemble']=False
    cfg['compact_storage']={'batch_size':3,'retained_input_samples':2,'preview_size':5,'temperature_shape':[7,7]}
    source.write_text(yaml.safe_dump(cfg));out=tmp_path/'selected';models=[]
    def factory(*args,reference_id,**kwargs):
        model=TestModel();model.metadata={**model.metadata,'fixed_run_id':reference_id};models.append(model)
        return model
    for reference_id in ['RUN_0','RUN_2']:
        manifest=b.run(source,out,reference_id=reference_id,model_factory=factory)
        assert manifest['reference_context']['fixed_run_id']==reference_id
        assert len(manifest['configuration']['scenarios'])==1
    assert [m.calls for m in models]==[9,9]
    recipe=json.loads((out/'RUN_2/B/truth-0000/input_recipe.json').read_text())
    assert recipe['configuration']['profile_pairing_group']=='paired'
    (tmp_path/'RUN_2.npy').unlink();(tmp_path/'model.yaml').unlink();source.unlink()
    result=regenerate(out/'RUN_2','B',6,3,tmp_path/'B-restored.npy')
    assert result['status']=='verified' and result['reference_id']=='RUN_2'


def test_source_fingerprints_allow_line_ending_conversion_but_detect_code_changes(tmp_path):
    first=tmp_path/'checkout.py';second=tmp_path/'wheel.py'
    first.write_bytes(b'def sample(x):\r\n    return x + 1\r\n')
    second.write_bytes(b'def sample(x):\n    return x + 1\n')
    assert _source_digest(first)==_source_digest(second)
    second.write_bytes(b'def sample(x):\n    return x + 2\n')
    assert _source_digest(first)!=_source_digest(second)
