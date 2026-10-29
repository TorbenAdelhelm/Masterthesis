from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cho_solve
import yaml

from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.conditional_density import condition_prefixes
from subsurface_uq.sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from subsurface_uq.sampling.gstools_reference import GSToolsReferenceFFTSampler,GSToolsReferenceSpec
from subsurface_uq.experiments.conditional_density_visual import run,load_config


def test_streamed_prefixes_match_full_matheron_and_independent_covariance(tmp_path):
    shape=(8,9);spec=GSToolsReferenceSpec('matern32_radial',8.,12.,sigma_R=.2,angle_deg=25.)
    indices=np.array([[0,0],[7,8],[0,8],[7,0],[4,4],[2,6],[6,2]])
    system=KnownMeanConditioner(shape,spec,indices,mean_log10=-9.3,cell_size_m=5.)
    sampler=GSToolsReferenceFFTSampler(replace(spec,sigma_R=1.),shape,cell_size_m=5.,workers=1)
    prior=np.array([sampler.sample_residual(i) for i in range(3)])
    readings=np.linspace(-9.6,-9.1,len(indices));counts=[1,2,3,5,7]
    directories={n:tmp_path/str(n) for n in counts}
    averages=condition_prefixes(system,readings,prior,counts,directories,chunk_size=11)
    queries=np.column_stack(np.unravel_index(np.arange(np.prod(shape)),shape))
    offset=(queries[:,None]-indices[None,:])*5.
    cross=spec.correlation(offset[...,0],offset[...,1])
    offset=(indices[:,None]-indices[None,:])*5.
    bb=spec.correlation(offset[...,0],offset[...,1])
    for n in counts:
        samples=np.load(directories[n]/'permeability_samples.npy')
        for i in range(len(prior)):
            expected=physical_permeability(system.conditional_log_field(prior[i],readings[:n],count=n,chunk_size=13))
            np.testing.assert_allclose(samples[i],expected,rtol=1.3e-7,atol=0.)
            np.testing.assert_allclose(np.log10(samples[i][tuple(indices[:n].T)].astype(float)),readings[:n],atol=3e-8)
        weights=cho_solve((np.linalg.cholesky(bb[:n,:n]),True),cross[:,:n].T).T
        mean=-9.3+weights@(readings[:n]+9.3)
        var=.2**2*(1-np.sum(weights*cross[:,:n],axis=1))
        np.testing.assert_allclose(np.load(directories[n]/'posterior_log10_mean.npy').ravel(),mean,atol=5e-7)
        np.testing.assert_allclose(np.load(directories[n]/'posterior_log10_std.npy').ravel()**2,var,atol=5e-9)
        assert averages[n]==pytest.approx(var.mean(),abs=1e-15)
    assert np.all(np.diff(list(averages.values()))<0)


def test_prefix_contract_rejects_changed_geometry_and_nonfinite_driving_values(tmp_path):
    spec=GSToolsReferenceSpec('matern32_radial',8.,10.,sigma_R=.2)
    system=KnownMeanConditioner((5,6),spec,np.array([[0,0],[4,5]]),mean_log10=-9.3,cell_size_m=5.)
    prior=np.zeros((2,5,6))
    with pytest.raises(ValueError,match='prefixes'):
        condition_prefixes(system,[-9.2,-9.4],prior,[1],{1:tmp_path/'one'})
    prior[:,2,3]=np.nan
    with pytest.raises(ValueError,match='nonfinite prior'):
        condition_prefixes(system,[-9.2,-9.4],prior,[1,2],{1:tmp_path/'one',2:tmp_path/'two'})


class TestModel:
    __test__=False
    input_shape=(9,9);cell_size_m=5.;background_temperature=10.6
    metadata={'files_sha256':{},'kind':'nonlinear-test-model'}
    def __init__(self):self.calls=0
    def predict_temperature(self,field):
        self.calls+=1;y=np.log10(field[1:8,1:8].astype(float))+9.3
        return (10.6+.2*y+.1*y*y).astype(np.float32)
    def normalization_diagnostics(self,field):return {'min':float(field.min()),'max':float(field.max())}


def configuration(tmp_path):
    law={'mean_log10_k':-9.3,'covariance':{'family':'matern32_radial','ell_row_m':8.,'ell_col_m':10.,
        'sigma_R':.2,'center':'median','angle_deg':0.}}
    raw=tmp_path/'map.npy';np.save(raw,np.ones((9,9),dtype=np.float32)*1e-9)
    profile={'schema_version':1,'family':'matern32_radial','profile_set_id':'test-profile',
        'cell_size_m':5.,'native_shape':[9,9],'profiles':{'test':{'law':law}},
        'source_maps':{'test':{'path':'map.npy','sha256':sha256_file(raw)}}}
    (tmp_path/'profiles.json').write_text(json.dumps(profile));(tmp_path/'model.yaml').write_text('test: true')
    cfg={'structural_profile_file':'profiles.json','profile_name':'test','measurement_counts':[2,6],
        'n_samples':8,'candidate_stride_cells':2,'seed':26,'embedding_shape':[32,32],
        'workers':1,'chunk_size':13,'model_config':'model.yaml','model_asset_root':'.',
        'torch_threads':1,'plot_stride_cells':1}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg));return path,cfg


def test_complete_gallery_retains_every_draw_and_verified_resume(tmp_path):
    source,cfg=configuration(tmp_path);model=TestModel();out=tmp_path/'study'
    manifest=run(source,out,model_factory=lambda *a,**kw:model)
    assert manifest['status']=='complete' and model.calls==17
    assert manifest['geometry_cache_bytes']==0
    assert not list(out.rglob('features.npy'))
    summary=json.loads((out/'summary.json').read_text())
    assert summary['native_inputs_and_outputs_retained']
    assert summary['counts'][0]['analytic_input_sd_rms_log10']>summary['counts'][1]['analytic_input_sd_rms_log10']
    observations=json.loads((out/'observations.json').read_text())
    seeds=[]
    for n in cfg['measurement_counts']:
        case=out/f'n-{n:04d}';samples=np.load(case/'permeability_samples.npy')
        assert samples.shape==(8,9,9) and np.load(case/'temperature_samples.npy').shape==(8,7,7)
        context=json.loads((case/'input_summary.json').read_text());seeds.append(context['posterior_seeds'])
        assert context['indices']==observations['indices'][:n]
        for i in range(1,9):
            for role in ['input','output','input_error','output_error','input_deviation','output_deviation']:
                assert (out/f'figures/n-{n:04d}/sample-{i:02d}-{role}.png').is_file()
        for kind in ['inputs','outputs']:
            for scale in ['shared','detail']:assert (out/f'figures/n-{n:04d}-{kind}-{scale}.png').is_file()
    assert seeds[0]==seeds[1]
    html=(out/'report.html').read_text();assert '__STUDY_DATA__' not in html and 'const DATA=' in html
    assert 'Inspect one sample' in html and 'Enhance error contrast' in html
    run(source,out,stage='temperatures',model_factory=lambda *a,**kw:model)
    assert model.calls==17
    array=np.load(out/'n-0002/temperature_samples.npy',mmap_mode='r+');array[0,0,0]+=1;array.flush();del array
    with pytest.raises(ValueError,match='artifact changed'):
        run(source,out,stage='temperatures',model_factory=lambda *a,**kw:model)


def test_configuration_guards_source_and_eight_sample_visual_contract(tmp_path):
    source,cfg=configuration(tmp_path);load_config(source)
    cfg['n_samples']=4;source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='exactly eight'):load_config(source)
    cfg['n_samples']=8;source.write_text(yaml.safe_dump(cfg));np.save(tmp_path/'map.npy',np.zeros((9,9)))
    with pytest.raises(ValueError,match='source map.*changed'):load_config(source)
