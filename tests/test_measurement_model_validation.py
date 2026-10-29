import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.distance import cdist
import yaml

from subsurface_uq.experiments import measurement_model_validation as workflow
from subsurface_uq.validation.measurement_model import (MeasurementModel, reml_objective, correlation,
    design_matrix, spatial_partition, balanced_block_folds, block_coordinates, buffered_training_mask,
    fit_model, paired_block_bootstrap)


@pytest.mark.parametrize('family',['exponential','matern32'])
def test_reml_likelihood_and_gradient_match_independent_dense_calculation(family):
    rng=np.random.default_rng(21)
    x=rng.uniform(0,7000,(24,2));y=rng.normal(-2.4,.5,24)
    h=np.column_stack((np.ones(24),(x-x.mean(axis=0))/10000))
    d=cdist(x,x);parameters=np.log([1400.,.5,.25])
    value,gradient=reml_objective(parameters,d,y,h,family)
    numerical=approx_derivative(lambda z:reml_objective(z,d,y,h,family)[0],parameters,method='3-point').ravel()
    np.testing.assert_allclose(gradient,numerical,rtol=2e-5,atol=2e-6)
    ell,sigma,tau=np.exp(parameters)
    k=sigma**2*correlation(d,ell,family)+tau**2*np.eye(24)
    inv=np.linalg.inv(k);info=h.T@inv@h
    beta=np.linalg.solve(info,h.T@inv@y);r=y-h@beta
    expected=.5*(r@inv@r+np.linalg.slogdet(k)[1]+np.linalg.slogdet(info)[1]+21*np.log(2*np.pi))
    assert value==pytest.approx(expected,abs=1e-10)


@pytest.mark.parametrize('trend',['constant','linear'])
def test_universal_prediction_retains_mean_uncertainty_and_distinguishes_nugget(trend):
    rng=np.random.default_rng(3)
    x=rng.uniform(0,8000,(24,2));y=rng.normal(-2.4,.5,24);q=rng.uniform(0,9000,(7,2))
    p={'family':'matern32','trend':trend,'length_m':1900.,'structured_std_log10':.4,
       'nugget_std_log10':.3,'coordinate_center':x.mean(axis=0).tolist(),'coordinate_scale_m':10000.}
    model=MeasurementModel('reml_matern32_'+trend,x,y,p)
    mean,observed,latent=model.predict(q)
    k=.4**2*correlation(cdist(x,x),1900.,'matern32')+.3**2*np.eye(24)
    cross=.4**2*correlation(cdist(q,x),1900.,'matern32')
    h=design_matrix(x,x.mean(axis=0),10000.,trend);hq=design_matrix(q,x.mean(axis=0),10000.,trend)
    ki=np.linalg.inv(k);i=np.linalg.inv(h.T@ki@h)
    b=i@h.T@ki@y;r=hq-cross@ki@h
    expected_mean=hq@b+cross@ki@(y-h@b)
    expected_var=.4**2-np.sum(cross*(cross@ki),axis=1)+np.sum(r*(r@i),axis=1)
    np.testing.assert_allclose(mean,expected_mean,atol=1e-12)
    np.testing.assert_allclose(latent,expected_var,atol=1e-12)
    np.testing.assert_allclose(observed-latent,.3**2,atol=1e-14)
    shifted=MeasurementModel(model.name,x,y-7.5,p).predict(q)
    np.testing.assert_allclose(shifted[0],mean-7.5,atol=1e-12)
    np.testing.assert_allclose(shifted[1],observed,atol=1e-14)


def coordinates():
    rng=np.random.default_rng(41)
    a,b=np.meshgrid(np.arange(12)*1500,np.arange(12)*1500)
    return np.column_stack((a.ravel(),b.ravel()))+rng.uniform(50,300,(144,2))+[4450000,5320000]


def test_spatial_partitions_and_buffers_group_colocated_readings():
    x=coordinates();x=np.vstack((x,x[30]))
    p=spatial_partition(x,block_size_m=6000.,test_fraction=.25,buffer_m=500.,seed=6)
    assert not np.any(p['development']&p['test'])
    assert p['development'][30]==p['development'][-1]
    assert p['test'][30]==p['test'][-1]
    assert cdist(x[p['development']],x[p['test']]).min()>=500.
    folds=balanced_block_folds(x,size=3000.,folds=3,seed=9)
    for block in np.unique(block_coordinates(x,3000),axis=0):
        assert len(np.unique(folds[np.all(block_coordinates(x,3000)==block,axis=1)]))==1
    for fold in range(3):
        test=folds==fold
        train=buffered_training_mask(x,~test,test,block_size_m=3000.,buffer_m=500.)
        assert cdist(x[train],x[test]).min()>=500.


def fixture_config(tmp_path,monkeypatch):
    cfg=yaml.safe_load(Path('configs/measurement_model.validation.yaml').read_text(encoding='utf-8'))
    cfg['models']=['iid_normal','legacy_radial_exponential','reml_exponential_constant']
    cfg['holdout'].update(block_size_m=6000.,fraction=.25,buffer_m=500.,seed=6)
    cfg['development'].update(block_size_m=3000.,buffer_m=500.,folds=3,seed=9)
    cfg['variogram'].update(lag_bin_m=500.,max_lag_m=5000.,min_pairs_per_bin=1)
    cfg['reml'].update(initial_lengths_m=[1200.],maxiter=150)
    cfg['bootstrap']['replicates']=100
    cfg['rules']['minimum_test_blocks']=3
    x=coordinates();rng=np.random.default_rng(5)
    y=-2.4+.25*np.sin((x[:,0]-4450000)/3000)+rng.normal(0,.12,len(x))
    def fake_loader(path,**kwargs):
        raw=json.loads(Path(path).read_text());x=np.asarray(raw['x']);y=np.asarray(raw['y'])
        rows=[{'excel_row':j+2,'x_m':xy[0],'y_m':xy[1],'hydraulic_conductivity_m_s':10.**value,
               'log10_hydraulic_conductivity':value} for j,(xy,value) in enumerate(zip(x,y))]
        return SimpleNamespace(to_rows=lambda:rows,coordinates_xy_m=x,log10_hydraulic_conductivity=y),{'n_fixture':len(x)}
    monkeypatch.setattr(workflow,'load_munich_hydraulic_conductivity_measurements',fake_loader)
    data=tmp_path/'observations.json';data.write_text(json.dumps({'x':x.tolist(),'y':y.tolist()}))
    cfg['measurements']='observations.json'
    source=tmp_path/'config.yaml';source.write_text(yaml.safe_dump(cfg))
    return source,cfg,x,y,data


def test_complete_observation_validation_and_verified_report_recovery(tmp_path,monkeypatch):
    source,cfg,x,y,data=fixture_config(tmp_path,monkeypatch)
    output=tmp_path/'study';result=workflow.run(source,output)
    assert result['status']=='complete' and result['lgcnn_evaluations']==0
    assert result['geological_field_sampling_validated'] is False
    heldout=json.loads((output/'holdout.json').read_text())
    assert len(heldout['models'])>=2
    for model in heldout['models'].values():
        assert model['assessment']['latent_field_calibrated'] is False
    workflow.run(source,output,stage='report')
    record=json.loads((output/'development.json').read_text())
    for r in record['fold_records']:
        assert not set(r['train_indices'])&set(r['test_indices'])
    path=output/'holdout.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='artifact changed'):
        workflow.run(source,output,stage='report')


def test_future_holdout_values_cannot_change_development_selection(tmp_path,monkeypatch):
    source,cfg,x,y,data=fixture_config(tmp_path,monkeypatch)
    first=tmp_path/'first';workflow.run(source,first,stage='plan');workflow.run(source,first,stage='develop')
    with np.load(first/'partition.npz') as p:
        test=p['test'].copy()
    changed=y.copy();changed[test]+=1.
    data.write_text(json.dumps({'x':x.tolist(),'y':changed.tolist()}))
    second=tmp_path/'second';workflow.run(source,second,stage='plan');workflow.run(source,second,stage='develop')
    a=json.loads((first/'development.json').read_text());b=json.loads((second/'development.json').read_text())
    assert a==b
    with pytest.raises(ValueError,match='source/configuration/installation changed'):
        workflow.run(source,first,stage='evaluate')


def test_paired_bootstrap_uses_spatial_blocks_and_can_reject_useless_model():
    y=np.array([0.,0.,1.,1.]);m=np.array([0.,0.,1.,1.]);s=np.ones(4)*.1
    blocks=np.array([[0,0],[0,0],[1,0],[1,0]])
    r=paired_block_bootstrap(y,m,s,m+1,s,blocks,replicates=100,seed=5)
    assert r['n_blocks']==2 and r['ci95'][1]<0
    result=workflow.assess_observation_model(
        {'point_weighted':{'standardized_rmse':1.,'standardized_bias':0.},'blocks':[{},{}],
         'equal_block_mean_crps_log10':1.,'equal_block_coverage90':.9},
        {'equal_block_mean_crps_log10':1.},
        {'ci95':[0.,0.]},
        {'minimum_test_blocks':2,'coverage90_min':.8,'coverage90_max':.98,'standardized_rmse_min':.75,
         'standardized_rmse_max':1.5,'absolute_standardized_bias_max':.35,'minimum_relative_crps_improvement':.05,
         'require_paired_bootstrap_improvement':True})
    assert not result['passes_exploratory_observation_checks']
    assert not result['latent_field_calibrated']


def test_config_requires_baseline_and_valid_parameter_bounds(tmp_path,monkeypatch):
    source,cfg,*_=fixture_config(tmp_path,monkeypatch)
    cfg['reml']['length_bounds_m']=[5000,100]
    source.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='parameter bounds'):
        workflow.load_config(source)


def test_positive_nugget_retains_conflicting_colocated_observations():
    rng=np.random.default_rng(1);x=rng.uniform(0,5000,(16,2));x[-1]=x[0]
    y=rng.normal(-2.4,.3,len(x));y[-1]=y[0]+.3
    cfg=yaml.safe_load(Path('configs/measurement_model.validation.yaml').read_text())
    cfg['reml']['initial_lengths_m']=[1200.]
    model=fit_model('reml_matern32_constant',x,y,cfg)
    m,var,latent=model.predict(x[[0]])
    assert var[0]>latent[0]>=0
    assert model.parameters['nugget_std_log10']>0
    assert len(model.y)==16
