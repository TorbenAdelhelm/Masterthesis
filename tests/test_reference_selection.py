import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from subsurface_uq.experiments import controlled_field_benchmark as controlled
from subsurface_uq.experiments import conditional_density_visual as gallery
from subsurface_uq.experiments.reference_selection import model_context, reference_name
from subsurface_uq.io import sha256_file
from subsurface_uq.sampling.reference_inventory import discover_reference_runs
from subsurface_uq.sampling.darus_real import RELEASE25_INITIAL_TIME_GROUP
from test_controlled_field_benchmark import structural_config, TestModel
from test_conditional_density_visual import configuration as gallery_configuration


def context_files(root, ids=('RUN_0','RUN_2'), shape=(9,9)):
    for reference_id in ids:
        pki=root/reference_id/'prepared_pki'
        (pki/'Inputs').mkdir(parents=True)
        (pki/'Inputs'/f'{reference_id}.pt').write_bytes(reference_id.encode())
        (pki/'info.yaml').write_text(yaml.safe_dump({'CellsNumber':list(shape),'CellsSize':[5.,5.],
            'reference_pilot_fixed_run':reference_id}))
    (root/'model.yaml').write_text(yaml.safe_dump({'release25':{
        'fixed_run_id':'RUN_2','prepared_pki_dir':'{reference_id}/prepared_pki'}}))


def selected_config(tmp_path):
    source,cfg=structural_config(tmp_path)
    cfg.update(n_truths=1,require_reference_id=True)
    artifact=json.loads((tmp_path/'profiles.json').read_text())
    artifact['source_maps']={}
    artifact['supporting_files']=[]
    for name,reference_id in [('A','RUN_0'),('B','RUN_2')]:
        path=tmp_path/f'{reference_id}.npy'
        np.save(path,np.full((9,9),1e-9,dtype=np.float32))
        artifact['source_maps'][reference_id]={'path':path.name,'sha256':sha256_file(path)}
        artifact['profiles'][name]['source_map']=reference_id
        provenance=tmp_path/f'{reference_id}.json'
        provenance.write_text(json.dumps({'references':[{'reference_run':reference_id}]}))
        artifact['supporting_files'].append({'path':provenance.name,'sha256':sha256_file(provenance),
            'reference_ids':[reference_id]})
    (tmp_path/'profiles.json').write_text(json.dumps(artifact))
    context_files(tmp_path)
    source.write_text(yaml.safe_dump(cfg))
    return source,cfg


def test_complete_reference_inventory_does_not_assume_ids_are_consecutive(tmp_path):
    for reference_id in ['RUN_0','RUN_1','RUN_2','RUN_3','RUN_4','RUN_5']:
        run=tmp_path/reference_id;run.mkdir()
        with h5py.File(run/'permeability.h5','w') as handle:
            handle.create_dataset('permeability',shape=(1,))
        if reference_id=='RUN_1':continue
        with h5py.File(run/'pflotran.h5','w') as handle:
            group=handle.create_group(RELEASE25_INITIAL_TIME_GROUP)
            for name in ['Permeability X [m^2]','Liquid Pressure [Pa]','Material ID','Temperature [C]']:
                group.create_dataset(name,shape=(2560*(5120 if reference_id=='RUN_5' else 2560),),dtype='f8')
    rows=discover_reference_runs(tmp_path)
    assert [r['reference_id'] for r in rows if r['can_prepare_matched_context']]==['RUN_0','RUN_2','RUN_3','RUN_4']
    assert next(r for r in rows if r['reference_id']=='RUN_1')['status']=='input_only'
    assert next(r for r in rows if r['reference_id']=='RUN_5')['status']=='different_grid'


def test_reference_selection_requires_a_flag_and_explicit_source_binding(tmp_path):
    source,cfg=selected_config(tmp_path)
    with pytest.raises(ValueError,match='requires --reference-id'):controlled.load_config(source)
    _,selected=controlled.load_config(source,reference_id='RUN_0')
    assert list(selected['scenarios'])==['A']
    assert selected['sensitivity_reference_profile']=='A'
    assert selected['reference_context']['fixed_run_id']=='RUN_0'
    with pytest.raises(ValueError,match='available:'):controlled.load_config(source,reference_id='RUN_4')
    with pytest.raises(ValueError,match='RUN name'):reference_name('../RUN_0')


def test_missing_or_mismatched_physical_context_never_falls_back_to_run2(tmp_path):
    source,cfg=selected_config(tmp_path)
    (tmp_path/'RUN_0/prepared_pki/Inputs/RUN_0.pt').unlink()
    with pytest.raises(ValueError,match='matching prepared inputs'):controlled.run(source,tmp_path/'study',stage='plan',reference_id='RUN_0')
    assert not (tmp_path/'study').exists()
    context=tmp_path/'RUN_2/prepared_pki/info.yaml'
    metadata=yaml.safe_load(context.read_text());metadata['reference_pilot_fixed_run']='RUN_0'
    context.write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError,match='different reference RUN'):controlled.load_config(source,reference_id='RUN_2')


def test_two_selected_runs_have_separate_results_and_matching_model_contexts(tmp_path):
    source,cfg=selected_config(tmp_path);models={}
    def factory(path,*,asset_root,torch_threads,reference_id):
        if reference_id not in models:
            model=TestModel();model.metadata={**model.metadata,'fixed_run_id':reference_id};models[reference_id]=model
        return models[reference_id]
    for reference_id,profile in [('RUN_0','A'),('RUN_2','B')]:
        manifest=controlled.run(source,tmp_path/'study',reference_id=reference_id,model_factory=factory)
        assert manifest['reference_id']==reference_id and manifest['model']['fixed_run_id']==reference_id
        assert list(manifest['configuration']['scenarios'])==[profile]
        assert (tmp_path/'study'/reference_id/profile/'truth-0000/prior/temperature_ensemble.npy').is_file()
        summary=json.loads((tmp_path/'study'/reference_id/'summary.json').read_text())
        assert summary['reference_id']==reference_id
        assert models[reference_id].calls==9
        controlled.run(source,tmp_path/'study',stage='temperatures',reference_id=reference_id,model_factory=factory)
        assert models[reference_id].calls==9


def test_only_selected_sources_are_required_and_selected_input_changes_refuse_resume(tmp_path):
    source,_=selected_config(tmp_path);out=tmp_path/'study'
    controlled.run(source,out,stage='plan',reference_id='RUN_0')
    (tmp_path/'RUN_2.npy').unlink()
    (tmp_path/'RUN_2.json').unlink()
    controlled.run(source,out,stage='validate',reference_id='RUN_0')
    (tmp_path/'RUN_0/prepared_pki/Inputs/RUN_0.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):controlled.run(source,out,stage='inputs',reference_id='RUN_0')


def test_prediction_refuses_a_factory_using_the_wrong_run(tmp_path):
    source,_=selected_config(tmp_path);model=TestModel()
    model.metadata={**model.metadata,'fixed_run_id':'RUN_2'}
    with pytest.raises(ValueError,match='does not match the selected'):
        controlled.run(source,tmp_path/'study',reference_id='RUN_0',model_factory=lambda *a,**kw:model)
    assert model.calls==0


def test_gallery_selects_profile_and_matching_context_before_temperature_queries(tmp_path):
    source,cfg=gallery_configuration(tmp_path)
    artifact=json.loads((tmp_path/'profiles.json').read_text())
    artifact['profiles']['test']['source_map']='RUN_0'
    artifact['source_maps']={'RUN_0':artifact['source_maps']['test']}
    (tmp_path/'profiles.json').write_text(json.dumps(artifact))
    context_files(tmp_path,ids=('RUN_0',))
    cfg['require_reference_id']=True;source.write_text(yaml.safe_dump(cfg))
    out=tmp_path/'gallery'
    manifest=gallery.run(source,out,stage='plan',reference_id='RUN_0')
    assert manifest['reference_id']=='RUN_0' and manifest['configuration']['profile_name']=='test'
    assert (out/'RUN_0/manifest.json').is_file()
    gallery.run(source,out,stage='inputs',reference_id='RUN_0')
    model=TestModel();model.metadata={**model.metadata,'fixed_run_id':'RUN_0'}
    gallery.run(source,out,stage='temperatures',reference_id='RUN_0',model_factory=lambda *a,**kw:model)
    assert model.calls==17


def test_cli_forwards_reference_id(monkeypatch,tmp_path):
    received={}
    monkeypatch.setattr(controlled,'run',lambda *a,**kw:received.update(kw))
    controlled.main(['--config','study.yaml','--output-dir',str(tmp_path),'--reference-id','RUN_4','--stage','plan'])
    assert received['reference_id']=='RUN_4'


def test_preparation_cli_lists_and_refuses_input_only_reference(tmp_path,capsys):
    spec=importlib.util.spec_from_file_location('reference_preparation',Path('examples/reference_pilot/prepare_data.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'RUN_1').mkdir()
    with h5py.File(tmp_path/'RUN_1/permeability.h5','w') as handle:handle.create_dataset('permeability',data=[1.])
    assert module.main(['--dataset-root',str(tmp_path),'--list-references'])==0
    assert 'input_only' in capsys.readouterr().out
    with pytest.raises(SystemExit):module.main(['--dataset-root',str(tmp_path),'--reference-id','RUN_1',
        '--cnn1-dir','missing','--release25-repo','missing'])
