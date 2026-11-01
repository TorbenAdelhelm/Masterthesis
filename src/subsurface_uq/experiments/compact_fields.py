"""Bounded-disk controlled UQ: batched inputs, native outputs, exact recipes.

Successful predictions commit temperature, preview and selected-input hashes
before disposable batch files are removed. Global sample IDs never restart at
zero within a batch. No field filtering, lossy scientific arrays or new law.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import numpy as np
from numpy.lib.format import open_memmap
from threadpoolctl import threadpool_limits

from ..io import runtime_versions,sha256_file
from ..sampling.conditional_density import prefix_moments,condition_sample_batch
from ..sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from ..sampling.gstools_reference import GSToolsReferenceSpec
from . import controlled_field_benchmark as b


def execution_runtime():
    return runtime_versions(('numpy','scipy','gstools','torch','noise','PyYAML','matplotlib','threadpoolctl'))


def numerical_runtime():
    value=runtime_versions(('numpy','scipy','gstools','threadpoolctl'))
    return {**{k:v for k,v in value.items() if k!='platform'},'input_batch_blas_threads':1}


def generator_sources():
    base=Path(__file__).parents[1]
    names=['experiments/compact_fields.py','experiments/controlled_field_benchmark.py',
           'sampling/conditional_density.py','sampling/controlled_fields.py','sampling/gstools_reference.py']
    return {name:sha256_file(base/name) for name in names}


def storage_estimate(cfg,cases):
    settings=cfg['compact_storage'];m=int(np.prod(cfg['shape']))
    t=int(np.prod(settings['temperature_shape']));p=min(settings['preview_size'],*cfg['shape'])
    n=cfg['n_samples'];r=settings['retained_input_samples']
    layouts=len(cfg['coverage_scenarios'])*cfg['n_designs']
    cases_per_truth=1+layouts*(len(cfg['measurement_counts'])-1)
    truths=cases//cases_per_truth
    arrays={'temperature_ensembles':cases*n*t*4,'selected_permeability':cases*r*m*4,
        'input_previews':cases*n*p*p*4,'input_moments':cases*2*m*4,
        'temperature_products':cases*(6+2*len(cfg['risk_thresholds_C']))*t*4,
        'hidden_truths':truths*(12*m+4*t)}
    base=sum(arrays.values())
    batch=settings['batch_size']*m*(8+4*cases_per_truth)
    # PNG compression is not assumed here. Actual coloured images are usually
    # substantially smaller than the raw RGBA bounds (2% encoding allowance).
    native=int(np.ceil(cases*r*(m+t)*4*1.02))
    atlases=int(np.ceil(cases*6*((n+9)//10)*10*p*p*4*1.02))
    preview_scratch_one_case=6*n*p*p*4
    return {'array_bytes_by_role':arrays,'retained_array_bytes':base,
        'native_image_rgba_bound_bytes':native,'browser_atlas_rgba_bound_bytes':atlases,
        'estimated_retained_upper_bytes':base+native+atlases,
        'peak_batch_scratch_bytes':batch,'geometry_cache_bytes':0,
        'estimated_peak_upper_bytes':max(base+batch,base+native+atlases+preview_scratch_one_case),
        'note':'native output shape is checked before ensemble propagation; estimates include all kernels and baselines; allow filesystem overhead'}


def _cleanup(path,parent):
    parent=Path(parent).resolve();path=Path(path)
    if path.exists():
        target=path.resolve()
        if target==parent or not target.is_relative_to(parent):
            raise ValueError('batch cleanup target escapes its study directory')
        shutil.rmtree(target)


def _configuration(cfg):
    keys=['shape','cell_size_m','measurement_counts','n_samples','seed','workers','chunk_size',
          'coverage_scenarios','n_designs','candidate_stride_cells','cluster_scale_fraction','compact_storage']
    keys += [k for k in ['profile_pairing_group','paired_embedding_shape'] if k in cfg]
    return {k:cfg[k] for k in keys}


def prepare(cfg,output,manifest):
    manifest['status']='preparing_inputs';b._atomic(output/'manifest.json',manifest)
    for name,law in cfg['scenarios'].items():
        spec=GSToolsReferenceSpec(**law['covariance']);sampler=None
        for trial in range(cfg['n_truths']):
            root=output/name/f'truth-{trial:04d}';root.mkdir(parents=True,exist_ok=True)
            if b._completed(root,'recipe_files.json'):
                b._completed(root,'truth_files.json')
                for _,_,directory in b._cases(cfg,name,trial,output):
                    if not b._completed(directory,'input_files.json'):raise ValueError('compact input context missing')
                continue
            if not b._completed(root,'truth_files.json'):
                if sampler is None:
                    sampler=b._prior(cfg,spec)
                    sampler.metadata['spec'].update(marginal='zero-mean unit Gaussian driving field',
                        assumption_status='controlled structural prior; no site-error calibration')
                    b._atomic(output/name/'prior.json',{**sampler.metadata,'law':law})
                seed=b.seed_for(cfg['seed'],b._stream(cfg,name),trial,'hidden-truth')
                logs=law['mean_log10_k']+spec.sigma_R*sampler.sample_residual(seed)
                np.save(root/'truth_log10.npy',logs);np.save(root/'truth_permeability.npy',physical_permeability(logs))
                del logs
                b._atomic(root/'truth.json',{'seed':seed,'trial':trial,'scenario':name,
                    'role':'independent hidden truth; only selected observations are revealed'})
                b._commit(root,['truth_log10.npy','truth_permeability.npy','truth.json'],'truth_files.json')
            truth=np.load(root/'truth_log10.npy',mmap_mode='r');designs=[];first=True
            for design,kind,index in b._designs(cfg,name):
                indices,metadata=b._locations(cfg,name,kind,index)
                system=KnownMeanConditioner(cfg['shape'],spec,indices,mean_log10=law['mean_log10_k'],cell_size_m=cfg['cell_size_m'])
                geometry=output/name/'geometry'/design;geometry.mkdir(parents=True,exist_ok=True)
                if not b._completed(geometry):
                    b._atomic(geometry/'design.json',{'indices':indices.tolist(),'metadata':metadata,
                        'condition_number':system.condition_number,'known_mean_log10':law['mean_log10_k']})
                    b._commit(geometry,['design.json'])
                counts=cfg['measurement_counts'] if first else cfg['measurement_counts'][1:]
                directories={n:root/'prior' if n==0 else root/design/f'n-{n:04d}' for n in counts}
                readings=np.asarray(truth[tuple(indices.T)])
                pending=[n for n,directory in directories.items() if not b._completed(directory,'input_files.json')]
                if pending:
                    averages=prefix_moments(system,readings,pending,directories,chunk_size=cfg['chunk_size'])
                    for n in pending:
                        b._prepare_case(cfg,name,trial,'prior' if n==0 else design,n,directories[n],truth,
                            system,None,average_variance=averages[n])
                designs.append({'design':design,'indices':indices.tolist(),'readings':readings.tolist(),
                    'counts':counts,'case_paths':{str(n):d.relative_to(root).as_posix() for n,d in directories.items()}})
                first=False
            del truth
            recipe={'schema_version':1,'profile':name,'trial':trial,'reference_id':cfg.get('reference_id'),
                'law':law,'configuration':_configuration(cfg),'designs':designs,
                'numerical_runtime':numerical_runtime(),'generator_source_sha256':generator_sources(),
                'selection_policy':'first fixed sample indices; no selection by field appearance or model result',
                'regeneration':'rebuild the original global batch and verify the recorded native-input fingerprint'}
            b._atomic(root/'input_recipe.json',recipe);b._commit(root,['input_recipe.json'],'recipe_files.json')
            print(f'Compact analytic inputs ready: {name}/truth-{trial:04d}',flush=True)
        del sampler
    manifest['status']='inputs_ready';b._atomic(output/'manifest.json',manifest)


def make_batch(recipe,root,batch,start,sampler):
    # Fixed BLAS execution makes input fingerprints independent of the model's
    # torch thread settings and ambient OPENBLAS/MKL thread limits.
    with threadpool_limits(limits=1,user_api='blas'):
        return _make_batch(recipe,root,batch,start,sampler)


def _make_batch(recipe,root,batch,start,sampler):
    """Commit one disposable input batch with its original RHS shape/order."""
    cfg=recipe['configuration'];stop=min(start+cfg['compact_storage']['batch_size'],cfg['n_samples'])
    context={'recipe_sha256':sha256_file(root/'input_recipe.json'),'start':start,'stop':stop}
    if batch.exists():
        marker=batch/'context.json'
        if not marker.exists() or json.loads(marker.read_text())!=context:
            raise ValueError('compact scratch batch context changed')
        if b._completed(batch,'batch_files.json'):return stop
    else:
        batch.mkdir(parents=True);b._atomic(batch/'context.json',context)
    marker=batch/'prior_records.json';records=json.loads(marker.read_text()) if marker.exists() else []
    path=batch/'prior.npy'
    if records and not path.exists():raise ValueError('recorded compact prior scratch is missing')
    prior=open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float64,shape=(stop-start,*cfg['shape']))
    if prior.shape!=(stop-start,*cfg['shape']) or prior.dtype!=np.float64 or len(records)>len(prior):
        raise ValueError('compact prior scratch shape/dtype changed')
    for local in range(len(prior)):
        i=start+local;seed=b.seed_for(cfg['seed'],b._stream(cfg,recipe['profile']),recipe['trial'],'posterior',i)
        if local<len(records):
            if records[local]!={'sample_index':i,'seed':seed,'sha256':b.array_hash(prior[local])}:
                raise ValueError('completed compact prior draw changed')
        else:
            prior[local]=sampler.sample_residual(seed);prior.flush()
            records.append({'sample_index':i,'seed':seed,'sha256':b.array_hash(prior[local])});b._atomic(marker,records)
    files=['context.json','prior.npy','prior_records.json'];law=recipe['law']
    for design in recipe['designs']:
        system=KnownMeanConditioner(cfg['shape'],GSToolsReferenceSpec(**law['covariance']),np.array(design['indices']),
            mean_log10=law['mean_log10_k'],cell_size_m=cfg['cell_size_m'])
        directories={n:batch/'cases'/design['case_paths'][str(n)] for n in design['counts']}
        condition_sample_batch(system,design['readings'],prior,design['counts'],directories,chunk_size=cfg['chunk_size'])
        files.extend(str((d/'permeability_samples.npy').relative_to(batch)) for d in directories.values())
    del prior
    b._commit(batch,files,'batch_files.json')
    print(f'Compact native input batch ready: {recipe["profile"]}, samples {start+1}-{stop}',flush=True)
    return stop


def _retention_state(directory,cfg,records):
    settings=cfg['compact_storage'];p=min(settings['preview_size'],*cfg['shape'])
    arrays=[]
    for name,shape in [('permeability_selected.npy',(settings['retained_input_samples'],*cfg['shape'])),
                       ('log_permeability_previews.npy',(cfg['n_samples'],p,p))]:
        path=directory/name
        if records and not path.exists():raise ValueError('committed compact input retention is missing')
        values=open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,shape=shape)
        if values.shape!=shape or values.dtype!=np.float32:raise ValueError('compact retention shape/dtype changed')
        arrays.append(values)
    selected,previews=arrays
    for i,row in enumerate(records):
        if row.get('input_preview_sha256')!=b.array_hash(previews[i]):
            raise ValueError('completed compact input preview changed')
        if i<len(selected) and row['input_sha256']!=b.array_hash(selected[i]):
            raise ValueError('completed selected input changed')
    return selected,previews


def propagate(cfg,output,manifest,model):
    from ..visualization.ensemble_browser import _thumbnail
    settings=cfg['compact_storage']
    for name,law in cfg['scenarios'].items():
        sampler=None;probes=b._probes(cfg,name)
        for trial in range(cfg['n_truths']):
            root=output/name/f'truth-{trial:04d}'
            if not b._completed(root,'recipe_files.json'):raise ValueError('compact recipe missing')
            b._completed(root,'truth_files.json')
            recipe=json.loads((root/'input_recipe.json').read_text())
            truth=np.load(root/'truth_log10.npy',mmap_mode='r');truth_temperature=b._truth_temperature(root,model)
            if list(truth_temperature.shape)!=settings['temperature_shape']:
                raise ValueError('native model output differs from compact_storage.temperature_shape')
            states=[]
            for design,n,directory in b._cases(cfg,name,trial,output):
                if not b._completed(directory,'input_files.json'):raise ValueError('compact analytic input context missing')
                if b._completed(directory,'result_files.json'):continue
                records,store=b._prediction_state(directory,cfg,truth_temperature.shape)
                selected,previews=_retention_state(directory,cfg,records)
                context=json.loads((directory/'input_summary.json').read_text())
                states.append((directory,records,store,selected,previews,context))
            batch=root/'batch_scratch'
            for start in range(0,cfg['n_samples'],settings['batch_size']):
                stop=min(start+settings['batch_size'],cfg['n_samples'])
                if not any(len(state[1])<stop for state in states):
                    if batch.exists() and json.loads((batch/'context.json').read_text()).get('start')==start:
                        _cleanup(batch,root)
                    continue
                if sampler is None:sampler=b._prior(cfg,GSToolsReferenceSpec(**law['covariance']))
                make_batch(recipe,root,batch,start,sampler)
                for i in range(start,stop):
                    seed=b.seed_for(cfg['seed'],b._stream(cfg,name),trial,'posterior',i)
                    for directory,records,store,selected,previews,context in states:
                        fields=np.load(batch/'cases'/directory.relative_to(root)/'permeability_samples.npy',mmap_mode='r')
                        field=np.array(fields[i-start],copy=True);del fields
                        fingerprint=b.array_hash(field)
                        if len(records)>i:
                            if records[i]['input_sha256']!=fingerprint:raise ValueError('regenerated compact input changed')
                            continue
                        if len(records)!=i:raise ValueError('compact prediction sample order changed')
                        indices=np.asarray(context['indices'],dtype=int).reshape(-1,2)
                        readings=np.asarray(context['observed_log10_k'])
                        logs=np.log10(field.astype(float))
                        exact=float(np.max(np.abs(logs[tuple(indices.T)]-readings),initial=0.))
                        if exact>1.e-5:raise ArithmeticError('compact exact readings not preserved; draw not replaced')
                        started=time.monotonic();temp=np.asarray(model.predict_temperature(field),dtype=np.float32)
                        if temp.shape!=truth_temperature.shape or not np.all(np.isfinite(temp)):
                            raise ValueError('invalid compact LGCNN prediction; draw not replaced')
                        store[i]=temp;store.flush()
                        previews[i]=_thumbnail(logs,previews.shape[-1]);previews.flush()
                        if i<len(selected):selected[i]=field;selected.flush()
                        row={'sample_index':i,'seed':seed,'input_sha256':fingerprint,
                            'temperature_sha256':b.array_hash(temp),'input_preview_sha256':b.array_hash(previews[i]),
                            'input_probe_log10':logs[tuple(probes.T)].tolist(),'maximum_exact_log10_error':exact,
                            'input_log10_range':[float(logs.min()),float(logs.max())],
                            'temperature_range_C':[float(temp.min()),float(temp.max())],
                            'seconds':time.monotonic()-started,'normalization':model.normalization_diagnostics(field)}
                        records.append(row);b._atomic(directory/'prediction_records.json',records)
                        print(f'LGCNN compact {name}/truth-{trial:04d}/{directory.name}: {i+1}/{cfg["n_samples"]}',flush=True)
                        del logs,field,temp
                _cleanup(batch,root)
            for state in states:
                for values in state[2:5]:values.flush()
            # Close all writable maps before ledgers/reports, also on Windows.
            del states
            if 'store' in locals():del store,selected,previews
            if 'state' in locals():del state
            for design,n,directory in b._cases(cfg,name,trial,output):
                b._summarize(cfg,name,trial,design,n,directory,truth_temperature,truth,model,probes)
            del truth
        del sampler
    b._verify(Path(),model.metadata.get('files_sha256',{}))
    manifest.update(status='propagated',model_evaluations=manifest['planned_model_evaluations'])
    b._atomic(output/'manifest.json',manifest)
