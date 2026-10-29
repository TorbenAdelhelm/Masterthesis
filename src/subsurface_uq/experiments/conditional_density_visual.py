"""A paired native measurement-density study with complete spatial galleries."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
from numpy.lib.format import open_memmap
import yaml

from ..io import sha256_file,git_head,runtime_versions
from ..sampling.controlled_fields import KnownMeanConditioner,physical_permeability
from ..sampling.conditional_density import condition_prefixes
from ..sampling.gstools_reference import GSToolsReferenceSpec,GSToolsReferenceFFTSampler
from ..sampling.measurement_density import coverage_locations
from ..surrogates.frozen_temperature import load_frozen_temperature_model
from .structural_profiles import verify_artifact
from .controlled_field_benchmark import seed_for,array_hash,_atomic,_commit,_completed,_verify
from .reference_conditioning import _relative_path,_positive_integer,_identity


_KEYS={'structural_profile_file','profile_name','measurement_counts','n_samples','candidate_stride_cells',
       'seed','embedding_shape','workers','chunk_size','model_config','model_asset_root','torch_threads','plot_stride_cells'}


def load_config(path):
    source=Path(path).resolve();cfg=yaml.safe_load(source.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict) or set(cfg)!=_KEYS:raise ValueError('declare exactly the density-visual study settings')
    for key in ['n_samples','candidate_stride_cells','workers','chunk_size','torch_threads','plot_stride_cells']:
        _positive_integer(cfg[key],key)
    _positive_integer(cfg['seed'],'seed',zero=True)
    counts=cfg['measurement_counts']
    if (not isinstance(counts,list) or not counts or sorted(set(counts))!=counts
            or any(_positive_integer(n,'observations')!=n for n in counts) or cfg['n_samples']!=8):
        raise ValueError('increasing positive observation counts and exactly eight samples required')
    artifact,files=verify_artifact(_relative_path(cfg['structural_profile_file'],source.parent))
    if cfg['profile_name'] not in artifact['profiles']:raise ValueError('unknown structural profile')
    shape=artifact['native_shape'];embedding=cfg['embedding_shape']
    if (not isinstance(embedding,list) or len(embedding)!=2
            or any(isinstance(n,bool) or not isinstance(n,int) or n<3*m or n&(n-1) for n,m in zip(embedding,shape))):
        raise ValueError('explicit common embedding needs powers of two >=3 times the crop')
    if artifact['cell_size_m']!=5.:raise ValueError('native 5 m structural profile required')
    law=artifact['profiles'][cfg['profile_name']]['law'];spec=GSToolsReferenceSpec(**law['covariance'])
    indices,design=coverage_locations(shape,counts[-1],stride=cfg['candidate_stride_cells'],
        seed=seed_for(cfg['seed'],'observation-layout'),kind='balanced')
    system=KnownMeanConditioner(shape,spec,indices,mean_log10=law['mean_log10_k'],cell_size_m=5.)
    for file in [source,_relative_path(cfg['model_config'],source.parent),
                 Path(__file__).parents[1]/'visualization/conditional_density_report.html',
                 *sorted(Path(__file__).parents[1].rglob('*.py'))]:
        files[str(file)]=sha256_file(file)
    return source,cfg,artifact,files,system,design


def _check_sources(manifest,current):
    if current!=manifest['source_sha256']:raise ValueError('execution/configuration/source context changed; preserve the run and use a fresh output')
    _verify(Path(),current)


def plan(source,cfg,artifact,files,system,design,output):
    if output.exists():raise ValueError('a fresh study output is required')
    output.mkdir(parents=True)
    manifest={'schema_version':1,'study_id':_identity({'configuration':cfg,'sources':files},'density-visual-'),
        'status':'planned','configuration':cfg,'profile_set_id':artifact['profile_set_id'],
        'law':artifact['profiles'][cfg['profile_name']]['law'],'shape':artifact['native_shape'],'cell_size_m':5.,
        'source_sha256':files,'code_commit':git_head(source.parent),'runtime':runtime_versions(),
        'planned_lgcnn_evaluations':1+len(cfg['measurement_counts'])*cfg['n_samples'],
        'observation_condition_number':system.condition_number,'design':design,
        'nested_locations':True,'independent_hidden_truths':1,'observation_error_variance':0.,
        'sample_filtering':None,'geometry_cache_bytes':0,
        'interpretation':'one synthetic known-law realization; paired illustrative density response, no calibration or geological inference'}
    _atomic(output/'manifest.json',manifest)
    _atomic(output/'observations.json',{'indices':system.indices.tolist(),'design':design})
    return manifest


def inputs(cfg,manifest,system,output):
    if _completed(output,'input_files.json'):return
    manifest['status']='generating_inputs';_atomic(output/'manifest.json',manifest)
    sampler=GSToolsReferenceFFTSampler(replace(system.spec,sigma_R=1.),system.shape,cell_size_m=5.,
        workers=cfg['workers'],embedding_shape=cfg['embedding_shape'])
    truth_seed=seed_for(cfg['seed'],'hidden-truth')
    truth=system.mean_log10+system.spec.sigma_R*sampler.sample_residual(truth_seed)
    np.save(output/'truth_log10.npy',truth);np.save(output/'truth_permeability.npy',physical_permeability(truth))
    readings=truth[tuple(system.indices.T)]
    data=json.loads((output/'observations.json').read_text());data.update(observed_log10_k=readings.tolist(),hidden_truth_seed=truth_seed)
    _atomic(output/'observations.json',data)
    prior=open_memmap(output/'prior_unit.npy',mode='w+',dtype=np.float64,shape=(cfg['n_samples'],*system.shape))
    seeds=[]
    for i in range(cfg['n_samples']):
        seed=seed_for(cfg['seed'],'posterior',i);seeds.append(seed)
        prior[i]=sampler.sample_residual(seed);prior.flush()
        print(f"Gaussian driving draw {i+1}/{cfg['n_samples']}",flush=True)
    directories={n:output/f'n-{n:04d}' for n in cfg['measurement_counts']}
    last=[-1]
    def progress(done,total):
        fraction=int(10*done/total)
        if fraction>last[0]:
            last[0]=fraction;print(f'Exact nested conditioning: {100*done/total:.0f}%',flush=True)
    variances=condition_prefixes(system,readings,prior,cfg['measurement_counts'],directories,
        chunk_size=cfg['chunk_size'],progress=progress)
    metadata={**sampler.metadata,'spec':{**sampler.metadata['spec'],'marginal':'zero-mean unit Gaussian driving field',
        'assumption_status':'declared structural prior; independent truth and posterior draws'}}
    _atomic(output/'sampler.json',{'sampler':metadata,'posterior_seeds':seeds,'truth_seed':truth_seed,
        'shared_prior_draws_across_counts':True,'covariance_fixed_across_counts':True})
    for n,path in directories.items():
        fields=np.load(path/'permeability_samples.npy',mmap_mode='r');errors=[];hashes=[]
        for field in fields:
            error=float(np.max(np.abs(np.log10(field[tuple(system.indices[:n].T)].astype(float))-readings[:n])))
            if error>1.e-5:raise ArithmeticError('exact readings not preserved; stop without replacing draws')
            errors.append(error);hashes.append(array_hash(field))
        _atomic(path/'input_summary.json',{'n_measurements':n,'n_samples':cfg['n_samples'],
            'input_sha256':hashes,'posterior_seeds':seeds,'maximum_exact_log10_errors':errors,
            'analytic_average_log10_variance':variances[n],'indices':system.indices[:n].tolist()})
        del fields
    del prior,truth,sampler
    files=['truth_log10.npy','truth_permeability.npy','observations.json','prior_unit.npy','sampler.json']
    files += [str(p.relative_to(output)) for d in directories.values() for p in d.iterdir() if p.is_file()]
    _commit(output,files,'input_files.json')
    manifest['status']='inputs_ready';_atomic(output/'manifest.json',manifest)


def temperatures(source,cfg,manifest,output,model_factory):
    if not _completed(output,'input_files.json'):raise ValueError('complete and verify all inputs before model propagation')
    already_complete=manifest['status']=='complete'
    model=model_factory(_relative_path(cfg['model_config'],source.parent),
        asset_root=_relative_path(cfg['model_asset_root'],source.parent),torch_threads=cfg['torch_threads'])
    if tuple(model.input_shape)!=tuple(manifest['shape']) or model.cell_size_m!=5.:raise ValueError('model/profile grid mismatch')
    if manifest.get('model') is not None and manifest['model']!=model.metadata:raise ValueError('model context changed')
    manifest.update(status='propagating',model=model.metadata);_atomic(output/'manifest.json',manifest)
    truth=np.load(output/'truth_permeability.npy',mmap_mode='r')
    if _completed(output,'truth_temperature_files.json'):
        reference=np.load(output/'truth_temperature.npy',mmap_mode='r')
    else:
        start=time.monotonic();reference=np.asarray(model.predict_temperature(np.array(truth,copy=True)),dtype=np.float32)
        if reference.ndim!=2 or not np.all(np.isfinite(reference)):raise ValueError('invalid reference temperature')
        np.save(output/'truth_temperature.npy',reference)
        _atomic(output/'truth_temperature.json',{'input_sha256':array_hash(truth),'temperature_sha256':array_hash(reference),
            'seconds':time.monotonic()-start,'normalization':model.normalization_diagnostics(truth)})
        _commit(output,['truth_temperature.npy','truth_temperature.json'],'truth_temperature_files.json')
        print('LGCNN hidden reference complete',flush=True)
    for n in cfg['measurement_counts']:
        target=output/f'n-{n:04d}'
        if _completed(target,'temperature_files.json'):continue
        fields=np.load(target/'permeability_samples.npy',mmap_mode='r')
        inputs_summary=json.loads((target/'input_summary.json').read_text())
        record_path=target/'prediction_records.json';records=json.loads(record_path.read_text()) if record_path.exists() else []
        path=target/'temperature_samples.npy'
        if len(records)>cfg['n_samples']:raise ValueError('invalid saved query count')
        if path.exists():
            store=np.load(path,mmap_mode='r+')
            if store.shape!=(cfg['n_samples'],*reference.shape) or store.dtype!=np.float32:raise ValueError('temperature store changed')
        else:
            if records:raise ValueError('query records lack saved temperatures')
            store=open_memmap(path,mode='w+',dtype=np.float32,shape=(cfg['n_samples'],*reference.shape))
        for i,r in enumerate(records):
            if (r['sample_index']!=i or r['input_sha256']!=array_hash(fields[i])
                    or r['temperature_sha256']!=array_hash(store[i]) or r['n_measurements']!=n
                    or r['posterior_seed']!=inputs_summary['posterior_seeds'][i]):raise ValueError('completed model query changed')
        for i in range(len(records),cfg['n_samples']):
            if array_hash(fields[i])!=inputs_summary['input_sha256'][i]:raise ValueError('model input changed')
            start=time.monotonic();temp=np.asarray(model.predict_temperature(np.array(fields[i],copy=True)),dtype=np.float32)
            if temp.shape!=reference.shape or not np.all(np.isfinite(temp)):raise ValueError('invalid model output; draw not replaced')
            store[i]=temp;store.flush()
            records.append({'sample_index':i,'n_measurements':n,'input_sha256':array_hash(fields[i]),
                'temperature_sha256':array_hash(temp),'posterior_seed':inputs_summary['posterior_seeds'][i],
                'seconds':time.monotonic()-start,'normalization':model.normalization_diagnostics(fields[i])})
            _atomic(record_path,records)
            print(f'LGCNN n={n}: {i+1}/{cfg["n_samples"]} ({records[-1]["seconds"]:.2f} s)',flush=True)
        del store,fields
        _commit(target,['temperature_samples.npy','prediction_records.json'],'temperature_files.json')
    _verify(Path(),model.metadata.get('files_sha256',{}))
    manifest.update(status='complete' if already_complete else 'propagated',completed_lgcnn_evaluations=manifest['planned_lgcnn_evaluations'],
        output_shape=list(reference.shape),background_temperature=model.background_temperature)
    _atomic(output/'manifest.json',manifest)


def run(config_path,output_dir,stage='all',*,model_factory=load_frozen_temperature_model):
    source,cfg,artifact,files,system,design=load_config(config_path);output=Path(output_dir).resolve()
    if stage in {'plan','all'}:manifest=plan(source,cfg,artifact,files,system,design,output)
    else:
        manifest=json.loads((output/'manifest.json').read_text());_check_sources(manifest,files)
        if cfg!=manifest['configuration']:raise ValueError('configuration changed')
        if manifest.get('report_sha256'):_verify(output,manifest['report_sha256'])
    if stage in {'inputs','all'}:inputs(cfg,manifest,system,output)
    if stage in {'temperatures','all'}:temperatures(source,cfg,manifest,output,model_factory)
    if stage in {'report','all'}:
        if manifest['status'] not in {'propagated','complete'}:raise ValueError('complete LGCNN propagation first')
        if not _completed(output,'input_files.json') or not _completed(output,'truth_temperature_files.json'):
            raise ValueError('missing verified native inputs or hidden reference')
        for n in cfg['measurement_counts']:
            if not _completed(output/f'n-{n:04d}','temperature_files.json'):raise ValueError('missing verified temperature results')
        from ..visualization.conditional_density import report
        report(cfg,manifest,output)
    _check_sources(manifest,files)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--stage',choices=['plan','inputs','temperatures','report','all'],default='all')
    args=parser.parse_args(argv);run(args.config,args.output_dir,args.stage)


if __name__=='__main__':main()
