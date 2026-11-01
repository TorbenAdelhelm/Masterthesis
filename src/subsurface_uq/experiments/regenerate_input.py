"""Regenerate a compact-study input and verify its native array fingerprint.

Uses the stored law, exact readings, seed recipe and original batch shape.
Original source maps, prepared model inputs and LGCNN weights are not needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid
import numpy as np

from ..io import sha256_file
from ..sampling.gstools_reference import GSToolsReferenceSpec
from . import controlled_field_benchmark as b
from .compact_fields import generator_sources,numerical_runtime,make_batch,_cleanup


def regenerate(study_dir,profile,n_measurements,sample_index,output_field,*,trial=0,design='balanced-000'):
    study=Path(study_dir).resolve();output=Path(output_field).resolve()
    manifest=json.loads((study/'manifest.json').read_text())
    cfg=manifest['configuration']
    if not cfg.get('compact_storage'):raise ValueError('regeneration requires a compact study')
    if profile not in cfg['scenarios']:raise ValueError('profile is not in the selected study')
    if (isinstance(trial,bool) or not isinstance(trial,int) or not 0<=trial<cfg['n_truths']
            or isinstance(sample_index,bool) or not isinstance(sample_index,int) or not 0<=sample_index<cfg['n_samples']):
        raise ValueError('valid global sample index and truth index required')
    if isinstance(n_measurements,bool) or not isinstance(n_measurements,int) or n_measurements not in cfg['measurement_counts']:
        raise ValueError('unknown observation count')
    b._name(profile,'profile');b._name(design,'design')
    root=study/profile/f'truth-{trial:04d}'
    if not b._completed(root,'recipe_files.json'):raise ValueError('compact regeneration recipe missing')
    recipe=json.loads((root/'input_recipe.json').read_text())
    if recipe['generator_source_sha256']!=generator_sources():
        raise ValueError('generator code differs from the recorded recipe; use its original code version')
    if recipe['numerical_runtime']!=numerical_runtime():
        raise ValueError('numerical runtime differs from the recorded recipe; use its recorded environment')
    candidates=[d for d in recipe['designs'] if n_measurements in d['counts']
                and (n_measurements==0 or d['design']==design)]
    if len(candidates)!=1:raise ValueError('unknown or ambiguous observation design')
    directory=root/candidates[0]['case_paths'][str(n_measurements)]
    if not b._completed(directory,'result_files.json'):
        raise ValueError('complete the temperature case before standalone input regeneration')
    records=json.loads((directory/'prediction_records.json').read_text())
    record=records[sample_index]
    if record['sample_index']!=sample_index:raise ValueError('recorded global sample index changed')
    sidecar=output.with_suffix(output.suffix+'.json')
    if output.suffix!='.npy' or output.exists() or sidecar.exists():
        raise ValueError('choose a fresh .npy output file and sidecar')
    output.parent.mkdir(parents=True,exist_ok=True)
    scratch=output.parent/('.regenerate-'+uuid.uuid4().hex)
    sampler=None
    try:
        sampler=b._prior(recipe['configuration'],GSToolsReferenceSpec(**recipe['law']['covariance']))
        start=sample_index//cfg['compact_storage']['batch_size']*cfg['compact_storage']['batch_size']
        make_batch(recipe,root,scratch,start,sampler)
        values=np.load(scratch/'cases'/directory.relative_to(root)/'permeability_samples.npy',mmap_mode='r')
        field=np.array(values[sample_index-start],copy=True);del values
        actual=b.array_hash(field)
        if actual!=record['input_sha256']:
            raise ValueError('regenerated input fingerprint differs; use the original numerical environment')
        temporary=scratch/'regenerated.npy';np.save(temporary,field,allow_pickle=False);temporary.replace(output)
        result={'status':'verified','reference_id':recipe.get('reference_id'),'profile':profile,
            'truth_index':trial,'design':'prior' if n_measurements==0 else design,'n_measurements':n_measurements,
            'sample_index':sample_index,'seed':record['seed'],'native_array_sha256':actual,
            'recipe_sha256':sha256_file(root/'input_recipe.json'),'shape':list(field.shape),'dtype':str(field.dtype)}
        b._atomic(sidecar,result)
        return result
    finally:
        del sampler
        _cleanup(scratch,output.parent)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir',required=True,help='Selected RUN output directory containing manifest.json')
    parser.add_argument('--profile',required=True)
    parser.add_argument('--n-measurements',type=int,required=True)
    parser.add_argument('--sample-index',type=int,required=True,help='Zero-based global MC sample index')
    parser.add_argument('--truth-index',type=int,default=0)
    parser.add_argument('--design',default='balanced-000')
    parser.add_argument('--output-field',required=True,help='Fresh .npy path; needs one batch of temporary free space')
    args=parser.parse_args(argv)
    result=regenerate(args.study_dir,args.profile,args.n_measurements,args.sample_index,args.output_field,
                      trial=args.truth_index,design=args.design)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
