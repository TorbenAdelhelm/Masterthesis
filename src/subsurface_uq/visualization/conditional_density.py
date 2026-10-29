"""Spatial atlases and an offline gallery for every density-study realization."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

from ..io import sha256_file
from ..statistics.online import OnlineFieldStatistics


def _save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _geometry(shape,temperature_shape,cell):
    if any(a<b or (a-b)%2 for a,b in zip(shape,temperature_shape)):
        raise ValueError('temperature crop must align symmetrically with the native input')
    h,w=shape;th,tw=temperature_shape;row=(h-th)//2*cell/1000;col=(w-tw)//2*cell/1000
    return [0,w*cell/1000,0,h*cell/1000],[col,col+tw*cell/1000,row,row+th*cell/1000]


def _draw(ax,a,extent,limits,cmap,title):
    im=ax.imshow(a,origin='lower',extent=extent,vmin=limits[0],vmax=limits[1],cmap=cmap,interpolation='antialiased')
    ax.set_title(title,fontsize=9);ax.set_xlabel('Column [km]',fontsize=8);ax.set_ylabel('Row [km]',fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def _outline(ax,extent):
    ax.add_patch(Rectangle((extent[0],extent[2]),extent[1]-extent[0],extent[3]-extent[2],
        fill=False,edgecolor='black',linestyle='--',linewidth=.7))


def _metrics(output,cfg,manifest):
    truth=np.load(output/'truth_log10.npy',mmap_mode='r');tstar=np.load(output/'truth_temperature.npy',mmap_mode='r')
    limits={'input':[float(truth.min()),float(truth.max())],'output':[float(tstar.min()),float(tstar.max())],
        'input_error':[0.,0.],'output_error':[0.,0.],'input_deviation':[0.,0.],'output_deviation':[0.,0.]}
    rows=[];local={};max_tsd=0.
    for n in cfg['measurement_counts']:
        target=output/f'n-{n:04d}';fields=np.load(target/'permeability_samples.npy',mmap_mode='r');temps=np.load(target/'temperature_samples.npy',mmap_mode='r')
        ys=OnlineFieldStatistics();ts=OnlineFieldStatistics();draws=[];local[n]={k:[0.,0.] for k in limits if 'error' in k or 'deviation' in k}
        for i,(field,temp) in enumerate(zip(fields,temps)):
            y=np.log10(field.astype(float));e=y-truth;te=temp.astype(float)-tstar
            ys.update(y);ts.update(temp)
            limits['input'][0]=min(limits['input'][0],float(y.min()));limits['input'][1]=max(limits['input'][1],float(y.max()))
            limits['output'][0]=min(limits['output'][0],float(temp.min()));limits['output'][1]=max(limits['output'][1],float(temp.max()))
            for name,a in [('input_error',e),('output_error',te)]:
                maximum=float(np.max(np.abs(a)));local[n][name]=[-max(maximum,local[n][name][1]),max(maximum,local[n][name][1])]
            draws.append({'sample':i+1,'input_log10_rmse':float(np.sqrt(np.mean(e*e))),
                'temperature_rmse_C':float(np.sqrt(np.mean(te*te)))})
        ystat=ys.finalize();tstat=ts.finalize();max_tsd=max(max_tsd,float(tstat.std.max()))
        for name,a in [('ensemble_log10_mean',ystat.mean),('ensemble_log10_std',ystat.std),
                       ('temperature_mean',tstat.mean),('temperature_std',tstat.std)]:np.save(target/(name+'.npy'),a.astype(np.float32))
        for field,temp in zip(fields,temps):
            for name,a in [('input_deviation',np.log10(field.astype(float))-ystat.mean),('output_deviation',temp.astype(float)-tstat.mean)]:
                maximum=float(np.max(np.abs(a)));local[n][name]=[-max(maximum,local[n][name][1]),max(maximum,local[n][name][1])]
        for name,pair in local[n].items():
            maximum=max(pair[1],1.e-12);local[n][name]=[-maximum,maximum]
            limits[name]=[-max(limits[name][1],maximum),max(limits[name][1],maximum)]
        inputs=json.loads((target/'input_summary.json').read_text());ye=ystat.mean-truth;te=tstat.mean-tstar
        rows.append({'n_measurements':n,'n_samples':len(fields),'samples':draws,
            'analytic_input_sd_rms_log10':float(np.sqrt(inputs['analytic_average_log10_variance'])),
            'empirical_input_sd_rms_log10':float(np.sqrt(ystat.variance.mean())),
            'mean_log10_field_rmse':float(np.sqrt(np.mean(ye*ye))),
            'temperature_sd_rms_C':float(np.sqrt(tstat.variance.mean())),
            'temperature_mean_field_rmse_C':float(np.sqrt(np.mean(te*te))),
            'mean_individual_temperature_rmse_C':float(np.mean([d['temperature_rmse_C'] for d in draws])),
            'maximum_exact_log10_error':max(inputs['maximum_exact_log10_errors'])})
        del fields,temps,ys,ts,ystat,tstat
    limits['analytic_input_sd']=[0.,manifest['law']['covariance']['sigma_R']]
    limits['temperature_sd']=[0.,max(max_tsd,1.e-12)]
    return rows,limits,local


def _flat_maps(output,cfg,limits,local):
    truth=np.load(output/'truth_log10.npy',mmap_mode='r');tstar=np.load(output/'truth_temperature.npy',mmap_mode='r')
    roles=['input','output','input_error','output_error','input_deviation','output_deviation']
    for n in cfg['measurement_counts']:
        target=output/f'n-{n:04d}';dest=output/'figures'/f'n-{n:04d}';dest.mkdir(parents=True,exist_ok=True)
        fields=np.load(target/'permeability_samples.npy',mmap_mode='r');temps=np.load(target/'temperature_samples.npy',mmap_mode='r')
        ym=np.load(target/'ensemble_log10_mean.npy',mmap_mode='r');tm=np.load(target/'temperature_mean.npy',mmap_mode='r')
        for i,(field,temp) in enumerate(zip(fields,temps)):
            y=np.log10(field.astype(float));arrays=[y,temp,y-truth,temp.astype(float)-tstar,y-ym,temp.astype(float)-tm]
            for role,a in zip(roles,arrays):
                stride=cfg['plot_stride_cells'] if role.startswith('input') else 1
                small=a[::stride,::stride]
                cmap='viridis' if role=='input' else 'inferno' if role=='output' else 'RdBu_r'
                low,high=limits[role]
                plt.imsave(dest/f'sample-{i+1:02d}-{role}.png',small,origin='lower',cmap=cmap,vmin=low,vmax=high)
                if role in local[n]:
                    low,high=local[n][role]
                    plt.imsave(dest/f'sample-{i+1:02d}-{role}-detail.png',small,origin='lower',cmap=cmap,vmin=low,vmax=high)
        print(f'Spatial sample images: n={n}, all {len(fields)} input/output pairs',flush=True)
    legends=output/'figures/legends';legends.mkdir(exist_ok=True)
    for role,pair in limits.items():
        if role in {'analytic_input_sd','temperature_sd'}:continue
        fig,ax=plt.subplots(figsize=(5,.55),layout='constrained');fig.colorbar(
            plt.cm.ScalarMappable(norm=Normalize(*pair),cmap='viridis' if role=='input' else 'inferno' if role=='output' else 'RdBu_r'),
            cax=ax,orientation='horizontal');ax.tick_params(labelsize=8)
        fig.savefig(legends/f'{role}.png',dpi=120);plt.close(fig)
    for n,pairs in local.items():
        for role,pair in pairs.items():
            fig,ax=plt.subplots(figsize=(5,.55),layout='constrained');fig.colorbar(
                plt.cm.ScalarMappable(norm=Normalize(*pair),cmap='RdBu_r'),cax=ax,orientation='horizontal')
            ax.tick_params(labelsize=8);fig.savefig(legends/f'n-{n:04d}-{role}.png',dpi=120);plt.close(fig)


def _atlases(output,cfg,limits,local,extent_k,extent_t):
    truth=np.load(output/'truth_log10.npy',mmap_mode='r');tstar=np.load(output/'truth_temperature.npy',mmap_mode='r')
    for n in cfg['measurement_counts']:
        target=output/f'n-{n:04d}';fields=np.load(target/'permeability_samples.npy',mmap_mode='r');temps=np.load(target/'temperature_samples.npy',mmap_mode='r')
        for kind,ensemble,reference,extent in [('inputs',fields,truth,extent_k),('outputs',temps,tstar,extent_t)]:
            for contrast in ['shared','detail']:
                fig,axes=plt.subplots(4,4,figsize=(14,14),layout='constrained')
                role='input' if kind=='inputs' else 'output';error_role=role+'_error'
                error_limits=limits[error_role] if contrast=='shared' else local[n][error_role]
                artists=[]
                for i,field in enumerate(ensemble):
                    a=np.log10(field.astype(float)) if kind=='inputs' else field
                    row=i//2;col=(i%2)*2
                    stride=cfg['plot_stride_cells'] if kind=='inputs' else 1
                    im=_draw(axes[row,col],a[::stride,::stride],extent,limits[role],
                        'viridis' if kind=='inputs' else 'inferno',f'Sample {i+1}: '+('log10 k' if kind=='inputs' else 'T [C]'))
                    err=_draw(axes[row,col+1],(a-reference)[::stride,::stride],extent,error_limits,'RdBu_r',f'Sample {i+1}: error vs hidden reference')
                    if kind=='inputs':_outline(axes[row,col],extent_t);_outline(axes[row,col+1],extent_t)
                    artists=[im,err]
                fig.colorbar(artists[0],ax=axes[:,[0,2]].ravel().tolist(),shrink=.5,label='log10 k [m2]' if kind=='inputs' else 'Temperature [C]')
                fig.colorbar(artists[1],ax=axes[:,[1,3]].ravel().tolist(),shrink=.5,label='log10(sample / truth)' if kind=='inputs' else 'Sample - reference [C]')
                fig.suptitle(f'{n} exact observations | all eight {kind}\n'+
                    ('Shared error scale across all counts' if contrast=='shared' else f'DETAIL error scale for n={n} only')+
                    '\nRed: higher than reference; blue: lower. '+('Dashed square: temperature output domain.' if kind=='inputs' else 'Fixed centre crop; reference is the same frozen LGCNN applied to the hidden field.'),fontsize=12)
                fig.savefig(output/'figures'/f'n-{n:04d}-{kind}-{contrast}.png',dpi=150);plt.close(fig)


def _overview(output,cfg,limits,extent_k,extent_t):
    truth=np.load(output/'truth_log10.npy',mmap_mode='r');tstar=np.load(output/'truth_temperature.npy',mmap_mode='r')
    data=json.loads((output/'observations.json').read_text());indices=np.array(data['indices'])
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    for ax,a,extent,role,title in zip(axes,[truth,tstar],[extent_k,extent_t],['input','output'],['Hidden permeability field','Hidden-field LGCNN temperature']):
        im=_draw(ax,a,extent,limits[role],'viridis' if role=='input' else 'inferno',title)
        fig.colorbar(im,ax=ax,shrink=.7,label='log10 k [m2]' if role=='input' else 'T [C]')
    _outline(axes[0],extent_t);fig.suptitle('One fixed synthetic hidden reference for all observation counts')
    fig.savefig(output/'figures/reference.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(cfg['measurement_counts']),6,figsize=(19,3.1*len(cfg['measurement_counts'])),layout='constrained',squeeze=False)
    artists=[]
    for row,n in enumerate(cfg['measurement_counts']):
        target=output/f'n-{n:04d}';ym=np.load(target/'ensemble_log10_mean.npy',mmap_mode='r');tm=np.load(target/'temperature_mean.npy',mmap_mode='r')
        ys=np.load(target/'posterior_log10_std.npy',mmap_mode='r');ts=np.load(target/'temperature_std.npy',mmap_mode='r')
        arrays=[truth,ym-truth,ys,tm,tm-tstar,ts]
        roles=['input','input_error','analytic_input_sd','output','output_error','temperature_sd']
        titles=[f'{n} observations','Mean log-k error','Exact conditional log-k SD','Mean LGCNN temperature','Mean temperature error','8-draw temperature SD']
        artists=[]
        for j,(a,role,title) in enumerate(zip(arrays,roles,titles)):
            stride=cfg['plot_stride_cells'] if j<3 else 1
            artists.append(_draw(axes[row,j],a[::stride,::stride],extent_k if j<3 else extent_t,
                limits[role],'RdBu_r' if 'error' in role else 'viridis' if j<3 else 'inferno' if role=='output' else 'magma',title))
            if j<3:_outline(axes[row,j],extent_t)
        xy=(indices[:n]+.5)*.005
        axes[row,0].scatter(xy[:,1],xy[:,0],s=2.5,c='black',alpha=.65)
    for j,label in enumerate(['log10 k [m2]','log10(mean / truth)','log10 SD','T [C]','Mean - reference [C]','SD [C]']):
        fig.colorbar(artists[j],ax=axes[:,j].tolist(),shrink=.45,label=label)
    fig.suptitle('Observation-density study | one fixed hidden field, matched draws, nested locations\nShared colour scales down each column. Errors compare with the hidden reference; SD shows spread, not error.',fontsize=14)
    fig.savefig(output/'figures/density-overview.png',dpi=150);plt.close(fig)


def _html(output,cfg,manifest,rows,limits,local,extents):
    data={'counts':cfg['measurement_counts'],'samples':cfg['n_samples'],'rows':rows,'limits':limits,
          'local':local,'extents':extents,'indices':json.loads((output/'observations.json').read_text())['indices'],
          'shape':manifest['shape']}
    # Purely local document: no remote dependencies or network requests.
    template=Path(__file__).with_name('conditional_density_report.html').read_text(encoding='utf-8')
    document=template.replace('__STUDY_DATA__',json.dumps(data,allow_nan=False).replace('</','<\\/'))
    document=document.replace('__STUDY_ID__',html.escape(manifest['study_id'])).replace('__PROFILE__',html.escape(cfg['profile_name']))
    (output/'report.html').write_text(document,encoding='utf-8')


def report(cfg,manifest,output):
    output=Path(output);(output/'figures').mkdir(exist_ok=True)
    rows,limits,local=_metrics(output,cfg,manifest)
    truth=np.load(output/'truth_log10.npy',mmap_mode='r');temp=np.load(output/'truth_temperature.npy',mmap_mode='r')
    extent_k,extent_t=_geometry(truth.shape,temp.shape,5.)
    _flat_maps(output,cfg,limits,local);_atlases(output,cfg,limits,local,extent_k,extent_t)
    _overview(output,cfg,limits,extent_k,extent_t);_html(output,cfg,manifest,rows,limits,local,[extent_k,extent_t])
    summary={'study_id':manifest['study_id'],'profile_name':cfg['profile_name'],'counts':rows,
        'shared_colour_limits':limits,'local_detail_colour_limits':local,'extents_km':{'input':extent_k,'temperature':extent_t},
        'input_error_definition':'log10(K_sample/K_hidden); positive means higher permeability',
        'output_error_definition':'T_sample - T_hidden_LGCNN in C; positive means warmer',
        'ensemble_deviation_definition':'sample minus empirical eight-draw mean, in log10 k or temperature',
        'uncertainty_definition':'analytical conditional log-k SD or empirical temperature SD (ddof=1)',
        'image_input_point_stride_cells':cfg['plot_stride_cells'],'temperature_images_native_resolution':True,
        'native_inputs_and_outputs_retained':True,'independent_hidden_truths':1,
        'limits':'single synthetic truth/layout, eight matched samples; no calibration, geological validation or tail-convergence claim'}
    _save(output/'summary.json',summary)
    text=['# Spatial observation-density study','',f"Profile: {cfg['profile_name']}. Study: {manifest['study_id']}.",'',
        f"Completed {len(cfg['measurement_counts'])*cfg['n_samples']} conditional LGCNN queries plus one hidden-reference query. Exact synthetic observations, fixed kernel/mean/variance, nested broad coverage and matched random draws across counts.",'',
        '[Open the interactive spatial gallery](report.html). It contains every input and output, signed reference errors, ensemble deviations, shared/detail colour scales and links to all native arrays.','',
        '![Spatial density overview](figures/density-overview.png)','',
        '| Observations | Exact input SD RMS [log10] | Input ensemble-mean error RMS [log10] | Temperature SD RMS [C] | Temperature ensemble-mean error RMS [C] |',
        '|---:|---:|---:|---:|---:|']
    for r in rows:text.append(f"| {r['n_measurements']} | {r['analytic_input_sd_rms_log10']:.6g} | {r['mean_log10_field_rmse']:.6g} | {r['temperature_sd_rms_C']:.6g} | {r['temperature_mean_field_rmse_C']:.6g} |")
    for n in cfg['measurement_counts']:
        text += ['',f'## {n} observations','',f'![All eight permeability inputs and errors](figures/n-{n:04d}-inputs-shared.png)','',
                 f'![All eight temperature outputs and errors](figures/n-{n:04d}-outputs-shared.png)']
    text+=['','Input maps cover the full domain; temperature maps cover the centre crop outlined in the input figures. '
        'Input reference error is log10(sample/hidden), so 0.301 means twice the hidden permeability and -0.301 means half. '
        'Temperature errors are in C. Spread around the eight-draw mean and error against the hidden reference are separate quantities. '
        'Detail scaling is explicitly per count and must not be used to compare error amplitudes across counts. '
        'The hidden temperature is an LGCNN prediction, not a physical-simulator reference. This single-truth study is illustrative rather than a calibration assessment.']
    (output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    products=[p for p in (output/'figures').rglob('*') if p.is_file()]
    products += [output/'report.html',output/'summary.json',output/'report.md']
    products += [p for n in cfg['measurement_counts'] for p in (output/f'n-{n:04d}').glob('*.npy')
                 if p.name in {'ensemble_log10_mean.npy','ensemble_log10_std.npy','temperature_mean.npy','temperature_std.npy'}]
    manifest.update(status='complete',report_sha256={str(p.relative_to(output)):sha256_file(p) for p in products})
    _save(output/'manifest.json',manifest)
