"""Offline spatial browser of ALL retained draws; native arrays remain authoritative."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _thumbnail(array,tile):
    rows=np.linspace(0,array.shape[-2]-1,tile).astype(int)
    cols=np.linspace(0,array.shape[-1]-1,tile).astype(int)
    return np.asarray(array[np.ix_(rows,cols)],dtype=np.float32)


def write_browser(cfg,output):
    from ..experiments.controlled_field_benchmark import _cases
    output=Path(output);root=output/"browser";root.mkdir(exist_ok=True)
    compact=cfg.get('compact_storage')
    tile=min(compact['preview_size'] if compact else 96,*cfg["shape"]);cases=[]
    roles=["input","output","input_error","output_error","input_deviation","output_deviation"]
    labels=["log10 permeability [m²]","LGCNN temperature [°C]","log10 k − hidden truth",
            "Temperature − hidden truth [°C]","log10 k − analytic conditional mean",
            "Temperature − ensemble mean [°C]"]
    limits={r:[np.inf,-np.inf] for r in roles}
    # Cache only point-subsampled pictures, never entire native ensembles.
    for name in cfg["scenarios"]:
        for trial in range(cfg["n_truths"]):
            truth_root=output/name/f"truth-{trial:04d}"
            ystar=_thumbnail(np.load(truth_root/"truth_log10.npy",mmap_mode="r"),tile)
            tstar=_thumbnail(np.load(truth_root/"truth_temperature.npy",mmap_mode="r"),tile)
            for design,n,directory in _cases(cfg,name,trial,output):
                temps=np.load(directory/"temperature_ensemble.npy",mmap_mode="r")
                if compact:
                    y=np.array(np.load(directory/'log_permeability_previews.npy',mmap_mode='r'),copy=True)
                    records=json.loads((directory/'prediction_records.json').read_text())
                    for role,key in [('input','input_log10_range'),('output','temperature_range_C')]:
                        limits[role]=[min(limits[role][0],min(r[key][0] for r in records)),
                                      max(limits[role][1],max(r[key][1] for r in records))]
                else:
                    fields=np.load(directory/"permeability_samples.npy",mmap_mode="r")
                    y=np.stack([np.log10(_thumbnail(f,tile).astype(float)) for f in fields]).astype(np.float32)
                    del fields
                t=np.stack([_thumbnail(f,tile) for f in temps])
                ym=_thumbnail(np.load(directory/"posterior_log10_mean.npy",mmap_mode="r"),tile)
                tm=_thumbnail(np.load(directory/"temperature_mean.npy",mmap_mode="r"),tile)
                arrays=[y,t,y-ystar,t-tstar,y-ym,t-tm]
                for role,a in zip(roles,arrays):
                    limits[role]=[min(limits[role][0],float(a.min())),max(limits[role][1],float(a.max()))]
                scratch=root/f"case-{len(cases):04d}.npz"
                np.savez(scratch,**dict(zip(roles,arrays)))
                cases.append({"label":f"{name} | truth {trial} | {design} | {n} observations",
                    "native_arrays":directory.relative_to(root.parent).as_posix(),"scratch":scratch,
                    "n_samples":len(temps),"images":{},"native_views":{}})
                del arrays,y,t,temps
    for role in roles:
        if "error" in role or "deviation" in role:
            bound=max(abs(v) for v in limits[role]);limits[role]=[-max(bound,1.e-10),max(bound,1.e-10)]
        elif limits[role][1]<=limits[role][0]:
            limits[role][1]=limits[role][0]+1.e-8
    for j,case in enumerate(cases):
        with np.load(case.pop("scratch")) as data:
            for role in roles:
                values=data[role];columns=10;rows=int(np.ceil(len(values)/columns))
                atlas=np.full((rows*tile,columns*tile),np.nan,dtype=np.float32)
                for i,a in enumerate(values):
                    r,c=divmod(i,columns);atlas[r*tile:(r+1)*tile,c*tile:(c+1)*tile]=a[::-1]
                filename=f"case-{j:04d}-{role}.png";case["images"][role]=filename
                plt.imsave(root/filename,atlas,cmap="coolwarm" if "error" in role or "deviation" in role else "viridis",
                           vmin=limits[role][0],vmax=limits[role][1])
            if compact:
                directory=output/case['native_arrays']
                selected=np.load(directory/'permeability_selected.npy',mmap_mode='r')
                temperatures=np.load(directory/'temperature_ensemble.npy',mmap_mode='r')
                for i,field in enumerate(selected):
                    case['native_views'][str(i)]={}
                    for role,a in [('input',np.log10(field.astype(float))),('output',temperatures[i])]:
                        filename=f'native-{j:04d}-{role}-{i:04d}.png'
                        plt.imsave(root/filename,a[::-1],cmap='viridis',vmin=limits[role][0],vmax=limits[role][1])
                        case['native_views'][str(i)][role]=filename
                del field,selected,temperatures
        (root/f"case-{j:04d}.npz").unlink()
    retention=(f"All temperature outputs and the first {compact['retained_input_samples']} full-resolution inputs per case are retained. Native-view links are available for these fixed sample indices. Other inputs can be regenerated and verified using their stored recipes."
               if compact else 'All native input and temperature arrays are retained.')
    payload={"cases":cases,"roles":roles,"labels":labels,"limits":limits,"tile":tile,"columns":10,
        "retention_description":retention,"native_input_shape":cfg['shape'],
        "native_temperature_shape":compact['temperature_shape'] if compact else None,"cell_size_m":cfg['cell_size_m'],
        "interpretation":"point-subsampled spatial previews with common scales; native selected-sample links preserve original grid dimensions; deviations and truth errors are distinct"}
    template=Path(__file__).with_name("ensemble_browser.html").read_text(encoding="utf-8")
    (output/"ensemble_browser.html").write_text(template.replace("__STUDY_DATA__",json.dumps(payload)),encoding="utf-8")
    return ["ensemble_browser.html",*[str(p.relative_to(output)) for p in sorted(root.glob("*.png"))]]
