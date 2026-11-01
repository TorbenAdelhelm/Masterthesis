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
    tile=min(96,*cfg["shape"]);cases=[]
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
                fields=np.load(directory/"permeability_samples.npy",mmap_mode="r")
                temps=np.load(directory/"temperature_ensemble.npy",mmap_mode="r")
                y=np.stack([np.log10(_thumbnail(f,tile).astype(float)) for f in fields]).astype(np.float32)
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
                    "n_samples":len(fields),"images":{}})
                del arrays,y,t,fields,temps
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
        (root/f"case-{j:04d}.npz").unlink()
    payload={"cases":cases,"roles":roles,"labels":labels,"limits":limits,"tile":tile,"columns":10,
        "interpretation":"point-subsampled spatial pictures; common scales across all kernels/counts/draws; deviations and truth errors are distinct; full native arrays retained"}
    template=Path(__file__).with_name("ensemble_browser.html").read_text(encoding="utf-8")
    (output/"ensemble_browser.html").write_text(template.replace("__STUDY_DATA__",json.dumps(payload)),encoding="utf-8")
    return ["ensemble_browser.html",*[str(p.relative_to(output)) for p in sorted(root.glob("*.png"))]]
