from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from subsurface_uq.surrogates import frozen_temperature as frozen


def _assets(tmp_path):
    for role in ("cnn1","cnn3"):
        directory=tmp_path/role
        directory.mkdir()
        for name in ("model.pt","info.yaml","HPS_options.yaml"):
            (directory/name).write_bytes(b"fixed checkpoint asset")
    (tmp_path/"upstream").mkdir()
    (tmp_path/"upstream/model.py").write_text("# upstream code\n")
    (tmp_path/"prepared/Inputs").mkdir(parents=True)
    (tmp_path/"prepared/Inputs/RUN_2.pt").write_bytes(b"fixed pressure/source/material")
    (tmp_path/"prepared/info.yaml").write_text(yaml.safe_dump({"CellsNumber":[8,8],"CellsSize":[5.,5.]}))
    config={"release25":{"repo":"upstream","cnn1_dir":"cnn1","cnn2_dir":"cnn3",
        "prepared_pki_dir":"prepared","fixed_run_id":"RUN_2","cell_size_m":5.,
        "background_temperature":10.6,"random_k":False},
        "qoi":{"receptors":[[1,1]],"mean_anomaly_roi":[0,3,0,3]},
        "streamlines":{"mode":"bounded","method":"RK45"},
        "grf":{"obsolete":"this study supplies its own declared law"}}
    source=tmp_path/"model.yaml"
    source.write_text(yaml.safe_dump(config))
    return source,config


def test_frozen_loader_uses_actual_declared_assets_and_native_grid(tmp_path,monkeypatch):
    source,_=_assets(tmp_path)
    settings={}
    bundle=SimpleNamespace(info={"Inputs":{"Permeability":{"min":1e-12,"max":1e-8}}})
    class Adapter:
        cnn1=cnn2=bundle
        def predict(self,permeability,fixed):
            assert fixed=="fixed context"
            return {"temperature":torch.full((1,4,4),11.),"velocity":torch.zeros(2,6,6)}
    def runtime(**kwargs):
        settings.update(kwargs)
        return SimpleNamespace(adapter=Adapter(),scenario=SimpleNamespace(shape=(8,8),fixed="fixed context"))
    monkeypatch.setattr(frozen.Release25Runtime,"from_paths",runtime)
    monkeypatch.setattr(frozen,"configure_release25_streamlines",lambda *args,**kwargs:None)
    model=frozen.load_frozen_temperature_model(source,torch_threads=1)
    assert settings["random_k"] is False
    assert settings["run_id"]=="RUN_2"
    assert settings["prepared_pki_dir"]==tmp_path/"prepared"
    assert len(model.metadata["files_sha256"])==10
    field=np.full((8,8),1e-10)
    field[0,0]=1e-7
    assert model.predict_temperature(field).shape==(4,4)
    diagnostic=model.normalization_diagnostics(field)
    assert diagnostic["cnn1"]["fraction_above"]==1/64
    assert diagnostic["cnn3"]["fraction_above"]==0
    with pytest.raises(ValueError,match="grid"):
        model.predict_temperature(field[:7])


@pytest.mark.parametrize("section,key,value,reason",[
    ("release25","random_k",True,"real-K"),
    ("release25","cell_size_m",20.,"native 5"),
    ("release25","fixed_run_id","../RUN_2","RUN name"),
    ("qoi","receptors",[[True,1]],"integer"),
    ("qoi","receptors",[[1,1],[1,1]],"distinct"),
    ("qoi","mean_anomaly_roi",[0,0,1,3],"positive extent"),
    ("streamlines","mode","changed","unsupported"),
])
def test_frozen_loader_rejects_inconsistent_context_before_model_load(tmp_path,monkeypatch,section,key,value,reason):
    source,config=_assets(tmp_path)
    config[section][key]=value
    source.write_text(yaml.safe_dump(config))
    def forbidden(**kwargs):
        pytest.fail("model loading should not happen for an invalid context")
    monkeypatch.setattr(frozen.Release25Runtime,"from_paths",forbidden)
    with pytest.raises(ValueError,match=reason):
        frozen.load_frozen_temperature_model(source)
