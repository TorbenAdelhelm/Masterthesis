from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import yaml

from subsurface_uq.surrogates.release25_runtime import (
    PreparedLGCNNScenario,
    Release25FixedFields,
    Release25ModelBundle,
    Release25Normalizer,
    Release25RuntimeAdapter,
    load_standard_release25_model,
)


def _metadata(names):
    return {
        name: {
            "index": index,
            "norm": None,
            "min": 0.0,
            "max": 1.0,
            "mean": 0.0,
            "std": 1.0,
        }
        for index, name in enumerate(names)
    }


def test_release25_normalizer_rescale_roundtrip():
    info = {
        "Inputs": {
            "x": {
                "index": 0,
                "norm": "Rescale",
                "min": 2.0,
                "max": 6.0,
                "mean": 4.0,
                "std": 1.0,
            }
        }
    }
    normalizer = Release25Normalizer(info)
    physical = torch.tensor([[[2.0, 4.0, 6.0]]])
    normalized = normalizer.transform(physical, "Inputs")
    torch.testing.assert_close(normalized, torch.tensor([[[0.0, 0.5, 1.0]]]))
    torch.testing.assert_close(normalizer.reverse(normalized, "Inputs"), physical)


def test_prepared_scenario_recovers_physical_pki(tmp_path):
    root = tmp_path / "prepared"
    (root / "Inputs").mkdir(parents=True)
    info = {
        "Inputs": {
            "Liquid Pressure [Pa]": {"index": 0, "norm": None},
            "Permeability X [m^2]": {"index": 1, "norm": None},
            "Material ID": {"index": 2, "norm": None},
        },
        "Labels": {},
    }
    (root / "info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
    pressure = torch.full((5, 7), 3.0)
    permeability = torch.full((5, 7), 2.0e-10)
    material = torch.ones((5, 7))
    material[2, 3] = 2.0
    torch.save(torch.stack([pressure, permeability, material]), root / "Inputs" / "RUN_0")

    scenario = PreparedLGCNNScenario.from_prepared_pki(root, "RUN_0")

    torch.testing.assert_close(scenario.original_permeability, permeability)
    torch.testing.assert_close(scenario.fixed.pressure, pressure)
    torch.testing.assert_close(scenario.fixed.material_id, material)
    assert scenario.shape == (5, 7)


class _Step1(nn.Module):
    def forward(self, x):
        # p,k,i -> two velocity channels. Keep the spatial shape for this unit test.
        return torch.stack([x[:, 1], x[:, 1]], dim=1)


class _Step3(nn.Module):
    def forward(self, x):
        # Six release25 channels; return the permeability channel as temperature.
        return x[:, 4:5]


def _bundle(role, model, info, input_code):
    return Release25ModelBundle(
        role=role,
        root=Path("."),
        checkpoint=Path("model.pt"),
        info=info,
        hparams={},
        input_code=input_code,
        model=model,
        normalizer=Release25Normalizer(info),
        device=torch.device("cpu"),
    )


def test_runtime_adapter_executes_all_three_steps(monkeypatch):
    parent = types.ModuleType("step2_streamlines")
    helpers = types.ModuleType("step2_streamlines.streamlines_helpers")

    def make_streamlines(*, mat_ids, vx, vy, dims, **kwargs):
        del mat_ids, vx, vy, kwargs
        return torch.zeros(dims, dtype=torch.float32)

    helpers.make_streamlines = make_streamlines
    monkeypatch.setitem(sys.modules, "step2_streamlines", parent)
    monkeypatch.setitem(sys.modules, "step2_streamlines.streamlines_helpers", helpers)
    monkeypatch.setattr(
        "subsurface_uq.surrogates.release25_runtime._install_release25_import_path",
        lambda _: Path("."),
    )

    info1 = {
        "Inputs": _metadata(["p", "k", "i"]),
        "Labels": _metadata(["vx", "vy"]),
    }
    info2 = {
        "Inputs": _metadata(["i", "vx", "vy", "s", "k", "so"]),
        "Labels": _metadata(["T"]),
    }
    adapter = Release25RuntimeAdapter(
        cnn1=_bundle("cnn1", _Step1(), info1, "pki"),
        cnn2=_bundle("cnn2", _Step3(), info2, "ik"),
        release25_repo="ignored",
    )
    k = torch.full((8, 10), 2.0e-10)
    material = torch.ones_like(k)
    material[3, 4] = 2.0
    fixed = Release25FixedFields(material_id=material, pressure=torch.ones_like(k))

    result = adapter.predict(k, fixed)

    assert result["velocity"].shape == (2, 8, 10)
    assert result["streamlines"].shape == (2, 8, 10)
    assert result["temperature"].shape == (1, 8, 10)
    np.testing.assert_allclose(result["temperature"].numpy()[0], k.numpy())


def test_standard_model_loader_uses_release25_hps_contract(tmp_path):
    repo = tmp_path / "release25"
    code = repo / "code"
    network_dir = code / "processing" / "networks"
    network_dir.mkdir(parents=True)
    (code / "processing" / "__init__.py").write_text("", encoding="utf-8")
    (network_dir / "__init__.py").write_text("", encoding="utf-8")
    (network_dir / "unetVariants.py").write_text(
        "import torch.nn as nn\n"
        "class UNet(nn.Module):\n"
        "    def __init__(self, in_channels, out_channels, **kwargs):\n"
        "        super().__init__(); self.conv=nn.Conv2d(in_channels,out_channels,1)\n"
        "    def forward(self,x): return self.conv(x)\n"
        "class UNetNoPad2(UNet):\n"
        "    pass\n",
        encoding="utf-8",
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    info = {
        "Inputs": _metadata(["p", "k", "i"]),
        "Labels": _metadata(["vx", "vy"]),
    }
    hps = {
        key: {"values": [value]}
        for key, value in {
            "inputs": "pki",
            "depth": 1,
            "init_features": 2,
            "kernel_size": 3,
            "stride": 1,
            "dilation": 1,
            "activation_fct": "ReLU",
            "norm": "batchnorm",
            "repeat_inner": False,
        }.items()
    }
    (model_dir / "info.yaml").write_text(yaml.safe_dump(info), encoding="utf-8")
    (model_dir / "HPS_options.yaml").write_text(yaml.safe_dump(hps), encoding="utf-8")

    # State dict matching the tiny fake upstream UNetNoPad2 implementation.
    state = {
        "conv.weight": torch.zeros((2, 3, 1, 1)),
        "conv.bias": torch.zeros(2),
    }
    torch.save(state, model_dir / "model.pt")

    # Ensure a previous release25 import from another test cannot mask this fixture.
    for name in list(sys.modules):
        if name == "processing" or name.startswith("processing."):
            sys.modules.pop(name)

    bundle = load_standard_release25_model(model_dir, repo, role="cnn1")

    assert bundle.input_code == "pki"
    assert bundle.in_channels == 3
    assert bundle.out_channels == 2
    assert bundle.model.training is False
