import json
from types import SimpleNamespace

import pytest
import torch

from subsurface_uq.surrogates import inference_device as device


GPU = {"index": 0, "name": "test GPU", "total_memory_bytes": 8 * 2**30,
       "compute_capability": [8, 0], "uuid": "test-uuid"}


def test_explicit_cpu_never_initializes_cuda(monkeypatch):
    monkeypatch.setattr(device, "_probe_cuda", lambda *a: pytest.fail("CPU must not probe CUDA"))
    record = device.select_inference_device("cpu")
    assert record["selected"] == "cpu" and record["gpu"] is None
    assert record["cuda_probe"] == "not_requested"


def test_auto_falls_back_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = device.select_inference_device()
    assert record["requested"] == "auto" and record["selected"] == "cpu"
    assert record["cuda_probe"] == "unavailable_or_failed"
    assert "not available" in record["reason"]


@pytest.mark.parametrize("error", [RuntimeError("kernel not supported"), OSError("driver failure"),
                                 AssertionError("Torch not compiled with CUDA")])
def test_auto_falls_back_after_a_failed_cuda_operation(monkeypatch, error):
    def broken(index):
        raise error
    monkeypatch.setattr(device, "_probe_cuda", broken)
    record = device.select_inference_device("auto")
    assert record["selected"] == "cpu" and str(error) in record["reason"]


@pytest.mark.parametrize("requested,index", [("auto", 0), ("cuda", 0), ("cuda:2", 2)])
def test_usable_cuda_is_selected_and_hardware_is_recorded(monkeypatch, requested, index):
    checked = []
    def usable(i):
        checked.append(i)
        return {**GPU, "index": i}
    monkeypatch.setattr(device, "_probe_cuda", usable)
    record = device.select_inference_device(requested)
    assert checked == [index]
    assert record["selected"] == f"cuda:{index}"
    assert record["gpu"]["name"] == "test GPU"
    assert record["cuda_probe"] == "float32_convolution_passed"
    assert "free_memory" not in json.dumps(record)
    assert "test GPU" in device.describe_inference_device(record)


def test_forced_cuda_fails_instead_of_selecting_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Requested cuda is not usable"):
        device.select_inference_device("cuda")


@pytest.mark.parametrize("requested", [None, 0, "", "CUDA", "cuda:-1", "cuda:abc", "cpu:2"])
def test_invalid_device_is_rejected(requested):
    with pytest.raises(ValueError, match="inference device must be"):
        device.select_inference_device(requested)


def test_cuda_probe_executes_float32_convolution_and_waits_for_result(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        name=GPU["name"], total_memory=GPU["total_memory_bytes"], major=8, minor=0, uuid="test-uuid"))
    def ones(shape, **kwargs):
        calls.append((shape, kwargs))
        return shape
    def convolution(inputs, weights):
        assert inputs == (1, 1, 16, 16) and weights == (1, 1, 3, 3)
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: 1764.0))
    monkeypatch.setattr(torch, "ones", ones)
    monkeypatch.setattr(torch.nn.functional, "conv2d", convolution)
    assert device._probe_cuda(0) == GPU
    assert all(kw == {"dtype": torch.float32, "device": "cuda:0"} for _, kw in calls)


def test_out_of_range_cuda_index_is_rejected_before_allocation(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="index 2 is not visible"):
        device.select_inference_device("cuda:2")


def test_cli_json_can_report_cpu_fallback_without_assets(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.main(["--json"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["selected"] == "cpu" and record["requested"] == "auto"


def test_cli_forced_cuda_exits_nonzero_when_unavailable(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as failure:
        device.main(["--device", "cuda"])
    assert failure.value.code == 1
    assert "CUDA check failed" in capsys.readouterr().err


@pytest.mark.parametrize('new_api', [False, True])
def test_cuda_numerics_use_fp32_without_mixing_precision_apis(monkeypatch, new_api):
    matmul=SimpleNamespace(allow_tf32=True)
    cudnn=SimpleNamespace(allow_tf32=True,benchmark=True,deterministic=False)
    if new_api:
        matmul.fp32_precision='tf32'
        cudnn.fp32_precision='tf32'
        cudnn.conv=SimpleNamespace(fp32_precision='tf32')
    fake=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul),cudnn=cudnn))
    monkeypatch.setattr(device,'torch',fake)
    record=device.configure_inference_numerics('cuda:0')
    assert record['tf32'] is False
    assert cudnn.benchmark is False and cudnn.deterministic is True
    if new_api:
        assert matmul.fp32_precision==cudnn.fp32_precision==cudnn.conv.fp32_precision=='ieee'
        assert matmul.allow_tf32 is True and cudnn.allow_tf32 is True
    else:
        assert matmul.allow_tf32 is False and cudnn.allow_tf32 is False


def test_cpu_numerics_do_not_touch_cuda_backend_settings(monkeypatch):
    monkeypatch.setattr(device,'torch',None)
    assert device.configure_inference_numerics('cpu')=={'dtype':'float32','autocast':False}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA hardware unavailable")
def test_real_cuda_convolution_when_available():
    assert device._probe_cuda(0)["total_memory_bytes"] > 0
