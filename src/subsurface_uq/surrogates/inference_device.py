"""Resolve an inference device before loading the frozen LGCNN checkpoints.

Auto selection checks a real float32 convolution on the first visible CUDA
device. It never substitutes CPU inference after a model evaluation fails.
The returned record is stable hardware/runtime provenance, without free-memory
or timing values that would invalidate a resume for unrelated activity.
"""
from __future__ import annotations

import argparse
import json
import re

import torch


def _probe_cuda(index: int) -> dict:
    """Check initialization, allocation and a CNN operation, without RNG draws."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this PyTorch installation")
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is not visible")
    selected = f"cuda:{index}"
    properties = torch.cuda.get_device_properties(index)
    with torch.no_grad():
        inputs = torch.ones((1, 1, 16, 16), dtype=torch.float32, device=selected)
        weights = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=selected)
        result = torch.nn.functional.conv2d(inputs, weights).sum().item()
    # .item() waits for the device, catching asynchronous launch errors too.
    if result != 1764.0:
        raise RuntimeError("CUDA float32 convolution check returned an invalid result")
    uuid = getattr(properties, "uuid", None)
    return {"index": index, "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "uuid": None if uuid is None else str(uuid)}


def select_inference_device(requested: str = "auto") -> dict:
    """Return the actual device and its selection reason; explicit CUDA is strict.

    ``cpu`` avoids CUDA initialization. ``auto`` selects the first visible GPU
    only if the convolution probe succeeds, otherwise CPU. ``cuda[:N]`` must
    pass the probe. CUDA_VISIBLE_DEVICES retains its normal PyTorch semantics.
    """
    if not isinstance(requested, str) or not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", requested):
        raise ValueError("inference device must be auto, cpu, cuda or cuda:N")
    record = {"requested": requested, "selected": "cpu", "gpu": None,
              "torch_version": str(torch.__version__),
              "cuda_build_version": torch.version.cuda,
              "hip_build_version": torch.version.hip,
              "cuda_probe": "not_requested", "reason": "CPU explicitly selected"}
    if requested == "cpu":
        return record
    index = 0 if requested in {"auto", "cuda"} else int(requested.split(":")[1])
    try:
        gpu = _probe_cuda(index)
    except (RuntimeError, AssertionError, OSError) as exc:
        if requested != "auto":
            raise RuntimeError(f"Requested {requested} is not usable: {exc}") from exc
        record.update(cuda_probe="unavailable_or_failed",
                      reason=f"CPU fallback: {type(exc).__name__}: {exc}")
        return record
    record.update(selected=f"cuda:{index}", gpu=gpu, cuda_probe="float32_convolution_passed",
                  reason="First visible CUDA device passed the convolution check" if requested == "auto"
                         else "Explicit CUDA device passed the convolution check")
    return record


def describe_inference_device(record: dict) -> str:
    message = f'Inference device: {record["selected"]} (requested: {record["requested"]})'
    if record["gpu"] is not None:
        gpu = record["gpu"]
        message += f'; {gpu["name"]}, {gpu["total_memory_bytes"] / 2**30:.1f} GiB total GPU memory'
    return message + '\n' + record['reason']


def configure_inference_numerics(selected: str) -> dict:
    """Keep CUDA convolutions in FP32 and request repeatable cuDNN algorithms."""
    record = {"dtype": "float32", "autocast": False}
    if not selected.startswith("cuda:"):
        return record
    # PyTorch >=2.9 has separate precision controls; never mix these with the
    # legacy allow_tf32 setters used by the supported older PyTorch releases.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        record["precision_api"] = "fp32_precision"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        record["precision_api"] = "allow_tf32"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    record.update(tf32=False, cudnn_benchmark=False, cudnn_deterministic=True)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check PyTorch CUDA inference support without model/data assets")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda or cuda:N; explicit CUDA must work")
    parser.add_argument("--json", action="store_true", help="Print the device selection and hardware/runtime record as JSON")
    args = parser.parse_args(argv)
    try:
        record = select_inference_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.exit(1, f"CUDA check failed: {exc}\n")
    print(json.dumps(record, indent=2) if args.json else describe_inference_device(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
