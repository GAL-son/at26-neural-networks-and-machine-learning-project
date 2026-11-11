"""Compute device. AMD GPUs work through the ROCm build of PyTorch (it shows up as "cuda")
or through torch-directml (any DX12 GPU on Windows; fewer ops, no float64)."""

from __future__ import annotations
import torch

def select_device(name: str = "auto") -> torch.device:
    """auto = ROCm/CUDA, then DirectML, then CPU. "rocm" is an alias of "cuda"."""
    if name in ("auto", "cuda", "rocm") and torch.cuda.is_available():
        return torch.device("cuda")
    if name in ("auto", "directml"):
        try:
            import torch_directml
            return torch_directml.device()
        except ImportError:
            pass
    if name in ("auto", "cpu"):
        return torch.device("cpu")
    raise RuntimeError(f"Device '{name}' is not available")

def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
        return f"{torch.cuda.get_device_name(device)} ({backend})"
    return "DirectML" if device.type == "privateuseone" else "CPU"




