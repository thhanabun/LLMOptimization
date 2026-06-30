from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Any

from .backends.cutile import detect_cutile_runtime
from .triton_kernels import triton_available

HARDWARE_PROFILE_SCHEMA_VERSION = "llm_memlab.hardware_profile.v1"


@dataclass(frozen=True)
class HardwareProfile:
    device: str = "cpu"
    gpu_name: str | None = None
    cuda_available: bool = False
    cuda_capability: tuple[int, int] | None = None
    cuda_capability_str: str | None = None
    total_vram_bytes: int | None = None
    sm_count: int | None = None
    tensor_cores: bool = False
    bf16_supported: bool = False
    fp8_supported: bool = False
    triton_supported: bool = False
    cutile_supported: bool = False
    cutile_architecture: str = "unknown"
    os: str = field(default_factory=platform.system)
    quirks: tuple[str, ...] = ()
    schema_version: str = HARDWARE_PROFILE_SCHEMA_VERSION

    @property
    def architecture(self) -> str:
        if self.cuda_capability is None:
            return "cpu"
        major, minor = self.cuda_capability
        if major >= 10:
            return "blackwell"
        if major >= 9:
            return "hopper"
        if major == 8 and minor >= 9:
            return "ada"
        if major == 8:
            return "ampere"
        if major == 7:
            return "volta-turing"
        return "pre-ampere"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device": self.device,
            "gpu_name": self.gpu_name,
            "cuda_available": self.cuda_available,
            "cuda_capability": self.cuda_capability_str,
            "architecture": self.architecture,
            "total_vram_bytes": self.total_vram_bytes,
            "sm_count": self.sm_count,
            "tensor_cores": self.tensor_cores,
            "bf16_supported": self.bf16_supported,
            "fp8_supported": self.fp8_supported,
            "triton_supported": self.triton_supported,
            "cutile_supported": self.cutile_supported,
            "cutile_architecture": self.cutile_architecture,
            "os": self.os,
            "quirks": list(self.quirks),
        }


def detect_hardware_profile(device: Any = None) -> HardwareProfile:
    quirks: list[str] = []
    torch = _try_import_torch()
    cutile = detect_cutile_runtime(device)
    if torch is None or not torch.cuda.is_available():
        return HardwareProfile(
            device="cpu",
            cuda_available=False,
            triton_supported=triton_available(),
            cutile_supported=cutile.available,
            cutile_architecture=cutile.architecture,
            quirks=("CUDA is not available",),
        )

    index = _device_index(torch, device)
    props = torch.cuda.get_device_properties(index)
    raw_capability = torch.cuda.get_device_capability(index)
    capability = (int(raw_capability[0]), int(raw_capability[1]))
    major, minor = capability
    if platform.system().lower().startswith("win"):
        quirks.append("windows-cuda-runtime")
    return HardwareProfile(
        device=f"cuda:{index}",
        gpu_name=torch.cuda.get_device_name(index),
        cuda_available=True,
        cuda_capability=capability,
        cuda_capability_str=f"{major}.{minor}",
        total_vram_bytes=int(getattr(props, "total_memory", 0)),
        sm_count=int(getattr(props, "multi_processor_count", 0)),
        tensor_cores=major >= 7,
        bf16_supported=bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
        fp8_supported=major >= 9,
        triton_supported=triton_available(),
        cutile_supported=cutile.available,
        cutile_architecture=cutile.architecture,
        quirks=tuple(quirks),
    )


def _device_index(torch: Any, device: Any) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    try:
        parsed = torch.device(device)
        return int(parsed.index or 0)
    except Exception:
        return 0


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None
