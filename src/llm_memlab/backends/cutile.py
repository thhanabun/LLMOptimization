from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

from ..quality_metrics import LogitQualityResult, compare_logits
from ..report import make_table

_CUTILE_MODULE_CANDIDATES = ("cutile", "cuda_tile", "nvidia.cutile")


@dataclass(frozen=True)
class CuTileRuntimeInfo:
    available: bool
    runtime_module: str | None = None
    runtime_version: str | None = None
    gpu_name: str | None = None
    capability: tuple[int, int] | None = None
    architecture: str = "unknown"
    production_candidate: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Available", self.available),
                ("Runtime", self.runtime_module or "n/a"),
                ("Runtime version", self.runtime_version or "n/a"),
                ("GPU", self.gpu_name or "n/a"),
                ("Capability", self.capability or "n/a"),
                ("Architecture", self.architecture),
                ("Production candidate", self.production_candidate),
                ("Reasons", "; ".join(self.reasons)),
            ],
        )


@dataclass(frozen=True)
class CuTileDispatchResult:
    output: Any
    backend_used: str
    info: CuTileRuntimeInfo
    fallback_reason: str | None = None


@dataclass(frozen=True)
class CuTileCertificationResult:
    info: CuTileRuntimeInfo
    quality: LogitQualityResult
    backend_used: str
    passed: bool
    fallback_reason: str | None = None

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Backend used", self.backend_used),
                ("Passed", self.passed),
                ("Fallback reason", self.fallback_reason or "n/a"),
                ("Mean abs", f"{self.quality.mean_abs_error:.6f}"),
                ("Top-1", f"{self.quality.top1_agreement:.1%}"),
                ("Runtime available", self.info.available),
                ("Architecture", self.info.architecture),
            ],
        )


def detect_cutile_runtime(device: Any = None) -> CuTileRuntimeInfo:
    reasons: list[str] = []
    runtime_module = _find_cutile_module()
    runtime_version = None
    if runtime_module is None:
        reasons.append("CuTile runtime module was not found")
    else:
        runtime_version = _module_version(runtime_module)
        reasons.append(f"CuTile runtime import candidate found: {runtime_module}")
    gpu_name, capability, architecture = _detect_gpu(device)
    if capability is None:
        reasons.append("CUDA GPU capability is unavailable")
    else:
        reasons.append(f"GPU architecture detected as {architecture}")
    production_candidate = bool(runtime_module and architecture in {"hopper", "blackwell"})
    if not production_candidate:
        reasons.append("CuTile is kept experimental unless runtime exists on Hopper/Blackwell-class GPUs")
    return CuTileRuntimeInfo(
        available=runtime_module is not None,
        runtime_module=runtime_module,
        runtime_version=runtime_version,
        gpu_name=gpu_name,
        capability=capability,
        architecture=architecture,
        production_candidate=production_candidate,
        reasons=tuple(reasons),
    )


def cutile_fused_decode_attention(
    q,
    k_pages,
    v_pages,
    page_table,
    lengths,
    *,
    page_size: int,
    scale: float | None = None,
    require_runtime: bool = False,
) -> CuTileDispatchResult:
    """Single-token paged fp16/bf16 decode attention with an honest CuTile fallback.

    The public contract is intentionally production-safe: if a future CuTile
    runtime exposes a compatible op we call it, otherwise we use the same
    memory shape and gather into PyTorch SDPA while reporting `backend_used`.
    """

    torch = _import_torch()
    info = detect_cutile_runtime(getattr(q, "device", None))
    if q.shape[0] != 1 or q.shape[-2] != 1:
        raise ValueError("CuTile fused decode v1 targets batch=1 and decode step=1")
    if q.shape[1] % k_pages.shape[1] != 0:
        raise ValueError("q_heads must be divisible by kv_heads for GQA/MQA")
    if q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError("CuTile fused decode v1 supports fp16/bf16/fp32 K/V paths; quantized K/V remains experimental elsewhere")
    op = _load_runtime_op(info)
    if op is not None:
        return CuTileDispatchResult(
            output=op(q, k_pages, v_pages, page_table, lengths, page_size=page_size, scale=scale),
            backend_used="cutile-experimental",
            info=info,
        )
    if require_runtime:
        reason = "CuTile runtime op is unavailable"
        raise RuntimeError(reason)
    output = _torch_paged_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=page_size, scale=scale)
    reason = "CuTile runtime op unavailable; used torch SDPA fallback"
    return CuTileDispatchResult(output=output, backend_used="torch-fallback", info=info, fallback_reason=reason)


def certify_cutile_decode_attention(
    q,
    k_pages,
    v_pages,
    page_table,
    lengths,
    *,
    page_size: int,
    max_mean_abs: float = 0.001,
    min_top1: float = 1.0,
) -> CuTileCertificationResult:
    reference = _torch_paged_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=page_size)
    dispatch = cutile_fused_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=page_size)
    quality = compare_logits(
        reference.float(), dispatch.output.float(), top_k=min(5, q.shape[-1]), max_mean_abs=max_mean_abs, min_top1=min_top1
    )
    passed = bool(quality.passed and dispatch.backend_used == "cutile-experimental" and dispatch.info.production_candidate)
    return CuTileCertificationResult(
        info=dispatch.info,
        quality=quality,
        backend_used=dispatch.backend_used,
        passed=passed,
        fallback_reason=dispatch.fallback_reason,
    )


def _torch_paged_decode_attention(q, k_pages, v_pages, page_table, lengths, *, page_size: int, scale: float | None = None):
    torch = _import_torch()
    batch, q_heads, _, head_dim = q.shape
    kv_heads = k_pages.shape[1]
    repeat = q_heads // kv_heads
    outputs = []
    for batch_idx in range(batch):
        length = int(lengths[batch_idx].item() if hasattr(lengths[batch_idx], "item") else lengths[batch_idx])
        pieces_k = []
        pieces_v = []
        remaining = length
        logical = 0
        while remaining > 0:
            take = min(page_size, remaining)
            physical = int(page_table[batch_idx, logical].item())
            pieces_k.append(k_pages[batch_idx : batch_idx + 1, :, physical, :take, :])
            pieces_v.append(v_pages[batch_idx : batch_idx + 1, :, physical, :take, :])
            remaining -= take
            logical += 1
        if pieces_k:
            k = torch.cat(pieces_k, dim=-2).repeat_interleave(repeat, dim=1)
            v = torch.cat(pieces_v, dim=-2).repeat_interleave(repeat, dim=1)
        else:
            k = torch.empty(1, q_heads, 0, head_dim, device=q.device, dtype=q.dtype)
            v = torch.empty_like(k)
        outputs.append(torch.nn.functional.scaled_dot_product_attention(q[batch_idx : batch_idx + 1], k, v, scale=scale))
    return torch.cat(outputs, dim=0)


def _find_cutile_module() -> str | None:
    for name in _CUTILE_MODULE_CANDIDATES:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return None


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name, fromlist=["__version__"])
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _load_runtime_op(info: CuTileRuntimeInfo):
    if info.runtime_module is None:
        return None
    try:
        module = __import__(info.runtime_module, fromlist=["fused_decode_attention"])
        return getattr(module, "fused_decode_attention", None)
    except Exception:
        return None


def _detect_gpu(device: Any = None) -> tuple[str | None, tuple[int, int] | None, str]:
    try:
        import torch

        if not torch.cuda.is_available() or (device is not None and str(device).startswith("cpu")):
            return None, None, "cpu"
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        if index is None:
            index = 0
        capability = torch.cuda.get_device_capability(index)
        return torch.cuda.get_device_name(index), capability, _architecture_name(capability)
    except Exception:
        return None, None, "unknown"


def _architecture_name(capability: tuple[int, int]) -> str:
    major, minor = capability
    del minor
    if major >= 10:
        return "blackwell"
    if major == 9:
        return "hopper"
    if major == 8:
        return "ampere"
    if major == 7:
        return "volta-turing"
    return "pre-ampere"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CuTile backend requires PyTorch") from exc
    return torch
