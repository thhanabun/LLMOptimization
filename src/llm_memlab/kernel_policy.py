from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .report import make_table

STABLE_BACKENDS = {"auto", "torch", "triton"}
ALL_BACKENDS = STABLE_BACKENDS | {"triton-experimental", "cutile-experimental", "cutile"}


@dataclass(frozen=True)
class KernelPolicy:
    backend: str = "auto"
    quant_dtype: str = "int8"
    prefer_fused_decode: bool = True
    allow_paged_attention: bool = True
    allow_experimental: bool = False
    max_head_dim: int = 256
    max_decode_tokens: int = 4096

    def validate(self) -> None:
        if self.backend not in ALL_BACKENDS:
            raise ValueError("backend must be auto, torch, triton, triton-experimental, cutile, or cutile-experimental")
        if self.quant_dtype not in {"int8", "uint8", "fp16", "bf16", "fp32", "fp8_e4m3fn"}:
            raise ValueError("unsupported quant_dtype")
        if self.max_head_dim <= 0 or self.max_decode_tokens <= 0:
            raise ValueError("max_head_dim and max_decode_tokens must be positive")


@dataclass(frozen=True)
class KernelSelection:
    backend: str
    attention_impl: str
    quant_dtype: str
    use_fused_decode: bool
    use_paged_attention: bool
    stable: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def experimental(self) -> bool:
        return not self.stable

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Backend", self.backend),
                ("Attention", self.attention_impl),
                ("Quant dtype", self.quant_dtype),
                ("Fused decode", self.use_fused_decode),
                ("Paged attention", self.use_paged_attention),
                ("Stable", self.stable),
                ("Reasons", "; ".join(self.reasons)),
            ],
        )


def default_kernel_policy() -> KernelPolicy:
    return KernelPolicy()


def select_kernel_policy(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    sequence_length: int,
    dtype: str = "fp16",
    device: Any = None,
    paged: bool = False,
    policy: KernelPolicy | None = None,
) -> KernelSelection:
    cfg = policy or default_kernel_policy()
    cfg.validate()
    reasons: list[str] = []
    quant_dtype = cfg.quant_dtype
    cuda_available = _cuda_available(device)
    triton_available = _triton_available()
    cutile_info = _cutile_info(device)

    if cfg.backend == "torch":
        reasons.append("torch backend forced by policy")
        return KernelSelection("torch", "dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))

    if not cuda_available:
        reasons.append("CUDA is not available for the selected device")
        return KernelSelection("torch", "dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))
    if cfg.backend in {"cutile", "cutile-experimental"}:
        if cutile_info is None:
            reasons.append("CuTile backend requested but CuTile detection failed; falling back")
        else:
            reasons.extend(cutile_info.reasons)
            if cutile_info.production_candidate and (cfg.allow_experimental or cfg.backend == "cutile-experimental"):
                return KernelSelection(
                    "cutile-experimental", "paged-fused-decode", quant_dtype, cfg.prefer_fused_decode, bool(paged), False, tuple(reasons)
                )
            reasons.append("CuTile is not production-selected on this GPU/runtime; falling back to Triton/Torch")
    if not triton_available:
        if cfg.backend in {"triton", "triton-experimental"}:
            reasons.append("Triton backend requested but Triton is unavailable; falling back to torch")
        else:
            reasons.append("Triton is unavailable")
        return KernelSelection("torch", "dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))

    reasons.append("CUDA and Triton are available")
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        reasons.append("q_heads must be divisible by kv_heads for GQA/MQA fused decode")
        return KernelSelection("triton", "quant-dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))
    if head_dim > cfg.max_head_dim:
        reasons.append(f"head_dim {head_dim} exceeds fused limit {cfg.max_head_dim}")
        return KernelSelection("triton", "quant-dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))
    if quant_dtype not in {"int8", "uint8"}:
        reasons.append("fused decode currently supports int8/uint8 quantized K/V")
        return KernelSelection("triton", "quant-dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))

    wants_experimental = cfg.backend == "triton-experimental" or cfg.allow_experimental
    use_paged = bool(paged and cfg.allow_paged_attention)
    if use_paged:
        if not wants_experimental:
            reasons.append(
                "paged fused decode is experimental; falling back unless allow_experimental=True or backend='triton-experimental'"
            )
            return KernelSelection("triton", "quant-dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))
        if sequence_length > cfg.max_decode_tokens:
            reasons.append("paged experimental attention selected with streaming softmax for long context")
        else:
            reasons.append("paged experimental fused decode selected")
        return KernelSelection(
            "triton-experimental", "paged-fused-decode", quant_dtype, cfg.prefer_fused_decode, True, False, tuple(reasons)
        )

    if sequence_length > cfg.max_decode_tokens:
        reasons.append("dense fused decode limit exceeded; use experimental paged attention explicitly for streaming long context")
        return KernelSelection("triton", "quant-dequant+sdpa", quant_dtype, False, False, True, tuple(reasons))

    reasons.append("dense fused decode selected")
    return KernelSelection("triton", "fused-decode", quant_dtype, cfg.prefer_fused_decode, False, True, tuple(reasons))


def _cuda_available(device: Any) -> bool:
    try:
        import torch

        if device is not None and str(device).startswith("cpu"):
            return False
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _triton_available() -> bool:
    try:
        from .triton_kernels import triton_available

        return triton_available()
    except Exception:
        return False


def _cutile_info(device: Any):
    try:
        from .backends.cutile import detect_cutile_runtime

        return detect_cutile_runtime(device)
    except Exception:
        return None
