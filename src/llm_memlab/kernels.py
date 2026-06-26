from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


@dataclass(frozen=True)
class KernelConfig:
    """Runtime options for PyTorch-compatible optimized kernels."""

    compile: bool = False
    compile_backend: str | None = None
    fullgraph: bool = False


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Optimized kernels require PyTorch. Install with: pip install torch") from exc
    return torch


def maybe_compile(fn: Callable[..., Any], config: KernelConfig | None = None) -> Callable[..., Any]:
    """Compile a kernel with torch.compile when requested and available."""

    cfg = config or KernelConfig()
    if not cfg.compile:
        return fn
    torch = require_torch()
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        return fn
    return compiler(fn, backend=cfg.compile_backend, fullgraph=cfg.fullgraph)


@lru_cache(maxsize=32)
def _compiled(name: str, backend: str | None, fullgraph: bool):
    registry = {
        "rms_norm": rms_norm,
        "apply_rope": apply_rope,
        "swiglu": swiglu,
        "scaled_dot_product_attention": scaled_dot_product_attention,
        "chunked_cross_entropy": chunked_cross_entropy,
    }
    return maybe_compile(registry[name], KernelConfig(True, backend, fullgraph))


def kernel(name: str, config: KernelConfig | None = None) -> Callable[..., Any]:
    """Return a kernel function, optionally compiled with torch.compile."""

    cfg = config or KernelConfig()
    if not cfg.compile:
        return {
            "rms_norm": rms_norm,
            "apply_rope": apply_rope,
            "swiglu": swiglu,
            "scaled_dot_product_attention": scaled_dot_product_attention,
            "chunked_cross_entropy": chunked_cross_entropy,
        }[name]
    return _compiled(name, cfg.compile_backend, cfg.fullgraph)


def rms_norm(x, weight, eps: float = 1e-6, bias=None):
    """RMSNorm with fp32 variance and output dtype preserved from x."""

    torch = require_torch()
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    y = y * weight.to(dtype=x.dtype)
    if bias is not None:
        y = y + bias.to(dtype=x.dtype)
    return y


def rotate_half_interleaved(x):
    """Rotate [..., x0, x1, x2, x3] into [..., -x1, x0, -x3, x2]."""

    torch = require_torch()
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to q and k.

    q/k are usually shaped [batch, heads, seq, head_dim]. cos/sin may be
    [seq, head_dim], [seq, head_dim / 2], or already broadcastable.
    """

    cos = _expand_rope_cache(cos, q)
    sin = _expand_rope_cache(sin, q)
    return (
        (q * cos) + (rotate_half_interleaved(q) * sin),
        (k * cos) + (rotate_half_interleaved(k) * sin),
    )


def swiglu(x, gate_weight, up_weight, down_weight, gate_bias=None, up_bias=None, down_bias=None):
    """SwiGLU MLP using PyTorch linear kernels and one explicit activation product."""

    torch = require_torch()
    functional = torch.nn.functional
    gate = functional.linear(x, gate_weight, gate_bias)
    up = functional.linear(x, up_weight, up_bias)
    hidden = functional.silu(gate) * up
    return functional.linear(hidden, down_weight, down_bias)


def scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False, scale=None):
    """Dispatch to PyTorch SDPA, which can use FlashAttention-style kernels on supported GPUs."""

    torch = require_torch()
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def chunked_cross_entropy(logits, targets, chunk_size: int = 1024, ignore_index: int = -100, reduction: str = "mean"):
    """Cross entropy over vocab chunks to avoid one huge loss allocation.

    This keeps logits intact but chunks the flattened token dimension, which is
    useful for long-context SFT where loss temporaries can spike near the end of
    the forward pass.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")

    torch = require_torch()
    functional = torch.nn.functional
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    if reduction == "none":
        pieces = []
        for start in range(0, flat_logits.shape[0], chunk_size):
            end = min(start + chunk_size, flat_logits.shape[0])
            pieces.append(
                functional.cross_entropy(
                    flat_logits[start:end],
                    flat_targets[start:end],
                    ignore_index=ignore_index,
                    reduction="none",
                )
            )
        return torch.cat(pieces, dim=0).reshape_as(targets)

    total = flat_logits.new_zeros(())
    valid = flat_logits.new_zeros(())
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        target_chunk = flat_targets[start:end]
        total = total + functional.cross_entropy(
            flat_logits[start:end],
            target_chunk,
            ignore_index=ignore_index,
            reduction="sum",
        )
        valid = valid + (target_chunk != ignore_index).sum().to(dtype=flat_logits.dtype)

    if reduction == "sum":
        return total
    return total / valid.clamp_min(1)


def _expand_rope_cache(cache, target):
    torch = require_torch()
    if cache.shape[-1] * 2 == target.shape[-1]:
        cache = cache.repeat_interleave(2, dim=-1)
    while cache.dim() < target.dim():
        cache = cache.unsqueeze(0)
    return cache.to(device=target.device, dtype=target.dtype)
