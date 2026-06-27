from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable



@dataclass(frozen=True)
class QuantizedAttentionDispatch:
    requested_backend: str
    selected_backend: str
    implementation: str
    quant_dtype: str
    reason: str = ""


def select_quantized_attention_backend(q, k, *, requested: str = "auto", quant_dtype: str = "int8") -> QuantizedAttentionDispatch:
    if requested not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of: auto, torch, triton")
    if requested == "torch":
        return QuantizedAttentionDispatch(requested, "torch", "dequant+sdpa", quant_dtype, "forced torch backend")
    if getattr(k, "is_cuda", False):
        try:
            from .triton_kernels import triton_available

            if triton_available():
                return QuantizedAttentionDispatch(requested, "triton", "triton-ready-fallback", quant_dtype, "fused kernel placeholder uses fallback today")
        except Exception as exc:
            if requested == "triton":
                raise
            return QuantizedAttentionDispatch(requested, "torch", "dequant+sdpa", quant_dtype, f"triton unavailable: {exc}")
    if requested == "triton":
        return QuantizedAttentionDispatch(requested, "torch", "dequant+sdpa", quant_dtype, "CUDA/Triton not available; using fallback")
    return QuantizedAttentionDispatch(requested, "torch", "dequant+sdpa", quant_dtype, "portable fallback")

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


@lru_cache(maxsize=64)
def _compiled(name: str, backend: str | None, fullgraph: bool):
    registry = _registry()
    return maybe_compile(registry[name], KernelConfig(True, backend, fullgraph))


def kernel(name: str, config: KernelConfig | None = None) -> Callable[..., Any]:
    """Return a kernel function, optionally compiled with torch.compile."""

    cfg = config or KernelConfig()
    if not cfg.compile:
        return _registry()[name]
    return _compiled(name, cfg.compile_backend, cfg.fullgraph)


def _registry() -> dict[str, Callable[..., Any]]:
    return {
        "rms_norm": rms_norm,
        "rms_norm_manual_backward": rms_norm_manual_backward,
        "triton_rms_norm": triton_rms_norm,
        "apply_rope": apply_rope,
        "triton_apply_rope": triton_apply_rope,
        "swiglu": swiglu,
        "triton_swiglu_activation": triton_swiglu_activation,
        "triton_quantize_int8_per_token": triton_quantize_int8_per_token,
        "triton_dequantize_int8_per_token": triton_dequantize_int8_per_token,
        "scaled_dot_product_attention": scaled_dot_product_attention,
        "quantized_kv_attention": quantized_kv_attention,
        "chunked_cross_entropy": chunked_cross_entropy,
        "linear_cross_entropy": linear_cross_entropy,
        "qkv_rope_attention": qkv_rope_attention,
        "qkv_rope_attention_cached": qkv_rope_attention_cached,
    }


def rms_norm(x, weight, eps: float = 1e-6, bias=None):
    """RMSNorm with fp32 variance and output dtype preserved from x."""

    torch = require_torch()
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(variance + eps).to(dtype=x.dtype)
    y = y * weight.to(dtype=x.dtype)
    if bias is not None:
        y = y + bias.to(dtype=x.dtype)
    return y


def triton_rms_norm(x, weight, eps: float = 1e-6, bias=None):
    from .triton_kernels import triton_rms_norm as _triton_rms_norm

    return _triton_rms_norm(x, weight, eps=eps, bias=bias)


def rms_norm_manual_backward(x, weight, eps: float = 1e-6, bias=None):
    """RMSNorm with a compact manual backward path."""

    torch = require_torch()

    class _RMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, eps):
            x_float = x.float()
            inv_rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + eps)
            y = x * inv_rms.to(dtype=x.dtype) * weight.to(dtype=x.dtype)
            if bias is not None:
                y = y + bias.to(dtype=x.dtype)
            ctx.save_for_backward(x, weight, inv_rms)
            ctx.has_bias = bias is not None
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, inv_rms = ctx.saved_tensors
            grad = grad_output.float()
            x_float = x.float()
            weight_float = weight.float()
            weighted_grad = grad * weight_float
            hidden = x.shape[-1]
            dot = (weighted_grad * x_float).sum(dim=-1, keepdim=True) / hidden
            dx = (weighted_grad * inv_rms) - (x_float * inv_rms.pow(3) * dot)
            reduce_dims = tuple(range(grad_output.dim() - 1))
            normalized = x_float * inv_rms
            dweight = (grad * normalized).sum(dim=reduce_dims).to(dtype=weight.dtype)
            dbias = grad.sum(dim=reduce_dims).to(dtype=weight.dtype) if ctx.has_bias else None
            return dx.to(dtype=x.dtype), dweight, dbias, None

    return _RMSNorm.apply(x, weight, bias, eps)


def rotate_half_interleaved(x):
    """Rotate [..., x0, x1, x2, x3] into [..., -x1, x0, -x3, x2]."""

    torch = require_torch()
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """Apply interleaved rotary position embeddings to q and k."""

    cos = _expand_rope_cache(cos, q)
    sin = _expand_rope_cache(sin, q)
    return (
        (q * cos) + (rotate_half_interleaved(q) * sin),
        (k * cos) + (rotate_half_interleaved(k) * sin),
    )


def triton_apply_rope(q, k, cos, sin):
    from .triton_kernels import triton_apply_rope as _triton_apply_rope

    return _triton_apply_rope(q, k, cos, sin)


def swiglu(x, gate_weight, up_weight, down_weight, gate_bias=None, up_bias=None, down_bias=None, *, use_triton: bool = False):
    """SwiGLU MLP using PyTorch linear kernels and optional Triton activation fusion."""

    torch = require_torch()
    functional = torch.nn.functional
    gate = functional.linear(x, gate_weight, gate_bias)
    up = functional.linear(x, up_weight, up_bias)
    hidden = triton_swiglu_activation(gate, up) if use_triton else functional.silu(gate) * up
    return functional.linear(hidden, down_weight, down_bias)


def triton_swiglu_activation(gate, up):
    from .triton_kernels import triton_swiglu_activation as _triton_swiglu_activation

    return _triton_swiglu_activation(gate, up)


def triton_quantize_int8_per_token(x, *, eps: float = 1e-6):
    from .triton_kernels import triton_quantize_int8_per_token as _triton_quantize_int8_per_token

    return _triton_quantize_int8_per_token(x, eps=eps)


def triton_dequantize_int8_per_token(q, scale, *, dtype=None):
    from .triton_kernels import triton_dequantize_int8_per_token as _triton_dequantize_int8_per_token

    return _triton_dequantize_int8_per_token(q, scale, dtype=dtype)


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


def quantized_kv_attention(q, k, v, *, quant_dtype: str = "int8", attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False, scale=None, eps: float = 1e-6, backend: str = "auto"):
    """Attention wrapper that stores K/V through a quantized roundtrip before SDPA.

    This exposes the fused quantized-KV attention contract today. `backend="auto"`
    uses a Triton-ready dispatch point on CUDA when available, then safely falls
    back to portable PyTorch dequant+SDPA. A future kernel can replace the Triton
    branch without changing callers.
    """

    _dispatch = select_quantized_attention_backend(q, k, requested=backend, quant_dtype=quant_dtype)
    from .kv_quality import _roundtrip

    k_dequant = _roundtrip(k, quant_dtype=quant_dtype, eps=eps)
    v_dequant = _roundtrip(v, quant_dtype=quant_dtype, eps=eps)
    return scaled_dot_product_attention(
        q,
        k_dequant,
        v_dequant,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def qkv_rope_attention(
    x,
    qkv_weight,
    out_weight,
    cos=None,
    sin=None,
    qkv_bias=None,
    out_bias=None,
    num_heads: int = 1,
    dropout_p: float = 0.0,
    is_causal: bool = True,
):
    """Transformer attention primitive: qkv projection, optional RoPE, SDPA, output projection."""

    q, k, v = project_qkv(x, qkv_weight, qkv_bias, num_heads)
    if cos is not None and sin is not None:
        q, k = apply_rope(q, k, cos, sin)
    out = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal)
    return merge_attention_output(out, out_weight, out_bias)


def qkv_rope_attention_cached(
    x,
    qkv_weight,
    out_weight,
    *,
    kv_cache,
    layer_idx: int,
    cache_position: int | None = None,
    cos=None,
    sin=None,
    qkv_bias=None,
    out_bias=None,
    num_heads: int = 1,
    dropout_p: float = 0.0,
):
    """Attention primitive that writes K/V into StaticKVCache and attends over cached tokens."""

    q, k_new, v_new = project_qkv(x, qkv_weight, qkv_bias, num_heads)
    if cos is not None and sin is not None:
        q, k_new = apply_rope(q, k_new, cos, sin)
    k, v = kv_cache.append_layer(layer_idx, k_new, v_new, position=cache_position)
    out = scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=False)
    return merge_attention_output(out, out_weight, out_bias)


def project_qkv(x, qkv_weight, qkv_bias, num_heads: int):
    torch = require_torch()
    functional = torch.nn.functional
    batch, seq, hidden = x.shape
    if hidden % num_heads != 0:
        raise ValueError("hidden size must be divisible by num_heads")
    qkv = functional.linear(x, qkv_weight, qkv_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    head_dim = hidden // num_heads
    q = q.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, num_heads, head_dim).transpose(1, 2)
    return q, k, v


def merge_attention_output(out, out_weight, out_bias=None):
    torch = require_torch()
    functional = torch.nn.functional
    batch, heads, seq, head_dim = out.shape
    merged = out.transpose(1, 2).contiguous().view(batch, seq, heads * head_dim)
    return functional.linear(merged, out_weight, out_bias)


def chunked_cross_entropy(logits, targets, chunk_size: int = 1024, ignore_index: int = -100, reduction: str = "mean"):
    """Cross entropy over token chunks to avoid one huge loss allocation."""

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
            pieces.append(functional.cross_entropy(flat_logits[start:end], flat_targets[start:end], ignore_index=ignore_index, reduction="none"))
        return torch.cat(pieces, dim=0).reshape_as(targets)

    total = flat_logits.new_zeros(())
    valid = flat_logits.new_zeros(())
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        target_chunk = flat_targets[start:end]
        total = total + functional.cross_entropy(flat_logits[start:end], target_chunk, ignore_index=ignore_index, reduction="sum")
        valid = valid + (target_chunk != ignore_index).sum().to(dtype=flat_logits.dtype)

    if reduction == "sum":
        return total
    return total / valid.clamp_min(1)


def linear_cross_entropy(hidden, lm_head_weight, targets, bias=None, chunk_size: int = 1024, ignore_index: int = -100, reduction: str = "mean"):
    """Compute LM-head projection and CE in token chunks without materializing full logits."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")

    torch = require_torch()
    functional = torch.nn.functional
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)

    if reduction == "none":
        pieces = []
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            logits = functional.linear(flat_hidden[start:end], lm_head_weight, bias)
            pieces.append(functional.cross_entropy(logits, flat_targets[start:end], ignore_index=ignore_index, reduction="none"))
        return torch.cat(pieces, dim=0).reshape_as(targets)

    total = flat_hidden.new_zeros(())
    valid = flat_hidden.new_zeros(())
    for start in range(0, flat_hidden.shape[0], chunk_size):
        end = min(start + chunk_size, flat_hidden.shape[0])
        target_chunk = flat_targets[start:end]
        logits = functional.linear(flat_hidden[start:end], lm_head_weight, bias)
        total = total + functional.cross_entropy(logits, target_chunk, ignore_index=ignore_index, reduction="sum")
        valid = valid + (target_chunk != ignore_index).sum().to(dtype=flat_hidden.dtype)

    if reduction == "sum":
        return total
    return total / valid.clamp_min(1)


def _expand_rope_cache(cache, target):
    if cache.shape[-1] * 2 == target.shape[-1]:
        cache = cache.repeat_interleave(2, dim=-1)
    while cache.dim() < target.dim():
        cache = cache.unsqueeze(0)
    return cache.to(device=target.device, dtype=target.dtype)




