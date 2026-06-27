from __future__ import annotations

from typing import Any

from .kernels import apply_rope, require_torch


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


def triton_rms_norm(x, weight, eps: float = 1e-6, bias=None):
    """RMSNorm using a Triton forward kernel when available, otherwise PyTorch."""

    if not _can_use_triton(x):
        from .kernels import rms_norm

        return rms_norm(x, weight, eps=eps, bias=bias)

    torch = require_torch()
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(X, W, B, Y, N: tl.constexpr, EPS: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X + row * N + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(x * x, axis=0) / N
        y = x * tl.rsqrt(mean_sq + EPS) * w
        if HAS_BIAS:
            b = tl.load(B + offsets, mask=mask, other=0.0).to(tl.float32)
            y += b
        tl.store(Y + row * N + offsets, y, mask=mask)

    original_shape = x.shape
    hidden = original_shape[-1]
    flat = x.contiguous().view(-1, hidden)
    out = torch.empty_like(flat)
    block = triton.next_power_of_2(hidden)
    if block > 131072:
        from .kernels import rms_norm

        return rms_norm(x, weight, eps=eps, bias=bias)
    _kernel[(flat.shape[0],)](flat, weight, bias, out, hidden, eps, bias is not None, BLOCK=block)
    return out.view(original_shape)


def triton_apply_rope(q, k, cos, sin):
    """Apply RoPE with a Triton kernel for CUDA 4D tensors, otherwise PyTorch."""

    if not (_can_use_triton(q) and q.dim() == 4 and k.shape == q.shape and q.shape[-1] % 2 == 0):
        return apply_rope(q, k, cos, sin)

    torch = require_torch()
    import triton
    import triton.language as tl

    cos, sin = _prepare_rope_cache(cos, sin, q)
    q_c = q.contiguous()
    k_c = k.contiguous()
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    q_out = torch.empty_like(q_c)
    k_out = torch.empty_like(k_c)
    total = q_c.numel()
    seq = q_c.shape[-2]
    dim = q_c.shape[-1]

    @triton.jit
    def _kernel(Q, K, COS, SIN, QO, KO, TOTAL: tl.constexpr, SEQ: tl.constexpr, DIM: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        d = offsets % DIM
        s = (offsets // DIM) % SEQ
        is_odd = d % 2
        pair_offsets = offsets - is_odd
        even_q = tl.load(Q + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        odd_q = tl.load(Q + pair_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        even_k = tl.load(K + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        odd_k = tl.load(K + pair_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        q_val = tl.load(Q + offsets, mask=mask, other=0.0).to(tl.float32)
        k_val = tl.load(K + offsets, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(COS + s * DIM + d, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(SIN + s * DIM + d, mask=mask, other=0.0).to(tl.float32)
        rot_q = tl.where(is_odd == 1, even_q, -odd_q)
        rot_k = tl.where(is_odd == 1, even_k, -odd_k)
        tl.store(QO + offsets, q_val * c + rot_q * sn, mask=mask)
        tl.store(KO + offsets, k_val * c + rot_k * sn, mask=mask)

    block = 256
    grid = (triton.cdiv(total, block),)
    _kernel[grid](q_c, k_c, cos_c, sin_c, q_out, k_out, total, seq, dim, BLOCK=block)
    return q_out.view_as(q), k_out.view_as(k)


def triton_swiglu_activation(gate, up):
    """Compute silu(gate) * up with Triton when available, otherwise PyTorch."""

    if not _can_use_triton(gate):
        torch = require_torch()
        return torch.nn.functional.silu(gate) * up

    torch = require_torch()
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(G, U, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        g = tl.load(G + offsets, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (g / (1.0 + tl.exp(-g))) * u
        tl.store(Y + offsets, y, mask=mask)

    gate_c = gate.contiguous()
    up_c = up.contiguous()
    out = torch.empty_like(gate_c)
    n = gate_c.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _kernel[grid](gate_c, up_c, out, n, BLOCK=block)
    return out.view_as(gate)


def triton_quantize_int8_per_token(x, *, eps: float = 1e-6):
    """Quantize last-dimension vectors to int8 with Triton on CUDA, otherwise PyTorch."""

    if not _can_use_triton(x):
        from .kv_cache import quantize_int8_per_token

        return quantize_int8_per_token(x, eps=eps)

    torch = require_torch()
    import triton
    import triton.language as tl

    original_shape = x.shape
    dim = original_shape[-1]
    flat = x.contiguous().view(-1, dim)
    q = torch.empty_like(flat, dtype=torch.int8)
    scales = torch.empty((flat.shape[0], 1), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(dim)
    if block > 131072:
        from .kv_cache import quantize_int8_per_token

        return quantize_int8_per_token(x, eps=eps)

    @triton.jit
    def _kernel(X, Q, S, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        vals = tl.load(X + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.max(tl.abs(vals), axis=0)
        scale = tl.maximum(absmax, EPS) / 127.0
        q_vals = tl.extra.libdevice.nearbyint(vals / scale)
        q_vals = tl.maximum(tl.minimum(q_vals, 127.0), -127.0)
        tl.store(Q + row * D + offsets, q_vals, mask=mask)
        tl.store(S + row, scale)

    _kernel[(flat.shape[0],)](flat, q, scales, dim, eps, BLOCK=block)
    return q.view(original_shape), scales.view(*original_shape[:-1], 1)


def triton_dequantize_int8_per_token(q, scale, *, dtype=None):
    """Dequantize int8 vectors.

    The public hook mirrors `triton_quantize_int8_per_token`. Dequantization is
    currently delegated to the vectorized PyTorch reference path because it is
    bandwidth-bound and keeps CPU/GPU behavior identical while the cache-aware
    attention kernels mature.
    """

    from .kv_cache import dequantize_int8_per_token

    return dequantize_int8_per_token(q, scale, dtype=dtype)


def _prepare_rope_cache(cos, sin, target):
    if cos.shape[-1] * 2 == target.shape[-1]:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    while cos.dim() > 2:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.dim() == 1:
        cos = cos.unsqueeze(0).expand(target.shape[-2], -1)
        sin = sin.unsqueeze(0).expand(target.shape[-2], -1)
    return cos.to(device=target.device, dtype=target.dtype), sin.to(device=target.device, dtype=target.dtype)


def _can_use_triton(tensor: Any) -> bool:
    return bool(triton_available() and hasattr(tensor, "is_cuda") and tensor.is_cuda)

