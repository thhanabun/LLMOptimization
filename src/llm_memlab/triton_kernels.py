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
    """Dequantize int8 vectors with a generic Triton last-dim kernel on CUDA."""

    if not _can_use_triton(q):
        from .kv_cache import dequantize_int8_per_token

        return dequantize_int8_per_token(q, scale, dtype=dtype)

    torch = require_torch()
    import triton
    import triton.language as tl

    original_shape = q.shape
    dim = original_shape[-1]
    q_flat = q.contiguous().view(-1, dim)
    scale_flat = scale.contiguous().view(-1, 1)
    out_dtype = dtype or torch.float16
    out = torch.empty(q_flat.shape, device=q.device, dtype=out_dtype)
    block = triton.next_power_of_2(dim)
    if block > 131072:
        from .kv_cache import dequantize_int8_per_token

        return dequantize_int8_per_token(q, scale, dtype=dtype)

    @triton.jit
    def _kernel(Q, S, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        qv = tl.load(Q + row * D + offsets, mask=mask, other=0).to(tl.float32)
        sc = tl.load(S + row).to(tl.float32)
        tl.store(OUT + row * D + offsets, qv * sc, mask=mask)

    _kernel[(q_flat.shape[0],)](q_flat, scale_flat, out, dim, BLOCK=block)
    return out.view(original_shape)


def triton_quantize_uint8_per_token(x, *, eps: float = 1e-6):
    """Asymmetric uint8 quantization with Triton on CUDA, otherwise PyTorch."""

    if not _can_use_triton(x):
        from .kv_cache import quantize_uint8_per_token

        return quantize_uint8_per_token(x, eps=eps)

    torch = require_torch()
    import triton
    import triton.language as tl

    original_shape = x.shape
    dim = original_shape[-1]
    flat = x.contiguous().view(-1, dim)
    q = torch.empty_like(flat, dtype=torch.uint8)
    scales = torch.empty((flat.shape[0], 1), device=x.device, dtype=torch.float32)
    zero_points = torch.empty((flat.shape[0], 1), device=x.device, dtype=torch.uint8)
    block = triton.next_power_of_2(dim)
    if block > 131072:
        from .kv_cache import quantize_uint8_per_token

        return quantize_uint8_per_token(x, eps=eps)

    @triton.jit
    def _kernel(X, Q, S, Z, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        vals = tl.load(X + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
        min_vals = tl.load(X + row * D + offsets, mask=mask, other=3.4028234663852886e38).to(tl.float32)
        max_vals = tl.load(X + row * D + offsets, mask=mask, other=-3.4028234663852886e38).to(tl.float32)
        x_min = tl.min(min_vals, axis=0)
        x_max = tl.max(max_vals, axis=0)
        scale = tl.maximum(x_max - x_min, EPS) / 255.0
        zp = tl.extra.libdevice.nearbyint(tl.maximum(tl.minimum(-x_min / scale, 255.0), 0.0))
        q_vals = tl.extra.libdevice.nearbyint(vals / scale + zp)
        q_vals = tl.maximum(tl.minimum(q_vals, 255.0), 0.0)
        tl.store(Q + row * D + offsets, q_vals, mask=mask)
        tl.store(S + row, scale)
        tl.store(Z + row, zp)

    _kernel[(flat.shape[0],)](flat, q, scales, zero_points, dim, eps, BLOCK=block)
    return q.view(original_shape), scales.view(*original_shape[:-1], 1), zero_points.view(*original_shape[:-1], 1)


def triton_dequantize_uint8_per_token(q, scale, zero_point, *, dtype=None):
    """Dequantize uint8 vectors with a generic Triton last-dim kernel on CUDA."""

    if not _can_use_triton(q):
        from .kv_cache import dequantize_uint8_per_token

        return dequantize_uint8_per_token(q, scale, zero_point, dtype=dtype)

    torch = require_torch()
    import triton
    import triton.language as tl

    original_shape = q.shape
    dim = original_shape[-1]
    q_flat = q.contiguous().view(-1, dim)
    scale_flat = scale.contiguous().view(-1, 1)
    zp_flat = zero_point.contiguous().view(-1, 1)
    out_dtype = dtype or torch.float16
    out = torch.empty(q_flat.shape, device=q.device, dtype=out_dtype)
    block = triton.next_power_of_2(dim)
    if block > 131072:
        from .kv_cache import dequantize_uint8_per_token

        return dequantize_uint8_per_token(q, scale, zero_point, dtype=dtype)

    @triton.jit
    def _kernel(Q, S, Z, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        qv = tl.load(Q + row * D + offsets, mask=mask, other=0).to(tl.float32)
        sc = tl.load(S + row).to(tl.float32)
        zp = tl.load(Z + row).to(tl.float32)
        tl.store(OUT + row * D + offsets, (qv - zp) * sc, mask=mask)

    _kernel[(q_flat.shape[0],)](q_flat, scale_flat, zp_flat, out, dim, BLOCK=block)
    return out.view(original_shape)


def triton_fused_int8_kv_attention(q, k_q, k_scale, v_q, v_scale, *, scale=None, is_causal: bool = False):
    """Decode attention that dequantizes int8 K/V inside a Triton softmax kernel.

    Fast path supports single-token non-causal decode with MHA/GQA/MQA layouts:
    q [B,QH,1,D], quantized K/V [B,KVH,T,D], per-token scales [B,KVH,T,1].
    """

    torch = require_torch()
    if is_causal or not _supports_fused_decode(q, k_q, v_q, k_scale, v_scale):
        from .kv_cache import dequantize_int8_per_token

        k = dequantize_int8_per_token(k_q, k_scale, dtype=q.dtype)
        v = dequantize_int8_per_token(v_q, v_scale, dtype=q.dtype)
        if _can_repeat_kv_for_gqa(q, k):
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    import triton
    import triton.language as tl

    q_c = q.contiguous()
    k_c = k_q.contiguous()
    v_c = v_q.contiguous()
    ks_c = k_scale.contiguous()
    vs_c = v_scale.contiguous()
    batch, q_heads, _, dim = q_c.shape
    kv_heads = k_c.shape[1]
    tokens = k_c.shape[-2]
    block_d = triton.next_power_of_2(dim)
    block_t = triton.next_power_of_2(tokens)
    if block_d > 256 or block_t > 4096:
        from .kv_cache import dequantize_int8_per_token

        k = dequantize_int8_per_token(k_q, k_scale, dtype=q.dtype)
        v = dequantize_int8_per_token(v_q, v_scale, dtype=q.dtype)
        if q_heads != kv_heads:
            repeat = q_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    out = torch.empty_like(q_c)
    sm_scale = float(scale) if scale is not None else dim**-0.5
    group = q_heads // kv_heads

    @triton.jit
    def _kernel(
        Q,
        KQ,
        KS,
        VQ,
        VS,
        OUT,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        SM: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // QH
        qh = row - b * QH
        kvh = qh // GROUP
        kv_row = b * KVH + kvh
        offs_t = tl.arange(0, BT)
        offs_d = tl.arange(0, BD)
        q_vals = tl.load(Q + row * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        kv_base = kv_row * T * D
        scale_base = kv_row * T
        k_vals = tl.load(
            KQ + kv_base + offs_t[:, None] * D + offs_d[None, :], mask=(offs_t[:, None] < T) & (offs_d[None, :] < D), other=0
        ).to(tl.float32)
        k_scales = tl.load(KS + scale_base + offs_t, mask=offs_t < T, other=0.0).to(tl.float32)
        scores = tl.sum(q_vals[None, :] * k_vals * k_scales[:, None], axis=1) * SM
        scores = tl.where(offs_t < T, scores, -3.4028234663852886e38)
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        denom = tl.sum(probs, axis=0)
        v_vals = tl.load(
            VQ + kv_base + offs_t[:, None] * D + offs_d[None, :], mask=(offs_t[:, None] < T) & (offs_d[None, :] < D), other=0
        ).to(tl.float32)
        v_scales = tl.load(VS + scale_base + offs_t, mask=offs_t < T, other=0.0).to(tl.float32)
        acc = tl.sum((probs / denom)[:, None] * v_vals * v_scales[:, None], axis=0)
        tl.store(OUT + row * D + offs_d, acc, mask=offs_d < D)

    _kernel[(batch * q_heads,)](q_c, k_c, ks_c, v_c, vs_c, out, q_heads, kv_heads, tokens, dim, group, sm_scale, BT=block_t, BD=block_d)
    return out.view_as(q)


def triton_fused_uint8_kv_attention(q, k_q, k_scale, k_zero_point, v_q, v_scale, v_zero_point, *, scale=None, is_causal: bool = False):
    """Decode attention that dequantizes asymmetric uint8 K/V inside Triton."""

    torch = require_torch()
    if is_causal or not _supports_fused_decode(q, k_q, v_q, k_scale, v_scale):
        from .kv_cache import dequantize_uint8_per_token

        k = dequantize_uint8_per_token(k_q, k_scale, k_zero_point, dtype=q.dtype)
        v = dequantize_uint8_per_token(v_q, v_scale, v_zero_point, dtype=q.dtype)
        if _can_repeat_kv_for_gqa(q, k):
            repeat = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)

    import triton
    import triton.language as tl

    q_c = q.contiguous()
    k_c = k_q.contiguous()
    v_c = v_q.contiguous()
    ks_c = k_scale.contiguous()
    vs_c = v_scale.contiguous()
    kz_c = k_zero_point.contiguous()
    vz_c = v_zero_point.contiguous()
    batch, q_heads, _, dim = q_c.shape
    kv_heads = k_c.shape[1]
    tokens = k_c.shape[-2]
    block_d = triton.next_power_of_2(dim)
    block_t = triton.next_power_of_2(tokens)
    if block_d > 256 or block_t > 4096:
        from .kv_cache import dequantize_uint8_per_token

        k = dequantize_uint8_per_token(k_q, k_scale, k_zero_point, dtype=q.dtype)
        v = dequantize_uint8_per_token(v_q, v_scale, v_zero_point, dtype=q.dtype)
        if q_heads != kv_heads:
            repeat = q_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    out = torch.empty_like(q_c)
    sm_scale = float(scale) if scale is not None else dim**-0.5
    group = q_heads // kv_heads

    @triton.jit
    def _kernel(
        Q,
        KQ,
        KS,
        KZ,
        VQ,
        VS,
        VZ,
        OUT,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        SM: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // QH
        qh = row - b * QH
        kvh = qh // GROUP
        kv_row = b * KVH + kvh
        offs_t = tl.arange(0, BT)
        offs_d = tl.arange(0, BD)
        q_vals = tl.load(Q + row * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        kv_base = kv_row * T * D
        scale_base = kv_row * T
        k_raw = tl.load(
            KQ + kv_base + offs_t[:, None] * D + offs_d[None, :], mask=(offs_t[:, None] < T) & (offs_d[None, :] < D), other=0
        ).to(tl.float32)
        k_scales = tl.load(KS + scale_base + offs_t, mask=offs_t < T, other=0.0).to(tl.float32)
        k_zp = tl.load(KZ + scale_base + offs_t, mask=offs_t < T, other=0).to(tl.float32)
        scores = tl.sum(q_vals[None, :] * (k_raw - k_zp[:, None]) * k_scales[:, None], axis=1) * SM
        scores = tl.where(offs_t < T, scores, -3.4028234663852886e38)
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        denom = tl.sum(probs, axis=0)
        v_raw = tl.load(
            VQ + kv_base + offs_t[:, None] * D + offs_d[None, :], mask=(offs_t[:, None] < T) & (offs_d[None, :] < D), other=0
        ).to(tl.float32)
        v_scales = tl.load(VS + scale_base + offs_t, mask=offs_t < T, other=0.0).to(tl.float32)
        v_zp = tl.load(VZ + scale_base + offs_t, mask=offs_t < T, other=0).to(tl.float32)
        acc = tl.sum((probs / denom)[:, None] * (v_raw - v_zp[:, None]) * v_scales[:, None], axis=0)
        tl.store(OUT + row * D + offs_d, acc, mask=offs_d < D)

    _kernel[(batch * q_heads,)](
        q_c, k_c, ks_c, kz_c, v_c, vs_c, vz_c, out, q_heads, kv_heads, tokens, dim, group, sm_scale, BT=block_t, BD=block_d
    )
    return out.view_as(q)


def triton_fused_int8_paged_kv_attention(
    q,
    k_pages,
    k_scales,
    v_pages,
    v_scales,
    page_table,
    *,
    length: int | None = None,
    lengths=None,
    page_size: int,
    scale=None,
    block_tokens: int = 4096,
):
    """Single-token decode attention over int8 paged K/V storage.

    v3 accepts a batch-specific page table shaped [batch, logical_page], per-batch
    variable lengths, and falls back to a block-wise streaming softmax for long
    contexts instead of materializing full dequantized K/V.
    """

    torch = require_torch()
    q_c, kp, ks, vp, vs, pt, lens = _prepare_paged_inputs(
        q, k_pages, k_scales, v_pages, v_scales, page_table, length=length, lengths=lengths
    )
    batch, q_heads, _, dim = q_c.shape
    kv_heads = kp.shape[1]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q heads must be divisible by kv heads for paged GQA/MQA")
    max_len = int(lens.max().item()) if lens.numel() else 0
    if max_len <= 0:
        return torch.empty_like(q_c).zero_()
    if max_len > block_tokens or not (_can_use_triton(q_c) and getattr(kp, "is_cuda", False) and getattr(pt, "is_cuda", False)):
        return _paged_int8_attention_streaming(q_c, kp, ks, vp, vs, pt, lens, page_size=page_size, scale=scale, block_tokens=block_tokens)

    import triton
    import triton.language as tl

    num_pages = kp.shape[2]
    logical_pages = pt.shape[1]
    block_d = triton.next_power_of_2(dim)
    block_t = triton.next_power_of_2(max_len)
    if block_d > 256:
        raise ValueError("paged fused attention supports head_dim<=256")
    out = torch.empty_like(q_c)
    sm_scale = float(scale) if scale is not None else dim**-0.5
    group = q_heads // kv_heads

    @triton.jit
    def _kernel(
        Q,
        KP,
        KS,
        VP,
        VS,
        PT,
        LENS,
        OUT,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        NP: tl.constexpr,
        LP: tl.constexpr,
        PS: tl.constexpr,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        SM: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // QH
        qh = row - b * QH
        kvh = qh // GROUP
        length_b = tl.load(LENS + b)
        offs_t = tl.arange(0, BT)
        offs_d = tl.arange(0, BD)
        logical_page = offs_t // PS
        page_offset = offs_t - logical_page * PS
        valid_t = offs_t < length_b
        physical_page = tl.load(PT + b * LP + logical_page, mask=valid_t & (logical_page < LP), other=0)
        q_vals = tl.load(Q + row * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        page_base = (((b * KVH + kvh) * NP + physical_page) * PS + page_offset) * D
        scale_base = ((b * KVH + kvh) * NP + physical_page) * PS + page_offset
        k_vals = tl.load(KP + page_base[:, None] + offs_d[None, :], mask=valid_t[:, None] & (offs_d[None, :] < D), other=0).to(tl.float32)
        k_scale = tl.load(KS + scale_base, mask=valid_t, other=0.0).to(tl.float32)
        scores = tl.sum(q_vals[None, :] * k_vals * k_scale[:, None], axis=1) * SM
        scores = tl.where(valid_t, scores, -3.4028234663852886e38)
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        denom = tl.sum(probs, axis=0)
        v_vals = tl.load(VP + page_base[:, None] + offs_d[None, :], mask=valid_t[:, None] & (offs_d[None, :] < D), other=0).to(tl.float32)
        v_scale = tl.load(VS + scale_base, mask=valid_t, other=0.0).to(tl.float32)
        acc = tl.sum((probs / denom)[:, None] * v_vals * v_scale[:, None], axis=0)
        tl.store(OUT + row * D + offs_d, acc, mask=offs_d < D)

    _kernel[(batch * q_heads,)](
        q_c,
        kp,
        ks,
        vp,
        vs,
        pt,
        lens,
        out,
        q_heads,
        kv_heads,
        num_pages,
        logical_pages,
        page_size,
        dim,
        group,
        sm_scale,
        BT=block_t,
        BD=block_d,
    )
    return out.view_as(q)


def triton_fused_uint8_paged_kv_attention(
    q,
    k_pages,
    k_scales,
    k_zero_points,
    v_pages,
    v_scales,
    v_zero_points,
    page_table,
    *,
    length: int | None = None,
    lengths=None,
    page_size: int,
    scale=None,
    block_tokens: int = 4096,
):
    """Single-token decode attention over asymmetric uint8 paged K/V storage."""

    torch = require_torch()
    q_c, kp, ks, vp, vs, pt, lens = _prepare_paged_inputs(
        q, k_pages, k_scales, v_pages, v_scales, page_table, length=length, lengths=lengths
    )
    kz = k_zero_points.contiguous()
    vz = v_zero_points.contiguous()
    batch, q_heads, _, dim = q_c.shape
    kv_heads = kp.shape[1]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q heads must be divisible by kv heads for paged GQA/MQA")
    max_len = int(lens.max().item()) if lens.numel() else 0
    if max_len <= 0:
        return torch.empty_like(q_c).zero_()
    if max_len > block_tokens or not (_can_use_triton(q_c) and getattr(kp, "is_cuda", False) and getattr(pt, "is_cuda", False)):
        return _paged_uint8_attention_streaming(
            q_c, kp, ks, kz, vp, vs, vz, pt, lens, page_size=page_size, scale=scale, block_tokens=block_tokens
        )

    import triton
    import triton.language as tl

    num_pages = kp.shape[2]
    logical_pages = pt.shape[1]
    block_d = triton.next_power_of_2(dim)
    block_t = triton.next_power_of_2(max_len)
    if block_d > 256:
        raise ValueError("paged fused attention supports head_dim<=256")
    out = torch.empty_like(q_c)
    sm_scale = float(scale) if scale is not None else dim**-0.5
    group = q_heads // kv_heads

    @triton.jit
    def _kernel(
        Q,
        KP,
        KS,
        KZ,
        VP,
        VS,
        VZ,
        PT,
        LENS,
        OUT,
        QH: tl.constexpr,
        KVH: tl.constexpr,
        NP: tl.constexpr,
        LP: tl.constexpr,
        PS: tl.constexpr,
        D: tl.constexpr,
        GROUP: tl.constexpr,
        SM: tl.constexpr,
        BT: tl.constexpr,
        BD: tl.constexpr,
    ):
        row = tl.program_id(0)
        b = row // QH
        qh = row - b * QH
        kvh = qh // GROUP
        length_b = tl.load(LENS + b)
        offs_t = tl.arange(0, BT)
        offs_d = tl.arange(0, BD)
        logical_page = offs_t // PS
        page_offset = offs_t - logical_page * PS
        valid_t = offs_t < length_b
        physical_page = tl.load(PT + b * LP + logical_page, mask=valid_t & (logical_page < LP), other=0)
        q_vals = tl.load(Q + row * D + offs_d, mask=offs_d < D, other=0.0).to(tl.float32)
        page_base = (((b * KVH + kvh) * NP + physical_page) * PS + page_offset) * D
        scale_base = ((b * KVH + kvh) * NP + physical_page) * PS + page_offset
        k_raw = tl.load(KP + page_base[:, None] + offs_d[None, :], mask=valid_t[:, None] & (offs_d[None, :] < D), other=0).to(tl.float32)
        k_scale = tl.load(KS + scale_base, mask=valid_t, other=0.0).to(tl.float32)
        k_zp = tl.load(KZ + scale_base, mask=valid_t, other=0).to(tl.float32)
        scores = tl.sum(q_vals[None, :] * (k_raw - k_zp[:, None]) * k_scale[:, None], axis=1) * SM
        scores = tl.where(valid_t, scores, -3.4028234663852886e38)
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        denom = tl.sum(probs, axis=0)
        v_raw = tl.load(VP + page_base[:, None] + offs_d[None, :], mask=valid_t[:, None] & (offs_d[None, :] < D), other=0).to(tl.float32)
        v_scale = tl.load(VS + scale_base, mask=valid_t, other=0.0).to(tl.float32)
        v_zp = tl.load(VZ + scale_base, mask=valid_t, other=0).to(tl.float32)
        acc = tl.sum((probs / denom)[:, None] * (v_raw - v_zp[:, None]) * v_scale[:, None], axis=0)
        tl.store(OUT + row * D + offs_d, acc, mask=offs_d < D)

    _kernel[(batch * q_heads,)](
        q_c,
        kp,
        ks,
        kz,
        vp,
        vs,
        vz,
        pt,
        lens,
        out,
        q_heads,
        kv_heads,
        num_pages,
        logical_pages,
        page_size,
        dim,
        group,
        sm_scale,
        BT=block_t,
        BD=block_d,
    )
    return out.view_as(q)


def _prepare_paged_inputs(q, k_pages, k_scales, v_pages, v_scales, page_table, *, length: int | None, lengths):
    torch = require_torch()
    q_c = q.contiguous()
    kp = k_pages.contiguous()
    vp = v_pages.contiguous()
    ks = k_scales.contiguous()
    vs = v_scales.contiguous()
    if q_c.dim() != 4 or q_c.shape[-2] != 1:
        raise ValueError("paged fused attention expects q shaped [batch, q_heads, 1, head_dim]")
    batch = q_c.shape[0]
    pt = page_table.to(device=q_c.device, dtype=torch.long).contiguous()
    if pt.dim() == 1:
        pt = pt.unsqueeze(0).expand(batch, -1).contiguous()
    if pt.dim() != 2 or pt.shape[0] != batch:
        raise ValueError("page_table must be shaped [logical_page] or [batch, logical_page]")
    if lengths is None:
        if length is None:
            length = int(pt.shape[1] * kp.shape[3])
        lens = torch.full((batch,), int(length), device=q_c.device, dtype=torch.int32)
    else:
        lens = lengths.to(device=q_c.device, dtype=torch.int32).contiguous()
        if lens.dim() != 1 or lens.shape[0] != batch:
            raise ValueError("lengths must be shaped [batch]")
    return q_c, kp, ks, vp, vs, pt, lens


def _paged_int8_attention_streaming(
    q, k_pages, k_scales, v_pages, v_scales, page_table, lengths, *, page_size: int, scale=None, block_tokens: int = 4096
):
    torch = require_torch()
    batch, q_heads, _, dim = q.shape
    kv_heads = k_pages.shape[1]
    group = q_heads // kv_heads
    sm_scale = float(scale) if scale is not None else dim**-0.5
    out = torch.empty_like(q)
    for b in range(batch):
        length = int(lengths[b].item())
        if length <= 0:
            out[b].zero_()
            continue
        for qh in range(q_heads):
            kvh = qh // group
            q_vec = q[b, qh, 0].float()
            m = q_vec.new_full((), -float("inf"))
            denom = q_vec.new_zeros(())
            acc = torch.zeros(dim, device=q.device, dtype=torch.float32)
            for start in range(0, length, block_tokens):
                end = min(start + block_tokens, length)
                k_block, v_block = _gather_int8_paged_block(
                    k_pages, k_scales, v_pages, v_scales, page_table[b], b, kvh, start, end, page_size, dtype=q.dtype
                )
                scores = torch.matmul(k_block.float(), q_vec) * sm_scale
                block_m = scores.max()
                new_m = torch.maximum(m, block_m)
                alpha = torch.exp(m - new_m) if torch.isfinite(m) else q_vec.new_zeros(())
                probs = torch.exp(scores - new_m)
                acc = acc * alpha + torch.matmul(probs, v_block.float())
                denom = denom * alpha + probs.sum()
                m = new_m
            out[b, qh, 0] = (acc / denom.clamp_min(1e-20)).to(dtype=q.dtype)
    return out


def _paged_uint8_attention_streaming(
    q,
    k_pages,
    k_scales,
    k_zero_points,
    v_pages,
    v_scales,
    v_zero_points,
    page_table,
    lengths,
    *,
    page_size: int,
    scale=None,
    block_tokens: int = 4096,
):
    torch = require_torch()
    batch, q_heads, _, dim = q.shape
    kv_heads = k_pages.shape[1]
    group = q_heads // kv_heads
    sm_scale = float(scale) if scale is not None else dim**-0.5
    out = torch.empty_like(q)
    for b in range(batch):
        length = int(lengths[b].item())
        if length <= 0:
            out[b].zero_()
            continue
        for qh in range(q_heads):
            kvh = qh // group
            q_vec = q[b, qh, 0].float()
            m = q_vec.new_full((), -float("inf"))
            denom = q_vec.new_zeros(())
            acc = torch.zeros(dim, device=q.device, dtype=torch.float32)
            for start in range(0, length, block_tokens):
                end = min(start + block_tokens, length)
                k_block, v_block = _gather_uint8_paged_block(
                    k_pages,
                    k_scales,
                    k_zero_points,
                    v_pages,
                    v_scales,
                    v_zero_points,
                    page_table[b],
                    b,
                    kvh,
                    start,
                    end,
                    page_size,
                    dtype=q.dtype,
                )
                scores = torch.matmul(k_block.float(), q_vec) * sm_scale
                block_m = scores.max()
                new_m = torch.maximum(m, block_m)
                alpha = torch.exp(m - new_m) if torch.isfinite(m) else q_vec.new_zeros(())
                probs = torch.exp(scores - new_m)
                acc = acc * alpha + torch.matmul(probs, v_block.float())
                denom = denom * alpha + probs.sum()
                m = new_m
            out[b, qh, 0] = (acc / denom.clamp_min(1e-20)).to(dtype=q.dtype)
    return out


def _gather_int8_paged_block(
    k_pages, k_scales, v_pages, v_scales, page_table, batch_idx: int, kv_head: int, start: int, end: int, page_size: int, *, dtype
):
    torch = require_torch()
    pieces_k = []
    pieces_v = []
    for token in range(start, end):
        logical_page = token // page_size
        physical_page = int(page_table[logical_page].item())
        offset = token % page_size
        k = k_pages[batch_idx, kv_head, physical_page, offset].float() * k_scales[batch_idx, kv_head, physical_page, offset].float()
        v = v_pages[batch_idx, kv_head, physical_page, offset].float() * v_scales[batch_idx, kv_head, physical_page, offset].float()
        pieces_k.append(k)
        pieces_v.append(v)
    return k_pages.new_empty((0, k_pages.shape[-1]), dtype=dtype) if not pieces_k else torch.stack(pieces_k).to(
        dtype=dtype
    ), v_pages.new_empty((0, v_pages.shape[-1]), dtype=dtype) if not pieces_v else torch.stack(pieces_v).to(dtype=dtype)


def _gather_uint8_paged_block(
    k_pages,
    k_scales,
    k_zero_points,
    v_pages,
    v_scales,
    v_zero_points,
    page_table,
    batch_idx: int,
    kv_head: int,
    start: int,
    end: int,
    page_size: int,
    *,
    dtype,
):
    torch = require_torch()
    pieces_k = []
    pieces_v = []
    for token in range(start, end):
        logical_page = token // page_size
        physical_page = int(page_table[logical_page].item())
        offset = token % page_size
        k = (
            k_pages[batch_idx, kv_head, physical_page, offset].float() - k_zero_points[batch_idx, kv_head, physical_page, offset].float()
        ) * k_scales[batch_idx, kv_head, physical_page, offset].float()
        v = (
            v_pages[batch_idx, kv_head, physical_page, offset].float() - v_zero_points[batch_idx, kv_head, physical_page, offset].float()
        ) * v_scales[batch_idx, kv_head, physical_page, offset].float()
        pieces_k.append(k)
        pieces_v.append(v)
    return k_pages.new_empty((0, k_pages.shape[-1]), dtype=dtype) if not pieces_k else torch.stack(pieces_k).to(
        dtype=dtype
    ), v_pages.new_empty((0, v_pages.shape[-1]), dtype=dtype) if not pieces_v else torch.stack(pieces_v).to(dtype=dtype)


def _can_repeat_kv_for_gqa(q, k) -> bool:
    return bool(
        getattr(q, "dim", lambda: 0)() == 4
        and getattr(k, "dim", lambda: 0)() == 4
        and q.shape[1] != k.shape[1]
        and k.shape[1] > 0
        and q.shape[1] % k.shape[1] == 0
    )


def _supports_fused_decode(q, k_q, v_q, k_scale, v_scale) -> bool:
    return bool(
        _can_use_triton(q)
        and getattr(k_q, "is_cuda", False)
        and getattr(v_q, "is_cuda", False)
        and q.dim() == 4
        and k_q.dim() == 4
        and v_q.dim() == 4
        and q.shape[-2] == 1
        and q.shape[0] == k_q.shape[0] == v_q.shape[0]
        and k_q.shape[1] > 0
        and q.shape[1] % k_q.shape[1] == 0
        and k_q.shape[1] == v_q.shape[1]
        and q.shape[-1] == k_q.shape[-1] == v_q.shape[-1]
        and k_q.shape[-2] == v_q.shape[-2]
        and tuple(k_scale.shape) == (*k_q.shape[:-1], 1)
        and tuple(v_scale.shape) == (*v_q.shape[:-1], 1)
    )


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
