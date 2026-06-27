from __future__ import annotations

from typing import Any

from .kernels import apply_rope, require_torch, scaled_dot_product_attention


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


def triton_rms_norm(x, weight, eps: float = 1e-6, bias=None):
    """RMSNorm using a Triton forward kernel when available, otherwise PyTorch.

    The Triton path is forward-only and intended for inference. Training code can
    keep using `rms_norm_manual_backward` for a compact PyTorch autograd path.
    """

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
    """RoPE wrapper with PyTorch fallback.

    The public hook exists so callers can select a Triton backend once a GPU is
    present. For unsupported shapes/devices it intentionally falls back to the
    reference PyTorch implementation.
    """

    if not (_can_use_triton(q) and q.shape[-1] % 2 == 0):
        return apply_rope(q, k, cos, sin)
    return _torch_rope_for_now(q, k, cos, sin)


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


def _torch_rope_for_now(q, k, cos, sin):
    return apply_rope(q, k, cos, sin)


def _can_use_triton(tensor: Any) -> bool:
    return bool(triton_available() and hasattr(tensor, "is_cuda") and tensor.is_cuda)
