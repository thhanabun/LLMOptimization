from __future__ import annotations

from .kernels import qkv_rope_attention, qkv_rope_attention_cached, require_torch, rms_norm_manual_backward, swiglu


def _torch_nn():
    torch = require_torch()
    return torch, torch.nn


def _lazy_torch_module():
    try:
        import torch
    except ImportError:  # pragma: no cover

        class _Fallback:
            pass

        return _Fallback
    return torch.nn.Module


class OptimizedRMSNorm(_lazy_torch_module()):
    """RMSNorm module backed by the compact manual-backward kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = False):
        torch, nn = _torch_nn()
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps

    def forward(self, x):
        return rms_norm_manual_backward(x, self.weight, self.eps, self.bias)


class OptimizedSwiGLUMLP(_lazy_torch_module()):
    """SwiGLU MLP block with separate gate/up/down projections."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False, use_triton: bool = False):
        torch, nn = _torch_nn()
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.use_triton = use_triton

    def forward(self, x):
        return swiglu(
            x,
            self.gate_proj.weight,
            self.up_proj.weight,
            self.down_proj.weight,
            self.gate_proj.bias,
            self.up_proj.bias,
            self.down_proj.bias,
            use_triton=self.use_triton,
        )


class OptimizedSelfAttention(_lazy_torch_module()):
    """Self-attention block using qkv packing, optional RoPE, SDPA, and optional StaticKVCache."""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False, dropout_p: float = 0.0, layer_idx: int | None = None):
        torch, nn = _torch_nn()
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        *,
        is_causal: bool = True,
        kv_cache=None,
        layer_idx: int | None = None,
        cache_position: int | None = None,
    ):
        effective_layer_idx = self.layer_idx if layer_idx is None else layer_idx
        if kv_cache is not None:
            if effective_layer_idx is None:
                raise ValueError("layer_idx is required when kv_cache is provided")
            return qkv_rope_attention_cached(
                x,
                self.qkv_proj.weight,
                self.out_proj.weight,
                kv_cache=kv_cache,
                layer_idx=effective_layer_idx,
                cache_position=cache_position,
                cos=cos,
                sin=sin,
                qkv_bias=self.qkv_proj.bias,
                out_bias=self.out_proj.bias,
                num_heads=self.num_heads,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        return qkv_rope_attention(
            x,
            self.qkv_proj.weight,
            self.out_proj.weight,
            cos=cos,
            sin=sin,
            qkv_bias=self.qkv_proj.bias,
            out_bias=self.out_proj.bias,
            num_heads=self.num_heads,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
        )


class OptimizedDecoderBlock(_lazy_torch_module()):
    """Llama-style pre-norm decoder block built from optimized primitives."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float = 1e-6,
        bias: bool = False,
        dropout_p: float = 0.0,
        layer_idx: int | None = None,
        use_triton: bool = False,
    ):
        torch, nn = _torch_nn()
        super().__init__()
        self.layer_idx = layer_idx
        self.input_norm = OptimizedRMSNorm(hidden_size, eps=norm_eps)
        self.self_attn = OptimizedSelfAttention(hidden_size, num_heads, bias=bias, dropout_p=dropout_p, layer_idx=layer_idx)
        self.post_attention_norm = OptimizedRMSNorm(hidden_size, eps=norm_eps)
        self.mlp = OptimizedSwiGLUMLP(hidden_size, intermediate_size, bias=bias, use_triton=use_triton)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        *,
        is_causal: bool = True,
        kv_cache=None,
        cache_position: int | None = None,
        layer_idx: int | None = None,
    ):
        effective_layer_idx = self.layer_idx if layer_idx is None else layer_idx
        x = x + self.self_attn(
            self.input_norm(x),
            cos=cos,
            sin=sin,
            is_causal=is_causal,
            kv_cache=kv_cache,
            layer_idx=effective_layer_idx,
            cache_position=cache_position,
        )
        x = x + self.mlp(self.post_attention_norm(x))
        return x


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    """Build interleaved RoPE cos/sin cache shaped [seq, head_dim]."""

    torch = require_torch()
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)
