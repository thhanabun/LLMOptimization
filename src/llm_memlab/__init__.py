"""Memory-first LLM analysis toolkit."""

from .bytes import dtype_size_bytes, format_bytes, parse_bytes
from .estimates import (
    MemoryEstimate,
    TransformerConfig,
    estimate_transformer_memory,
    preset_config,
)
from .ir import GraphSpec, OperationSpec, TensorSpec
from .kernels import (
    KernelConfig,
    apply_rope,
    chunked_cross_entropy,
    kernel,
    linear_cross_entropy,
    qkv_rope_attention,
    qkv_rope_attention_cached,
    rms_norm,
    rms_norm_manual_backward,
    scaled_dot_product_attention,
    swiglu,
    triton_apply_rope,
    triton_rms_norm,
    triton_swiglu_activation,
)
from .kv_cache import DecodeConfig, DecodeResult, KVCacheConfig, StaticKVCache, greedy_decode, sample_next_token
from .planner import BufferPlan, MemoryPlanner, TensorLifetime
from .torch_debugger import TorchTrace, trace_forward
from .triton_kernels import triton_available

__all__ = [
    "BufferPlan",
    "DecodeConfig",
    "DecodeResult",
    "GraphSpec",
    "KVCacheConfig",
    "KernelConfig",
    "MemoryEstimate",
    "MemoryPlanner",
    "OperationSpec",
    "StaticKVCache",
    "TensorLifetime",
    "TensorSpec",
    "TorchTrace",
    "TransformerConfig",
    "apply_rope",
    "chunked_cross_entropy",
    "dtype_size_bytes",
    "estimate_transformer_memory",
    "format_bytes",
    "greedy_decode",
    "kernel",
    "linear_cross_entropy",
    "parse_bytes",
    "preset_config",
    "qkv_rope_attention",
    "qkv_rope_attention_cached",
    "rms_norm",
    "rms_norm_manual_backward",
    "sample_next_token",
    "scaled_dot_product_attention",
    "swiglu",
    "trace_forward",
    "triton_apply_rope",
    "triton_available",
    "triton_rms_norm",
    "triton_swiglu_activation",
]
