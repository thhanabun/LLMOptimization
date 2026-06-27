"""Memory-first LLM analysis toolkit."""

from .benchmark import BenchmarkConfig, BenchmarkResult, benchmark_callable, benchmark_decode, benchmark_forward, compare_benchmarks
from .bytes import dtype_size_bytes, format_bytes, parse_bytes
from .estimates import MemoryEstimate, TransformerConfig, estimate_transformer_memory, preset_config
from .html_report import trace_to_html, write_trace_html
from .inspector import ModelArchitectureInfo, inspect_model, load_hf_model
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
    triton_dequantize_int8_per_token,
    triton_quantize_int8_per_token,
    triton_rms_norm,
    triton_swiglu_activation,
)
from .kv_cache import (
    DecodeConfig,
    DecodeResult,
    KVCacheConfig,
    QuantizedStaticKVCache,
    StaticKVCache,
    dequantize_int8_per_token,
    greedy_decode,
    quantize_int8_per_token,
    sample_next_token,
)
from .kv_quality import KVQualityResult, evaluate_int8_kv_quality
from .patchers import PatchReport, optimize_hf_model
from .planner import BufferPlan, MemoryPlanner, TensorLifetime
from .torch_debugger import TorchTrace, trace_forward
from .triton_kernels import triton_available

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "BufferPlan",
    "DecodeConfig",
    "DecodeResult",
    "GraphSpec",
    "KVCacheConfig",
    "KVQualityResult",
    "KernelConfig",
    "MemoryEstimate",
    "MemoryPlanner",
    "ModelArchitectureInfo",
    "OperationSpec",
    "PatchReport",
    "QuantizedStaticKVCache",
    "StaticKVCache",
    "TensorLifetime",
    "TensorSpec",
    "TorchTrace",
    "TransformerConfig",
    "apply_rope",
    "benchmark_callable",
    "benchmark_decode",
    "benchmark_forward",
    "chunked_cross_entropy",
    "compare_benchmarks",
    "dequantize_int8_per_token",
    "dtype_size_bytes",
    "estimate_transformer_memory",
    "evaluate_int8_kv_quality",
    "format_bytes",
    "greedy_decode",
    "inspect_model",
    "kernel",
    "linear_cross_entropy",
    "load_hf_model",
    "optimize_hf_model",
    "parse_bytes",
    "preset_config",
    "qkv_rope_attention",
    "qkv_rope_attention_cached",
    "quantize_int8_per_token",
    "rms_norm",
    "rms_norm_manual_backward",
    "sample_next_token",
    "scaled_dot_product_attention",
    "swiglu",
    "trace_forward",
    "trace_to_html",
    "triton_apply_rope",
    "triton_available",
    "triton_dequantize_int8_per_token",
    "triton_quantize_int8_per_token",
    "triton_rms_norm",
    "triton_swiglu_activation",
    "write_trace_html",
]
