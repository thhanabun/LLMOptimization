from __future__ import annotations

from .backends.cutile import certify_cutile_decode_attention, cutile_fused_decode_attention, detect_cutile_runtime
from .decode_benchmarks import FusedDecodeBenchmark, benchmark_fused_decode_attention
from .kernel_policy import KernelPolicy
from .kernels import (
    triton_fused_int8_kv_attention,
    triton_fused_int8_paged_kv_attention,
    triton_fused_uint8_kv_attention,
    triton_fused_uint8_paged_kv_attention,
)
from .triton_kernels import triton_available

EXPERIMENTAL_WARNING = (
    "llm_memlab.experimental contains opt-in Triton kernels and policies that need shape, GPU, and numerical certification "
    "before production rollout. Use production APIs as the default fallback path."
)


def experimental_kernel_policy(*, quant_dtype: str = "int8") -> KernelPolicy:
    return KernelPolicy(backend="triton-experimental", quant_dtype=quant_dtype, allow_experimental=True)


__all__ = [
    "EXPERIMENTAL_WARNING",
    "FusedDecodeBenchmark",
    "benchmark_fused_decode_attention",
    "certify_cutile_decode_attention",
    "cutile_fused_decode_attention",
    "detect_cutile_runtime",
    "experimental_kernel_policy",
    "triton_available",
    "triton_fused_int8_kv_attention",
    "triton_fused_int8_paged_kv_attention",
    "triton_fused_uint8_kv_attention",
    "triton_fused_uint8_paged_kv_attention",
]
