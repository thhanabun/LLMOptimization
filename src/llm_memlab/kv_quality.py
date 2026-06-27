from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kv_cache import (
    dequantize_int8_per_token,
    dequantize_uint8_per_token,
    quantize_int8_per_token,
    quantize_uint8_per_token,
    resolve_kv_storage_dtype,
)
from .report import make_table


@dataclass(frozen=True)
class KVQualityResult:
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    fp_bytes: int
    quantized_bytes: int
    quant_dtype: str = "int8"

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes == 0:
            return 0.0
        return self.fp_bytes / self.quantized_bytes

    def to_text(self) -> str:
        rows = [
            ("Quant dtype", self.quant_dtype),
            ("Mean abs error", f"{self.mean_abs_error:.6f}"),
            ("Max abs error", f"{self.max_abs_error:.6f}"),
            ("Cosine similarity", f"{self.cosine_similarity:.6f}"),
            ("FP bytes", self.fp_bytes),
            ("Quantized bytes", self.quantized_bytes),
            ("Compression", f"{self.compression_ratio:.2f}x"),
        ]
        return make_table(("Metric", "Value"), rows)


@dataclass(frozen=True)
class AttentionKVQualityResult:
    kv_quality: KVQualityResult
    output_mean_abs_error: float
    output_max_abs_error: float
    output_cosine_similarity: float
    attention_shape: tuple[int, ...]

    @property
    def compression_ratio(self) -> float:
        return self.kv_quality.compression_ratio

    def to_text(self) -> str:
        rows = [
            ("Quant dtype", self.kv_quality.quant_dtype),
            ("Attention output shape", self.attention_shape),
            ("Output mean abs error", f"{self.output_mean_abs_error:.6f}"),
            ("Output max abs error", f"{self.output_max_abs_error:.6f}"),
            ("Output cosine similarity", f"{self.output_cosine_similarity:.6f}"),
            ("KV mean abs error", f"{self.kv_quality.mean_abs_error:.6f}"),
            ("KV max abs error", f"{self.kv_quality.max_abs_error:.6f}"),
            ("KV cosine similarity", f"{self.kv_quality.cosine_similarity:.6f}"),
            ("FP KV bytes", self.kv_quality.fp_bytes),
            ("Stored KV bytes", self.kv_quality.quantized_bytes),
            ("Compression", f"{self.compression_ratio:.2f}x"),
        ]
        return make_table(("Metric", "Value"), rows)


def evaluate_int8_kv_quality(x, *, eps: float = 1e-6) -> KVQualityResult:
    return evaluate_kv_quantization_quality(x, quant_dtype="int8", eps=eps)


def evaluate_kv_quantization_quality(x, *, quant_dtype: str | Any = "int8", eps: float = 1e-6) -> KVQualityResult:
    torch = _import_torch()
    dtype_name, storage_dtype = resolve_kv_storage_dtype(quant_dtype)
    if dtype_name == "int8":
        q, scale = quantize_int8_per_token(x, eps=eps)
        y = dequantize_int8_per_token(q, scale, dtype=x.dtype)
        quantized_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size()
    elif dtype_name == "uint8":
        q, scale, zero_point = quantize_uint8_per_token(x, eps=eps)
        y = dequantize_uint8_per_token(q, scale, zero_point, dtype=x.dtype)
        quantized_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size() + zero_point.numel() * zero_point.element_size()
    else:
        stored = x.to(dtype=storage_dtype)
        y = stored.to(dtype=x.dtype)
        quantized_bytes = stored.numel() * stored.element_size()
    return _quality_result(torch, x, y, quantized_bytes=quantized_bytes, quant_dtype=dtype_name)


def evaluate_attention_kv_quality(q, k, v, *, quant_dtype: str | Any = "int8", is_causal: bool = False, eps: float = 1e-6) -> AttentionKVQualityResult:
    torch = _import_torch()
    dtype_name, _ = resolve_kv_storage_dtype(quant_dtype)
    k_quant = _roundtrip(k, quant_dtype=quant_dtype, eps=eps)
    v_quant = _roundtrip(v, quant_dtype=quant_dtype, eps=eps)
    baseline = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    compressed = torch.nn.functional.scaled_dot_product_attention(q, k_quant, v_quant, is_causal=is_causal)
    diff = (baseline - compressed).float().abs()
    cosine = torch.nn.functional.cosine_similarity(baseline.float().flatten(), compressed.float().flatten(), dim=0)
    kv_quality = _merged_kv_quality(k, v, k_quant, v_quant, quant_dtype=dtype_name, original_quant_dtype=quant_dtype, eps=eps)
    return AttentionKVQualityResult(
        kv_quality=kv_quality,
        output_mean_abs_error=float(diff.mean().item()),
        output_max_abs_error=float(diff.max().item()),
        output_cosine_similarity=float(cosine.item()),
        attention_shape=tuple(baseline.shape),
    )


def _roundtrip(x, *, quant_dtype: str | Any, eps: float):
    _, storage_dtype = resolve_kv_storage_dtype(quant_dtype)
    dtype_name, _ = resolve_kv_storage_dtype(quant_dtype)
    if dtype_name == "int8":
        q, scale = quantize_int8_per_token(x, eps=eps)
        return dequantize_int8_per_token(q, scale, dtype=x.dtype)
    if dtype_name == "uint8":
        q, scale, zero_point = quantize_uint8_per_token(x, eps=eps)
        return dequantize_uint8_per_token(q, scale, zero_point, dtype=x.dtype)
    return x.to(dtype=storage_dtype).to(dtype=x.dtype)


def _merged_kv_quality(k, v, k_quant, v_quant, *, quant_dtype: str, original_quant_dtype: str | Any, eps: float) -> KVQualityResult:
    torch = _import_torch()
    original = torch.cat([k.float().flatten(), v.float().flatten()])
    restored = torch.cat([k_quant.float().flatten(), v_quant.float().flatten()])
    quantized_bytes = _stored_bytes(k, quant_dtype=original_quant_dtype, eps=eps) + _stored_bytes(v, quant_dtype=original_quant_dtype, eps=eps)
    fp_bytes = k.numel() * k.element_size() + v.numel() * v.element_size()
    return _quality_result(torch, original, restored, quantized_bytes=quantized_bytes, quant_dtype=quant_dtype, fp_bytes=fp_bytes)


def _stored_bytes(x, *, quant_dtype: str | Any, eps: float) -> int:
    _, storage_dtype = resolve_kv_storage_dtype(quant_dtype)
    dtype_name, _ = resolve_kv_storage_dtype(quant_dtype)
    if dtype_name == "int8":
        q, scale = quantize_int8_per_token(x, eps=eps)
        return q.numel() * q.element_size() + scale.numel() * scale.element_size()
    if dtype_name == "uint8":
        q, scale, zero_point = quantize_uint8_per_token(x, eps=eps)
        return q.numel() * q.element_size() + scale.numel() * scale.element_size() + zero_point.numel() * zero_point.element_size()
    stored = x.to(dtype=storage_dtype)
    return stored.numel() * stored.element_size()


def _quality_result(torch, x, y, *, quantized_bytes: int, quant_dtype: str, fp_bytes: int | None = None) -> KVQualityResult:
    diff = (x.float() - y.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0)
    if fp_bytes is None:
        fp_bytes = x.numel() * x.element_size()
    return KVQualityResult(
        mean_abs_error=float(diff.mean().item()),
        max_abs_error=float(diff.max().item()),
        cosine_similarity=float(cosine.item()),
        fp_bytes=int(fp_bytes),
        quantized_bytes=int(quantized_bytes),
        quant_dtype=quant_dtype,
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("KV quality evaluation requires PyTorch. Install with: pip install torch") from exc
    return torch

