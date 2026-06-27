from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import dequantize_int8_per_token, quantize_int8_per_token
from .report import make_table


@dataclass(frozen=True)
class KVQualityResult:
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    fp_bytes: int
    quantized_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.quantized_bytes == 0:
            return 0.0
        return self.fp_bytes / self.quantized_bytes

    def to_text(self) -> str:
        rows = [
            ("Mean abs error", f"{self.mean_abs_error:.6f}"),
            ("Max abs error", f"{self.max_abs_error:.6f}"),
            ("Cosine similarity", f"{self.cosine_similarity:.6f}"),
            ("FP bytes", self.fp_bytes),
            ("Quantized bytes", self.quantized_bytes),
            ("Compression", f"{self.compression_ratio:.2f}x"),
        ]
        return make_table(("Metric", "Value"), rows)


def evaluate_int8_kv_quality(x, *, eps: float = 1e-6) -> KVQualityResult:
    torch = _import_torch()
    q, scale = quantize_int8_per_token(x, eps=eps)
    y = dequantize_int8_per_token(q, scale, dtype=x.dtype)
    diff = (x - y).float().abs()
    cosine = torch.nn.functional.cosine_similarity(x.float().flatten(), y.float().flatten(), dim=0)
    fp_bytes = x.numel() * x.element_size()
    quantized_bytes = q.numel() * q.element_size() + scale.numel() * scale.element_size()
    return KVQualityResult(
        mean_abs_error=float(diff.mean().item()),
        max_abs_error=float(diff.max().item()),
        cosine_similarity=float(cosine.item()),
        fp_bytes=int(fp_bytes),
        quantized_bytes=int(quantized_bytes),
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("KV quality evaluation requires PyTorch. Install with: pip install torch") from exc
    return torch
