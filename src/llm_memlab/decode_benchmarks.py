from __future__ import annotations

from dataclasses import dataclass

from .benchmark import BenchmarkConfig, BenchmarkResult, benchmark_callable
from .benchmark_store import BenchmarkRecord, collect_run_metadata
from .quality_metrics import LogitQualityResult, compare_logits
from .report import make_table


@dataclass(frozen=True)
class FusedDecodeBenchmark:
    fused: BenchmarkResult
    dequant_sdpa: BenchmarkResult
    quality: LogitQualityResult

    @property
    def speedup(self) -> float:
        return self.dequant_sdpa.mean_ms / self.fused.mean_ms if self.fused.mean_ms > 0 else 0.0

    def to_text(self) -> str:
        rows = [
            ("Fused decode", f"{self.fused.mean_ms:.3f} ms"),
            ("Dequant+SDPA", f"{self.dequant_sdpa.mean_ms:.3f} ms"),
            ("Speedup", f"{self.speedup:.3f}x"),
            ("Quality pass", self.quality.passed),
            ("Mean abs", f"{self.quality.mean_abs_error:.6f}"),
        ]
        return make_table(("Metric", "Value"), rows)


@dataclass(frozen=True)
class DecodeBackendMatrix:
    records: tuple[BenchmarkRecord, ...]
    qualities: dict[str, LogitQualityResult]
    cutile_backend_used: str
    cutile_fallback_reason: str | None

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.qualities.values())

    def to_text(self) -> str:
        rows = [
            (
                record.name,
                f"{record.mean_ms:.3f}",
                record.extra.get("backend_used", record.extra.get("backend", "")),
                self.qualities.get(record.name).passed if record.name in self.qualities else "",
                record.extra.get("mean_abs", ""),
            )
            for record in self.records
        ]
        header = make_table(
            ("Metric", "Value"),
            [
                ("Passed", self.passed),
                ("CuTile backend used", self.cutile_backend_used),
                ("CuTile fallback", self.cutile_fallback_reason or "n/a"),
            ],
        )
        return header + "\n\n" + make_table(("Case", "Mean ms", "Backend", "Quality", "Mean abs"), rows)


def benchmark_fused_decode_attention(
    *,
    batch: int = 1,
    q_heads: int = 8,
    kv_heads: int = 2,
    tokens: int = 128,
    head_dim: int = 64,
    dtype="fp16",
    quant_dtype: str = "int8",
    repeats: int = 20,
    warmup: int = 3,
    seed: int | None = 0,
) -> FusedDecodeBenchmark:
    torch = _import_torch()
    from .kernels import (
        scaled_dot_product_attention,
        triton_dequantize_int8_per_token,
        triton_dequantize_uint8_per_token,
        triton_fused_int8_kv_attention,
        triton_fused_uint8_kv_attention,
        triton_quantize_int8_per_token,
        triton_quantize_uint8_per_token,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_fused_decode_attention requires CUDA")
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads for GQA/MQA benchmarking")
    quant_dtype = quant_dtype.lower()
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch_dtype = torch.float16 if dtype in {"fp16", "float16"} else torch.bfloat16 if dtype in {"bf16", "bfloat16"} else torch.float32
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch_dtype)
    k = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch_dtype)
    v = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch_dtype)
    repeat = q_heads // kv_heads

    if quant_dtype == "int8":
        qk, ks = triton_quantize_int8_per_token(k)
        qv, vs = triton_quantize_int8_per_token(v)

        def fused_fn():
            return triton_fused_int8_kv_attention(q, qk, ks, qv, vs)

        def ref_fn():
            kd = triton_dequantize_int8_per_token(qk, ks, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
            vd = triton_dequantize_int8_per_token(qv, vs, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
            return scaled_dot_product_attention(q, kd, vd)
    elif quant_dtype == "uint8":
        qk, ks, kz = triton_quantize_uint8_per_token(k)
        qv, vs, vz = triton_quantize_uint8_per_token(v)

        def fused_fn():
            return triton_fused_uint8_kv_attention(q, qk, ks, kz, qv, vs, vz)

        def ref_fn():
            kd = triton_dequantize_uint8_per_token(qk, ks, kz, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
            vd = triton_dequantize_uint8_per_token(qv, vs, vz, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
            return scaled_dot_product_attention(q, kd, vd)
    else:
        raise ValueError("quant_dtype must be int8 or uint8")

    with torch.no_grad():
        fused_out = fused_fn()
        ref_out = ref_fn()
    quality = compare_logits(ref_out.float(), fused_out.float(), top_k=min(5, head_dim), max_mean_abs=0.03, min_top1=0.90)
    cfg = BenchmarkConfig(warmup=warmup, repeats=repeats)
    fused_bench = benchmark_callable("fused-decode", fused_fn, cfg)
    ref_bench = benchmark_callable("dequant-sdpa", ref_fn, cfg)
    return FusedDecodeBenchmark(fused_bench, ref_bench, quality)


def benchmark_decode_backend_matrix(
    *,
    batch: int = 1,
    q_heads: int = 8,
    kv_heads: int = 2,
    tokens: int = 128,
    head_dim: int = 64,
    page_size: int = 16,
    dtype: str = "fp16",
    quant_dtype: str = "int8",
    repeats: int = 20,
    warmup: int = 3,
    seed: int | None = 0,
) -> DecodeBackendMatrix:
    torch = _import_torch()
    from .backends.cutile import cutile_fused_decode_attention
    from .kernels import (
        scaled_dot_product_attention,
        triton_dequantize_int8_per_token,
        triton_fused_int8_kv_attention,
        triton_quantize_int8_per_token,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_decode_backend_matrix requires CUDA")
    if batch != 1:
        raise ValueError("CuTile matrix v1 targets batch=1")
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    if quant_dtype != "int8":
        raise ValueError("decode backend matrix v1 compares int8 Triton against fp paged CuTile fallback")
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch_dtype = torch.float16 if dtype in {"fp16", "float16"} else torch.bfloat16 if dtype in {"bf16", "bfloat16"} else torch.float32
    q = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=torch_dtype)
    k = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch_dtype)
    v = torch.randn(batch, kv_heads, tokens, head_dim, device="cuda", dtype=torch_dtype)
    repeat = q_heads // kv_heads
    qk, ks = triton_quantize_int8_per_token(k)
    qv, vs = triton_quantize_int8_per_token(v)
    k_pages, v_pages, page_table, lengths = _make_float_pages(torch, k, v, page_size=page_size)

    def torch_fn():
        return scaled_dot_product_attention(q, k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1))

    def dequant_fn():
        kd = triton_dequantize_int8_per_token(qk, ks, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
        vd = triton_dequantize_int8_per_token(qv, vs, dtype=torch_dtype).repeat_interleave(repeat, dim=1)
        return scaled_dot_product_attention(q, kd, vd)

    def triton_fn():
        return triton_fused_int8_kv_attention(q, qk, ks, qv, vs)

    def cutile_fn():
        return cutile_fused_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=page_size).output

    with torch.no_grad():
        reference = torch_fn()
        cutile_dispatch = cutile_fused_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=page_size)
        outputs = {
            "decode/torch-sdpa": reference,
            "decode/dequant-sdpa": dequant_fn(),
            "decode/triton-fused": triton_fn(),
            "decode/cutile-fused": cutile_dispatch.output,
        }
    qualities = {
        name: compare_logits(reference.float(), output.float(), top_k=min(5, head_dim), max_mean_abs=0.03, min_top1=0.90)
        for name, output in outputs.items()
    }
    cfg = BenchmarkConfig(warmup=warmup, repeats=repeats)
    benches = {
        "decode/torch-sdpa": benchmark_callable("decode/torch-sdpa", torch_fn, cfg),
        "decode/dequant-sdpa": benchmark_callable("decode/dequant-sdpa", dequant_fn, cfg),
        "decode/triton-fused": benchmark_callable("decode/triton-fused", triton_fn, cfg),
        "decode/cutile-fused": benchmark_callable("decode/cutile-fused", cutile_fn, cfg),
    }
    metadata = collect_run_metadata(dtype=dtype, sequence_length=tokens, warmup=warmup, repeats=repeats, seed=seed, backend="decode-matrix")
    records = []
    for name, bench in benches.items():
        quality = qualities[name]
        records.append(
            BenchmarkRecord(
                name=name,
                kind="decode",
                mean_ms=bench.mean_ms,
                min_ms=bench.min_ms,
                max_ms=bench.max_ms,
                peak_cuda_bytes=bench.peak_cuda_bytes,
                extra={
                    "q_heads": q_heads,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "tokens": tokens,
                    "page_size": page_size,
                    "quality_passed": quality.passed,
                    "mean_abs": quality.mean_abs_error,
                    "top1": quality.top1_agreement,
                    "backend_used": cutile_dispatch.backend_used if name == "decode/cutile-fused" else name.split("/")[-1],
                    "fallback_reason": cutile_dispatch.fallback_reason if name == "decode/cutile-fused" else None,
                },
                metadata=metadata.__dict__,
            )
        )
    return DecodeBackendMatrix(
        records=tuple(records),
        qualities=qualities,
        cutile_backend_used=cutile_dispatch.backend_used,
        cutile_fallback_reason=cutile_dispatch.fallback_reason,
    )


def _make_float_pages(torch, k, v, *, page_size: int):
    batch, kv_heads, tokens, head_dim = k.shape
    num_pages = (tokens + page_size - 1) // page_size
    k_pages = torch.zeros(batch, kv_heads, num_pages, page_size, head_dim, device=k.device, dtype=k.dtype)
    v_pages = torch.zeros_like(k_pages)
    page_table = torch.arange(num_pages, device=k.device, dtype=torch.long).view(1, num_pages).repeat(batch, 1)
    for page in range(num_pages):
        start = page * page_size
        end = min(start + page_size, tokens)
        k_pages[:, :, page, : end - start, :] = k[:, :, start:end, :]
        v_pages[:, :, page, : end - start, :] = v[:, :, start:end, :]
    lengths = torch.full((batch,), tokens, device=k.device, dtype=torch.long)
    return k_pages, v_pages, page_table, lengths


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Fused decode benchmarks require PyTorch") from exc
    return torch
