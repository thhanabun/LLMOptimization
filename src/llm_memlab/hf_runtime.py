from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_store import BenchmarkRecord, collect_run_metadata, write_benchmark_csv, write_benchmark_json
from .hf_adapter import (
    MemoryFirstHFConfig,
    MemoryFirstGenerateResult,
    detect_hf_adapter_info,
    memory_first_generate_hf,
    select_memory_adapter,
)
from .quality_metrics import TokenQualityResult, compare_token_sequences
from .report import make_table

HF_RUNTIME_SCHEMA_VERSION = "llm_memlab.hf_runtime.v1"


@dataclass(frozen=True)
class HFGenerateRun:
    name: str
    elapsed_ms: float
    new_tokens: int
    peak_cuda_bytes: int | None
    allocated_cuda_bytes: int | None
    text_preview: str = ""


@dataclass(frozen=True)
class HFMemoryFirstBenchmark:
    model: str
    family: str
    adapter: str
    baseline: HFGenerateRun
    memory_first: HFGenerateRun
    token_quality: TokenQualityResult
    adapter_result: MemoryFirstGenerateResult
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = HF_RUNTIME_SCHEMA_VERSION

    @property
    def speedup(self) -> float:
        return self.baseline.elapsed_ms / self.memory_first.elapsed_ms if self.memory_first.elapsed_ms > 0 else 0.0

    @property
    def peak_delta_bytes(self) -> int | None:
        if self.baseline.peak_cuda_bytes is None or self.memory_first.peak_cuda_bytes is None:
            return None
        return self.memory_first.peak_cuda_bytes - self.baseline.peak_cuda_bytes

    @property
    def passed(self) -> bool:
        return bool(self.token_quality.exact_match or self.token_quality.token_agreement >= 1.0)

    def to_records(self) -> list[BenchmarkRecord]:
        common = {
            "schema_version": self.schema_version,
            "model": self.model,
            "family": self.family,
            "adapter": self.adapter,
            "adapter_cache_impl": self.adapter_result.cache_impl,
            "direct_cache": self.adapter_result.direct_cache,
            "fallback_reason": self.adapter_result.fallback_reason,
            "token_agreement": self.token_quality.token_agreement,
            "exact_match": self.token_quality.exact_match,
            "speedup": self.speedup,
            "peak_delta_bytes": self.peak_delta_bytes,
        }
        return [
            BenchmarkRecord(
                name=f"hf/{self.family}/baseline-generate",
                kind="generate",
                mean_ms=self.baseline.elapsed_ms,
                min_ms=self.baseline.elapsed_ms,
                max_ms=self.baseline.elapsed_ms,
                peak_cuda_bytes=self.baseline.peak_cuda_bytes,
                extra={**common, "path": "baseline", "new_tokens": self.baseline.new_tokens},
                metadata=self.metadata,
            ),
            BenchmarkRecord(
                name=f"hf/{self.family}/memory-first-generate",
                kind="generate",
                mean_ms=self.memory_first.elapsed_ms,
                min_ms=self.memory_first.elapsed_ms,
                max_ms=self.memory_first.elapsed_ms,
                peak_cuda_bytes=self.memory_first.peak_cuda_bytes,
                extra={**common, "path": "memory-first", "new_tokens": self.memory_first.new_tokens},
                metadata=self.metadata,
            ),
        ]

    def write_json(self, path: str | Path) -> Path:
        return write_benchmark_json(self.to_records(), path)

    def write_csv(self, path: str | Path) -> Path:
        return write_benchmark_csv(self.to_records(), path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "family": self.family,
            "adapter": self.adapter,
            "baseline": asdict(self.baseline),
            "memory_first": asdict(self.memory_first),
            "speedup": self.speedup,
            "peak_delta_bytes": self.peak_delta_bytes,
            "token_quality": asdict(self.token_quality),
            "adapter_cache_impl": self.adapter_result.cache_impl,
            "direct_cache": self.adapter_result.direct_cache,
            "fallback_reason": self.adapter_result.fallback_reason,
            "metadata": self.metadata,
            "passed": self.passed,
        }

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Schema", self.schema_version),
                ("Model", self.model),
                ("Family", self.family),
                ("Adapter", self.adapter),
                ("Adapter cache impl", self.adapter_result.cache_impl),
                ("Direct cache", self.adapter_result.direct_cache),
                ("Fallback reason", self.adapter_result.fallback_reason or "n/a"),
                ("Baseline", f"{self.baseline.elapsed_ms:.3f} ms"),
                ("Memory-first", f"{self.memory_first.elapsed_ms:.3f} ms"),
                ("Speedup", f"{self.speedup:.3f}x"),
                ("Baseline peak", _fmt_bytes(self.baseline.peak_cuda_bytes)),
                ("Memory-first peak", _fmt_bytes(self.memory_first.peak_cuda_bytes)),
                ("Peak delta", _fmt_bytes(self.peak_delta_bytes)),
                ("Token agreement", f"{self.token_quality.token_agreement:.1%}"),
                ("Exact match", self.token_quality.exact_match),
                ("Passed", self.passed),
            ],
        )


def benchmark_memory_first_hf_generate(
    model_id: str,
    *,
    prompt: str = "Hello",
    max_new_tokens: int = 8,
    adapter_tokens: int | None = None,
    device: str | None = None,
    dtype: str = "auto",
    local_files_only: bool = False,
    cache: str = "quantized",
    quant_dtype: str = "int8",
    allow_experimental_direct_cache: bool = False,
) -> HFMemoryFirstBenchmark:
    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    device = _resolve_device(torch, device)
    torch_dtype = _resolve_dtype(torch, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    load_kwargs: dict[str, Any] = {"local_files_only": local_files_only, "low_cpu_mem_usage": True}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if device == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    if device:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    adapter = select_memory_adapter(
        model,
        MemoryFirstHFConfig(
            cache=cache,
            quant_dtype=quant_dtype,
            max_new_tokens=adapter_tokens or max_new_tokens,
            allow_experimental_direct_cache=allow_experimental_direct_cache,
        ),
    )
    family = detect_hf_adapter_info(model).family
    metadata = collect_run_metadata(
        dtype=str(torch_dtype or dtype), sequence_length=int(encoded["input_ids"].shape[-1]), backend="hf-generate"
    ).__dict__
    metadata.update(
        {
            "model": model_id,
            "device": device,
            "max_new_tokens": max_new_tokens,
            "adapter_tokens": adapter_tokens or max_new_tokens,
            "allow_experimental_direct_cache": allow_experimental_direct_cache,
        }
    )

    try:
        baseline_ids, baseline_run = _timed_generate(
            torch,
            model,
            tokenizer,
            encoded,
            name="baseline",
            max_new_tokens=max_new_tokens,
            use_memory_first=False,
        )
        adapter_result, memory_run = _timed_memory_first(
            torch,
            model,
            tokenizer,
            encoded,
            config=MemoryFirstHFConfig(
                cache=cache,
                quant_dtype=quant_dtype,
                max_new_tokens=adapter_tokens or max_new_tokens,
                allow_experimental_direct_cache=allow_experimental_direct_cache,
            ),
            max_new_tokens=adapter_tokens or max_new_tokens,
        )
        token_quality = compare_token_sequences(baseline_ids[:, : adapter_result.sequences.shape[-1]], adapter_result.sequences)
        return HFMemoryFirstBenchmark(
            model=model_id,
            family=family,
            adapter=adapter.__class__.__name__,
            baseline=baseline_run,
            memory_first=memory_run,
            token_quality=token_quality,
            adapter_result=adapter_result,
            metadata=metadata,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assert_hf_benchmark_passed(
    result: HFMemoryFirstBenchmark, *, min_token_agreement: float = 1.0, max_slowdown_pct: float | None = None
) -> None:
    reasons: list[str] = []
    if result.token_quality.token_agreement < min_token_agreement:
        reasons.append(f"token agreement {result.token_quality.token_agreement:.3f} < {min_token_agreement:.3f}")
    if max_slowdown_pct is not None and result.speedup > 0:
        slowdown_pct = (
            ((result.memory_first.elapsed_ms - result.baseline.elapsed_ms) / result.baseline.elapsed_ms * 100.0)
            if result.baseline.elapsed_ms > 0
            else 0.0
        )
        if slowdown_pct > max_slowdown_pct:
            reasons.append(f"slowdown {slowdown_pct:.1f}% > {max_slowdown_pct:.1f}%")
    if reasons:
        raise AssertionError("HF memory-first benchmark failed: " + "; ".join(reasons))


def _timed_generate(
    torch, model, tokenizer, encoded: dict[str, Any], *, name: str, max_new_tokens: int, use_memory_first: bool
) -> tuple[Any, HFGenerateRun]:
    del use_memory_first
    _reset_peak(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    _sync(torch)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return output_ids, HFGenerateRun(
        name=name,
        elapsed_ms=elapsed_ms,
        new_tokens=int(output_ids.shape[-1] - encoded["input_ids"].shape[-1]),
        peak_cuda_bytes=_peak(torch),
        allocated_cuda_bytes=_allocated(torch),
        text_preview=tokenizer.decode(output_ids[0], skip_special_tokens=True).replace("\n", " ")[:300],
    )


def _timed_memory_first(
    torch, model, tokenizer, encoded: dict[str, Any], *, config: MemoryFirstHFConfig, max_new_tokens: int
) -> tuple[MemoryFirstGenerateResult, HFGenerateRun]:
    _reset_peak(torch)
    started = time.perf_counter()
    with torch.inference_mode():
        model_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}
        result = memory_first_generate_hf(
            model, encoded["input_ids"], config, max_new_tokens=max_new_tokens, do_sample=False, **model_kwargs
        )
    _sync(torch)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, HFGenerateRun(
        name="memory-first",
        elapsed_ms=elapsed_ms,
        new_tokens=int(result.sequences.shape[-1] - encoded["input_ids"].shape[-1]),
        peak_cuda_bytes=_peak(torch),
        allocated_cuda_bytes=_allocated(torch),
        text_preview=tokenizer.decode(result.sequences[0], skip_special_tokens=True).replace("\n", " ")[:300],
    )


def _resolve_device(torch, device: str | None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(torch, dtype: str, device: str):
    value = dtype.lower()
    if value == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be auto, fp16, bf16, or fp32")


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak(torch) -> int | None:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def _allocated(torch) -> int | None:
    return int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024**3:.3f} GiB"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HF runtime benchmark requires PyTorch") from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("HF runtime benchmark requires transformers") from exc
    return AutoModelForCausalLM, AutoTokenizer
