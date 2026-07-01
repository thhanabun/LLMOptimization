from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_dashboard import BenchmarkDashboard
from .benchmark_store import BenchmarkRecord, collect_run_metadata, write_benchmark_csv, write_benchmark_json
from .bytes import format_bytes
from .hf_runtime import HFMemoryFirstBenchmark, benchmark_memory_first_hf_generate
from .report import make_table

SERVING_BENCHMARK_SCHEMA_VERSION = "llm_memlab.serving_benchmark.v1"


@dataclass(frozen=True)
class ServingRun:
    name: str
    backend: str
    available: bool
    elapsed_ms: float | None = None
    first_token_ms: float | None = None
    tokens_per_second: float | None = None
    new_tokens: int = 0
    peak_cuda_bytes: int | None = None
    text_preview: str = ""
    token_match: bool | None = None
    cache_hit: bool | None = None
    prefix_cache: bool | None = None
    fallback_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingBenchmarkResult:
    model: str
    prompt: str
    runs: tuple[ServingRun, ...]
    hf_benchmark: HFMemoryFirstBenchmark | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SERVING_BENCHMARK_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return all(run.token_match is not False for run in self.runs if run.available)

    def to_records(self) -> list[BenchmarkRecord]:
        records: list[BenchmarkRecord] = []
        for run in self.runs:
            mean_ms = float(run.elapsed_ms or 0.0)
            metadata = dict(self.metadata)
            metadata.update(run.metadata)
            metadata["backend"] = run.backend
            records.append(
                BenchmarkRecord(
                    name=f"serving/{run.name}",
                    kind="serving-generate",
                    mean_ms=mean_ms,
                    min_ms=mean_ms,
                    max_ms=mean_ms,
                    peak_cuda_bytes=run.peak_cuda_bytes,
                    extra={
                        "schema_version": self.schema_version,
                        "model": self.model,
                        "path": run.name,
                        "backend_selected": run.backend,
                        "available": run.available,
                        "first_token_ms": run.first_token_ms,
                        "tokens_per_second": run.tokens_per_second,
                        "new_tokens": run.new_tokens,
                        "cache_hit": run.cache_hit,
                        "prefix_cache": run.prefix_cache,
                        "token_match": run.token_match,
                        "quality_passed": run.token_match if run.token_match is not None else None,
                        "fallback_reason": run.fallback_reason,
                        "text_preview": run.text_preview,
                    },
                    metadata=metadata,
                )
            )
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "prompt": self.prompt,
            "runs": [asdict(run) for run in self.runs],
            "metadata": self.metadata,
            "passed": self.passed,
            "hf_benchmark": self.hf_benchmark.to_dict() if self.hf_benchmark else None,
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        payload = self.to_dict()
        payload["records"] = [asdict(record) for record in self.to_records()]
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def write_records_json(self, path: str | Path) -> Path:
        return write_benchmark_json(self.to_records(), path)

    def write_csv(self, path: str | Path) -> Path:
        return write_benchmark_csv(self.to_records(), path)

    def write_html(self, path: str | Path, *, title: str = "llm-memlab serving dashboard") -> Path:
        output = Path(path)
        output.write_text(BenchmarkDashboard(tuple(self.to_records())).to_html(title=title), encoding="utf-8")
        return output

    def to_text(self) -> str:
        rows = [
            (
                run.name,
                run.backend,
                run.available,
                _fmt_ms(run.elapsed_ms),
                _fmt_ms(run.first_token_ms),
                _fmt_float(run.tokens_per_second),
                format_bytes(run.peak_cuda_bytes) if run.peak_cuda_bytes is not None else "n/a",
                run.cache_hit if run.cache_hit is not None else "n/a",
                run.prefix_cache if run.prefix_cache is not None else "n/a",
                run.token_match if run.token_match is not None else "n/a",
                run.fallback_reason or "",
            )
            for run in self.runs
        ]
        return make_table(
            (
                "Path",
                "Backend",
                "Available",
                "Latency",
                "First token",
                "Tok/s",
                "Peak",
                "Cache hit",
                "Prefix cache",
                "Token match",
                "Fallback",
            ),
            rows,
        )


def benchmark_serving_paths(
    model: str,
    *,
    prompt: str = "Hello",
    max_new_tokens: int = 8,
    adapter_tokens: int | None = None,
    device: str | None = None,
    dtype: str = "auto",
    local_files_only: bool = False,
    cache: str = "paged",
    quant_dtype: str = "int8",
    include_vllm: bool = True,
    allow_experimental_direct_cache: bool = False,
) -> ServingBenchmarkResult:
    hf = benchmark_memory_first_hf_generate(
        model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        adapter_tokens=adapter_tokens,
        device=device,
        dtype=dtype,
        local_files_only=local_files_only,
        cache=cache,
        quant_dtype=quant_dtype,
        allow_experimental_direct_cache=allow_experimental_direct_cache,
    )
    metadata = dict(hf.metadata)
    metadata.update(collect_run_metadata(dtype=dtype, backend="serving-benchmark").__dict__)
    runs = [
        _run_from_hf("hf-generate", "hf", hf.baseline, token_match=True),
        _run_from_hf("memory-first-hf", "llm-memlab", hf.memory_first, token_match=hf.passed),
    ]
    if include_vllm:
        runs.append(_run_vllm_or_fallback(model, prompt, max_new_tokens=max_new_tokens, dtype=dtype))
    return ServingBenchmarkResult(model=model, prompt=prompt, runs=tuple(runs), hf_benchmark=hf, metadata=metadata)


def _run_from_hf(name: str, backend: str, run, *, token_match: bool) -> ServingRun:
    return ServingRun(
        name=name,
        backend=backend,
        available=True,
        elapsed_ms=run.elapsed_ms,
        first_token_ms=run.elapsed_ms / max(1, run.new_tokens),
        tokens_per_second=run.new_tokens / max(run.elapsed_ms / 1000.0, 1e-9),
        new_tokens=run.new_tokens,
        peak_cuda_bytes=run.peak_cuda_bytes,
        text_preview=run.text_preview,
        token_match=token_match,
        cache_hit=None,
        prefix_cache=None,
    )


def _run_vllm_or_fallback(model: str, prompt: str, *, max_new_tokens: int, dtype: str) -> ServingRun:
    from .backends.vllm import detect_vllm_runtime, run_vllm_generate

    info = detect_vllm_runtime()
    if not info.available:
        return ServingRun(
            name="vllm",
            backend="vllm-serving",
            available=False,
            fallback_reason=info.fallback_reason,
            metadata={"vllm_version": info.version, "platform": info.platform},
        )
    try:
        result = run_vllm_generate(model, prompt, max_new_tokens=max_new_tokens, dtype=dtype, enable_prefix_caching=True)
    except Exception as exc:  # pragma: no cover - vLLM is optional and hardware-specific.
        return ServingRun(
            name="vllm",
            backend="vllm-serving",
            available=False,
            fallback_reason=f"vLLM runtime failed; fallback to HF paths: {exc}",
            metadata={"vllm_version": info.version, "platform": info.platform},
        )
    return ServingRun(
        name="vllm",
        backend="vllm-serving",
        available=True,
        elapsed_ms=result.elapsed_ms,
        first_token_ms=result.first_token_ms,
        tokens_per_second=result.tokens_per_second,
        new_tokens=result.new_tokens,
        peak_cuda_bytes=result.peak_cuda_bytes,
        text_preview=result.text.replace("\n", " ")[:300],
        token_match=None,
        cache_hit=None,
        prefix_cache=result.prefix_cache_enabled,
        metadata={"vllm_version": result.info.version, "platform": result.info.platform},
    )


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} ms"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
