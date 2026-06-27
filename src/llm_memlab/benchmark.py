from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .bytes import format_bytes
from .kv_cache import DecodeConfig, greedy_decode
from .report import make_table


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup: int = 3
    repeats: int = 10
    use_inference_mode: bool = True
    reset_cuda_peak: bool = True


@dataclass
class BenchmarkResult:
    name: str
    elapsed_ms: list[float] = field(default_factory=list)
    peak_cuda_bytes: int | None = None
    output_shape: str = "n/a"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mean_ms(self) -> float:
        if not self.elapsed_ms:
            return 0.0
        return sum(self.elapsed_ms) / len(self.elapsed_ms)

    @property
    def min_ms(self) -> float:
        return min(self.elapsed_ms) if self.elapsed_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.elapsed_ms) if self.elapsed_ms else 0.0

    def to_rows(self):
        rows = [
            ("Name", self.name),
            ("Repeats", len(self.elapsed_ms)),
            ("Mean", f"{self.mean_ms:.3f} ms"),
            ("Min", f"{self.min_ms:.3f} ms"),
            ("Max", f"{self.max_ms:.3f} ms"),
            ("Output", self.output_shape),
            ("Peak CUDA", _fmt_optional_bytes(self.peak_cuda_bytes)),
        ]
        rows.extend((key, value) for key, value in self.extra.items())
        return rows

    def to_text(self) -> str:
        return make_table(("Metric", "Value"), self.to_rows())


def benchmark_callable(name: str, fn: Callable[[], Any], config: BenchmarkConfig | None = None) -> BenchmarkResult:
    torch = _import_torch()
    cfg = config or BenchmarkConfig()
    for _ in range(cfg.warmup):
        _run(fn, cfg.use_inference_mode, torch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if cfg.reset_cuda_peak:
            torch.cuda.reset_peak_memory_stats()
    elapsed: list[float] = []
    output = None
    for _ in range(cfg.repeats):
        started = time.perf_counter()
        output = _run(fn, cfg.use_inference_mode, torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1000)
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return BenchmarkResult(name=name, elapsed_ms=elapsed, peak_cuda_bytes=peak, output_shape=_shape(output))


def benchmark_forward(model, input_factory: Callable[[], tuple[tuple[Any, ...], dict[str, Any]]], config: BenchmarkConfig | None = None) -> BenchmarkResult:
    def call():
        args, kwargs = input_factory()
        return model(*args, **kwargs)

    return benchmark_callable(model.__class__.__name__, call, config)


def benchmark_decode(model, input_ids, decode_config: DecodeConfig, config: BenchmarkConfig | None = None) -> BenchmarkResult:
    bench = benchmark_callable("decode", lambda: greedy_decode(model, input_ids, decode_config), config)
    result = greedy_decode(model, input_ids, decode_config)
    bench.extra["Decode tok/s"] = f"{result.tokens_per_second:.2f}"
    bench.extra["New tokens"] = len(result.steps)
    return bench


def compare_benchmarks(results: list[BenchmarkResult]) -> str:
    rows = [
        (result.name, f"{result.mean_ms:.3f}", f"{result.min_ms:.3f}", f"{result.max_ms:.3f}", _fmt_optional_bytes(result.peak_cuda_bytes), result.output_shape)
        for result in results
    ]
    return make_table(("Name", "Mean ms", "Min ms", "Max ms", "Peak CUDA", "Output"), rows)


def _run(fn, inference_mode: bool, torch):
    if inference_mode:
        with torch.inference_mode():
            return fn()
    return fn()


def _shape(value: Any) -> str:
    if hasattr(value, "shape"):
        return str(tuple(value.shape))
    if isinstance(value, dict) and "logits" in value and hasattr(value["logits"], "shape"):
        return "logits=" + str(tuple(value["logits"].shape))
    if hasattr(value, "sequences") and hasattr(value.sequences, "shape"):
        return "sequences=" + str(tuple(value.sequences.shape))
    return type(value).__name__


def _fmt_optional_bytes(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(value)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Benchmarking requires PyTorch. Install with: pip install torch") from exc
    return torch
