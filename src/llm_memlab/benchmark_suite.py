from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .benchmark import BenchmarkConfig, BenchmarkResult, benchmark_callable
from .bytes import format_bytes
from .report import make_table


@dataclass(frozen=True)
class InferenceSuiteResult:
    model_name: str
    prefill: BenchmarkResult | None = None
    decode: BenchmarkResult | None = None
    generate: BenchmarkResult | None = None
    prompt_tokens: int = 0
    new_tokens: int = 0
    peak_cuda_bytes: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def prefill_tokens_per_second(self) -> float:
        if self.prefill is None or self.prefill.mean_ms <= 0 or self.prompt_tokens <= 0:
            return 0.0
        return self.prompt_tokens * 1000 / self.prefill.mean_ms

    @property
    def decode_tokens_per_second(self) -> float:
        if self.generate is None or self.generate.mean_ms <= 0 or self.new_tokens <= 0:
            return 0.0
        return self.new_tokens * 1000 / self.generate.mean_ms

    def to_text(self) -> str:
        rows = [
            ("Model", self.model_name),
            ("Prompt tokens", self.prompt_tokens),
            ("New tokens", self.new_tokens),
            ("Prefill mean", _fmt_ms(self.prefill)),
            ("Generate mean", _fmt_ms(self.generate)),
            ("Prefill tok/s", f"{self.prefill_tokens_per_second:.2f}"),
            ("Decode tok/s", f"{self.decode_tokens_per_second:.2f}"),
            ("Peak CUDA", "n/a" if self.peak_cuda_bytes is None else format_bytes(self.peak_cuda_bytes)),
        ]
        text = [make_table(("Metric", "Value"), rows)]
        if self.notes:
            text.extend(["", "Notes"])
            text.extend(f"- {note}" for note in self.notes)
        return "\n".join(text)


def benchmark_inference_suite(
    model: Any,
    encoded: dict[str, Any],
    *,
    model_name: str = "model",
    max_new_tokens: int = 16,
    config: BenchmarkConfig | None = None,
) -> InferenceSuiteResult:
    torch = _import_torch()
    cfg = config or BenchmarkConfig(warmup=1, repeats=3)
    prompt_tokens = _prompt_len(encoded)
    notes: list[str] = []

    prefill = benchmark_callable(f"{model_name}:prefill", lambda: model(**encoded), cfg)
    generate = None
    if hasattr(model, "generate"):
        generate = benchmark_callable(
            f"{model_name}:generate",
            lambda: model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False),
            cfg,
        )
    else:
        notes.append("Model has no generate(); decode benchmark skipped.")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    return InferenceSuiteResult(
        model_name=model_name,
        prefill=prefill,
        generate=generate,
        prompt_tokens=prompt_tokens,
        new_tokens=max_new_tokens if generate is not None else 0,
        peak_cuda_bytes=peak,
        notes=tuple(notes),
    )


def compare_inference_suites(results: list[InferenceSuiteResult]) -> str:
    rows = [
        (
            result.model_name,
            f"{result.prefill.mean_ms:.3f}" if result.prefill else "n/a",
            f"{result.generate.mean_ms:.3f}" if result.generate else "n/a",
            f"{result.prefill_tokens_per_second:.2f}",
            f"{result.decode_tokens_per_second:.2f}",
            "n/a" if result.peak_cuda_bytes is None else format_bytes(result.peak_cuda_bytes),
        )
        for result in results
    ]
    return make_table(("Model", "Prefill ms", "Generate ms", "Prefill tok/s", "Decode tok/s", "Peak CUDA"), rows)


def _prompt_len(encoded: dict[str, Any]) -> int:
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
        return int(input_ids.shape[-1])
    return 0


def _fmt_ms(result: BenchmarkResult | None) -> str:
    return "n/a" if result is None else f"{result.mean_ms:.3f} ms"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Inference benchmark suite requires PyTorch. Install with: pip install torch") from exc
    return torch
