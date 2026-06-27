from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkResult
from .benchmark_suite import InferenceSuiteResult


@dataclass(frozen=True)
class BenchmarkRecord:
    name: str
    kind: str
    mean_ms: float
    min_ms: float
    max_ms: float
    peak_cuda_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def record_from_benchmark(result: BenchmarkResult, *, kind: str = "callable") -> BenchmarkRecord:
    return BenchmarkRecord(result.name, kind, result.mean_ms, result.min_ms, result.max_ms, result.peak_cuda_bytes, dict(result.extra))


def records_from_suite(result: InferenceSuiteResult) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    if result.prefill is not None:
        records.append(record_from_benchmark(result.prefill, kind="prefill"))
    if result.generate is not None:
        rec = record_from_benchmark(result.generate, kind="generate")
        rec.extra.update({"prefill_tok_s": result.prefill_tokens_per_second, "decode_tok_s": result.decode_tokens_per_second})
        records.append(rec)
    return records


def write_benchmark_json(records: list[BenchmarkRecord], path: str | Path) -> Path:
    output = Path(path)
    output.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    return output


def read_benchmark_json(path: str | Path) -> list[BenchmarkRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkRecord(**item) for item in data]


def write_benchmark_csv(records: list[BenchmarkRecord], path: str | Path) -> Path:
    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "kind", "mean_ms", "min_ms", "max_ms", "peak_cuda_bytes", "extra"])
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["extra"] = json.dumps(row["extra"], sort_keys=True)
            writer.writerow(row)
    return output
