from __future__ import annotations

import csv
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkResult
from .benchmark_suite import InferenceSuiteResult

BENCHMARK_SCHEMA_VERSION = "llm_memlab.benchmark.v2"


@dataclass(frozen=True)
class BenchmarkMetadata:
    python: str
    platform: str
    torch: str | None = None
    torch_cuda: str | None = None
    triton: str | None = None
    gpu: str | None = None
    gpu_capability: str | None = None
    dtype: str | None = None
    sequence_length: int | None = None
    commit: str | None = None
    warmup: int | None = None
    repeats: int | None = None
    seed: int | None = None
    backend: str | None = None
    schema_version: str = BENCHMARK_SCHEMA_VERSION


@dataclass(frozen=True)
class BenchmarkRecord:
    name: str
    kind: str
    mean_ms: float
    min_ms: float
    max_ms: float
    peak_cuda_bytes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = BENCHMARK_SCHEMA_VERSION


@dataclass(frozen=True)
class BenchmarkGateConfig:
    max_slowdown_pct: float = 10.0
    require_quality_passed: bool = True
    max_quality_mean_abs: float | None = None
    min_speedup: float | None = None


@dataclass(frozen=True)
class BenchmarkComparison:
    baseline: BenchmarkRecord
    candidate: BenchmarkRecord
    speedup: float
    delta_pct: float
    passed: bool
    threshold_pct: float
    quality_passed: bool = True
    reasons: tuple[str, ...] = ()

    def to_text(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        reason = "" if not self.reasons else " reason=" + "; ".join(self.reasons)
        return f"{self.candidate.name}: {self.candidate.mean_ms:.3f} ms vs {self.baseline.mean_ms:.3f} ms, speedup={self.speedup:.3f}x, delta={self.delta_pct:.1f}%, quality={self.quality_passed}, {status}{reason}"


@dataclass(frozen=True)
class BenchmarkGateResult:
    comparisons: tuple[BenchmarkComparison, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    def to_text(self) -> str:
        if not self.comparisons:
            return "No matching benchmark records found."
        return "\n".join(item.to_text() for item in self.comparisons)


@dataclass(frozen=True)
class BenchmarkHistory:
    records: tuple[BenchmarkRecord, ...]
    source_count: int

    def baseline_records(self) -> list[BenchmarkRecord]:
        grouped: dict[tuple[str, str], list[BenchmarkRecord]] = {}
        for record in self.records:
            grouped.setdefault((record.name, record.kind), []).append(record)
        baselines: list[BenchmarkRecord] = []
        for items in grouped.values():
            items = sorted(items, key=lambda item: item.mean_ms)
            mid = len(items) // 2
            baselines.append(items[mid])
        return baselines

    def to_text(self) -> str:
        return make_history_table(self.baseline_records(), self.source_count)


def collect_run_metadata(
    *,
    dtype: str | None = None,
    sequence_length: int | None = None,
    warmup: int | None = None,
    repeats: int | None = None,
    seed: int | None = None,
    backend: str | None = None,
) -> BenchmarkMetadata:
    torch_version = None
    torch_cuda = None
    gpu = None
    gpu_capability = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            gpu_capability = ".".join(str(item) for item in torch.cuda.get_device_capability(0))
    except Exception:
        pass
    triton_version = None
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        pass
    return BenchmarkMetadata(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch_version,
        torch_cuda=torch_cuda,
        triton=triton_version,
        gpu=gpu,
        gpu_capability=gpu_capability,
        dtype=dtype,
        sequence_length=sequence_length,
        commit=_git_commit(),
        warmup=warmup,
        repeats=repeats,
        seed=seed,
        backend=backend,
    )


def record_from_benchmark(
    result: BenchmarkResult, *, kind: str = "callable", metadata: BenchmarkMetadata | dict[str, Any] | None = None
) -> BenchmarkRecord:
    return BenchmarkRecord(
        result.name,
        kind,
        result.mean_ms,
        result.min_ms,
        result.max_ms,
        result.peak_cuda_bytes,
        dict(result.extra),
        _metadata_dict(metadata),
    )


def records_from_suite(
    result: InferenceSuiteResult, *, metadata: BenchmarkMetadata | dict[str, Any] | None = None
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    if result.prefill is not None:
        records.append(record_from_benchmark(result.prefill, kind="prefill", metadata=metadata))
    if result.decode is not None:
        records.append(record_from_benchmark(result.decode, kind="decode", metadata=metadata))
    if result.generate is not None:
        rec = record_from_benchmark(result.generate, kind="generate", metadata=metadata)
        rec.extra.update({"prefill_tok_s": result.prefill_tokens_per_second, "decode_tok_s": result.decode_tokens_per_second})
        records.append(rec)
    return records


def compare_benchmark_records(
    baseline: BenchmarkRecord, candidate: BenchmarkRecord, *, max_slowdown_pct: float = 10.0, gate: BenchmarkGateConfig | None = None
) -> BenchmarkComparison:
    cfg = gate or BenchmarkGateConfig(max_slowdown_pct=max_slowdown_pct)
    speedup = baseline.mean_ms / candidate.mean_ms if candidate.mean_ms > 0 else 0.0
    delta_pct = ((candidate.mean_ms - baseline.mean_ms) / baseline.mean_ms * 100.0) if baseline.mean_ms > 0 else 0.0
    reasons: list[str] = []
    if delta_pct > cfg.max_slowdown_pct:
        reasons.append(f"slowdown {delta_pct:.1f}% > {cfg.max_slowdown_pct:.1f}%")
    if cfg.min_speedup is not None and speedup < cfg.min_speedup:
        reasons.append(f"speedup {speedup:.3f}x < {cfg.min_speedup:.3f}x")
    quality_passed = _quality_passed(candidate, cfg)
    if not quality_passed:
        reasons.append("quality threshold failed")
    return BenchmarkComparison(baseline, candidate, speedup, delta_pct, not reasons, cfg.max_slowdown_pct, quality_passed, tuple(reasons))


def compare_record_sets(
    baseline: list[BenchmarkRecord],
    candidate: list[BenchmarkRecord],
    *,
    max_slowdown_pct: float = 10.0,
    gate: BenchmarkGateConfig | None = None,
) -> list[BenchmarkComparison]:
    cfg = gate or BenchmarkGateConfig(max_slowdown_pct=max_slowdown_pct)
    by_key = {(record.name, record.kind): record for record in baseline}
    comparisons: list[BenchmarkComparison] = []
    for record in candidate:
        base = by_key.get((record.name, record.kind))
        if base is not None:
            comparisons.append(compare_benchmark_records(base, record, gate=cfg))
    return comparisons


def benchmark_gate(
    baseline: list[BenchmarkRecord], candidate: list[BenchmarkRecord], config: BenchmarkGateConfig | None = None
) -> BenchmarkGateResult:
    cfg = config or BenchmarkGateConfig()
    return BenchmarkGateResult(tuple(compare_record_sets(baseline, candidate, gate=cfg)))


def read_benchmark_files(paths: list[str | Path] | tuple[str | Path, ...]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            records.extend(read_benchmark_csv(path))
        else:
            records.extend(read_benchmark_json(path))
    return records


def benchmark_history(paths: list[str | Path] | tuple[str | Path, ...]) -> BenchmarkHistory:
    return BenchmarkHistory(tuple(read_benchmark_files(paths)), len(paths))


def benchmark_history_gate(
    baseline_paths: list[str | Path] | tuple[str | Path, ...],
    candidate_paths: list[str | Path] | tuple[str | Path, ...],
    config: BenchmarkGateConfig | None = None,
) -> BenchmarkGateResult:
    history = benchmark_history(baseline_paths)
    candidates = read_benchmark_files(candidate_paths)
    return benchmark_gate(history.baseline_records(), candidates, config)


def assert_no_regressions(comparisons: list[BenchmarkComparison] | BenchmarkGateResult) -> None:
    items = list(comparisons.comparisons) if isinstance(comparisons, BenchmarkGateResult) else list(comparisons)
    failures = [item.to_text() for item in items if not item.passed]
    if failures:
        raise AssertionError("Benchmark regression threshold exceeded:\n" + "\n".join(failures))


def write_benchmark_json(records: list[BenchmarkRecord], path: str | Path) -> Path:
    output = Path(path)
    output.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")
    return output


def make_history_table(records: list[BenchmarkRecord], source_count: int) -> str:
    rows = [
        (
            record.name,
            record.kind,
            f"{record.mean_ms:.3f}",
            record.metadata.get("gpu", "n/a"),
            record.metadata.get("commit", "n/a"),
        )
        for record in records
    ]
    from .report import make_table

    return make_table(("Name", "Kind", "Median ms", "GPU", "Commit"), [("Sources", source_count, "", "", ""), *rows])


def read_benchmark_json(path: str | Path) -> list[BenchmarkRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("records", [])
    return [_record_from_dict(item) for item in data]


def write_benchmark_csv(records: list[BenchmarkRecord], path: str | Path) -> Path:
    output = Path(path)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["schema_version", "name", "kind", "mean_ms", "min_ms", "max_ms", "peak_cuda_bytes", "extra", "metadata"]
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["extra"] = json.dumps(row["extra"], sort_keys=True)
            row["metadata"] = json.dumps(row["metadata"], sort_keys=True)
            writer.writerow(row)
    return output


def read_benchmark_csv(path: str | Path) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                BenchmarkRecord(
                    name=row["name"],
                    kind=row["kind"],
                    mean_ms=float(row["mean_ms"]),
                    min_ms=float(row["min_ms"]),
                    max_ms=float(row["max_ms"]),
                    peak_cuda_bytes=int(row["peak_cuda_bytes"]) if row.get("peak_cuda_bytes") else None,
                    extra=json.loads(row.get("extra") or "{}"),
                    metadata=json.loads(row.get("metadata") or "{}"),
                    schema_version=row.get("schema_version") or BENCHMARK_SCHEMA_VERSION,
                )
            )
    return records


def _quality_passed(candidate: BenchmarkRecord, cfg: BenchmarkGateConfig) -> bool:
    extra = candidate.extra or {}
    if cfg.require_quality_passed and extra.get("quality_passed") is False:
        return False
    if cfg.max_quality_mean_abs is not None and extra.get("mean_abs") is not None:
        try:
            if float(extra["mean_abs"]) > cfg.max_quality_mean_abs:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _record_from_dict(item: dict[str, Any]) -> BenchmarkRecord:
    payload = dict(item)
    payload.setdefault("schema_version", BENCHMARK_SCHEMA_VERSION)
    payload.setdefault("extra", {})
    payload.setdefault("metadata", {})
    return BenchmarkRecord(**payload)


def _metadata_dict(metadata: BenchmarkMetadata | dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, BenchmarkMetadata):
        return asdict(metadata)
    payload = dict(metadata)
    payload.setdefault("schema_version", BENCHMARK_SCHEMA_VERSION)
    return payload


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, capture_output=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None
