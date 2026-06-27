from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .benchmark import BenchmarkResult, compare_benchmarks
from .bytes import format_bytes
from .report import make_table


@dataclass(frozen=True)
class OptimizationFinding:
    area: str
    finding: str
    impact: str


@dataclass(frozen=True)
class OptimizationReport:
    title: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    patch_report: Any = None
    memory_policy: Any = None
    baseline_trace: Any = None
    optimized_trace: Any = None
    kv_quality: Any = None
    attention_stats: tuple[Any, ...] = ()
    findings: tuple[OptimizationFinding, ...] = ()

    def speedup(self) -> float | None:
        if len(self.benchmarks) < 2 or self.benchmarks[1].mean_ms <= 0:
            return None
        return self.benchmarks[0].mean_ms / self.benchmarks[1].mean_ms

    def to_text(self) -> str:
        parts = [self.title]
        if self.benchmarks:
            parts.extend(["", "Benchmark", compare_benchmarks(self.benchmarks)])
        speedup = self.speedup()
        summary_rows = []
        if speedup is not None:
            summary_rows.append(("Speed vs baseline", f"{speedup:.2f}x"))
        if self.kv_quality is not None and hasattr(self.kv_quality, "compression_ratio"):
            summary_rows.append(("KV compression", f"{self.kv_quality.compression_ratio:.2f}x"))
        if self.patch_report is not None:
            summary_rows.append(("Total patched", getattr(self.patch_report, "total_patched", "n/a")))
        if self.baseline_trace is not None:
            summary_rows.append(("Baseline trace", _trace_summary(self.baseline_trace)))
        if self.optimized_trace is not None:
            summary_rows.append(("Optimized trace", _trace_summary(self.optimized_trace)))
        if summary_rows:
            parts.extend(["", "Optimization summary", make_table(("Metric", "Value"), summary_rows)])
        if self.memory_policy is not None:
            parts.extend(["", "Memory policy", self.memory_policy.to_text() if hasattr(self.memory_policy, "to_text") else str(self.memory_policy)])
        if self.patch_report is not None:
            parts.extend(["", "Patch report", self.patch_report.to_text() if hasattr(self.patch_report, "to_text") else str(self.patch_report)])
        if self.kv_quality is not None:
            parts.extend(["", "KV quality", self.kv_quality.to_text() if hasattr(self.kv_quality, "to_text") else str(self.kv_quality)])
        if self.attention_stats:
            rows = [(item.name, f"{item.entropy:.4f}", f"{item.max_probability:.4f}", f"{item.dead_head_fraction:.1%}") for item in self.attention_stats]
            parts.extend(["", "Attention stats", make_table(("Layer", "Entropy", "Max prob", "Dead heads"), rows)])
        findings = self.findings or infer_findings(self)
        if findings:
            rows = [(item.area, item.finding, item.impact) for item in findings]
            parts.extend(["", "Findings", make_table(("Area", "Finding", "Impact"), rows)])
        return "\n".join(parts)


def infer_findings(report: OptimizationReport) -> tuple[OptimizationFinding, ...]:
    findings: list[OptimizationFinding] = []
    if report.patch_report is not None:
        skipped = getattr(report.patch_report, "skipped", [])
        patched = getattr(report.patch_report, "total_patched", 0)
        findings.append(OptimizationFinding("patch", f"patched {patched} modules", "kernel/runtime coverage"))
        if skipped:
            findings.append(OptimizationFinding("patch", f"{len(skipped)} modules skipped", "check unsupported layouts before expecting speedup"))
    if report.kv_quality is not None and hasattr(report.kv_quality, "compression_ratio"):
        findings.append(OptimizationFinding("kv-cache", f"compression {report.kv_quality.compression_ratio:.2f}x", "memory reduction with measured output drift"))
    speedup = report.speedup()
    if speedup is not None:
        verdict = "faster" if speedup >= 1 else "slower"
        findings.append(OptimizationFinding("speed", f"optimized path is {speedup:.2f}x vs baseline", verdict))
    if report.baseline_trace is not None:
        hot = report.baseline_trace.slowest(1)
        if hot:
            findings.append(OptimizationFinding("debug", f"hot layer: {hot[0].name}", f"{hot[0].elapsed_ms:.3f} ms, {format_bytes(hot[0].output_bytes)} output"))
    return tuple(findings)


def _trace_summary(trace: Any) -> str:
    records = list(getattr(trace, "records", []))
    output_bytes = sum(getattr(record, "output_bytes", 0) for record in records)
    return f"{getattr(trace, 'total_ms', 0.0):.3f} ms, {len(records)} layers, {format_bytes(output_bytes)} outputs"
