from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkResult
from .bytes import format_bytes


@dataclass(frozen=True)
class CompareReport:
    title: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    patch_report: Any = None
    baseline_trace: Any = None
    optimized_trace: Any = None
    kv_quality: Any = None
    memory_policy: Any = None
    optimization_report: Any = None
    attention_stats: tuple[Any, ...] = ()


def compare_report_to_html(report: CompareReport) -> str:
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{_e(report.title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17202a; }}
h1 {{ margin-bottom: 4px; }}
section {{ margin-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e9f0; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fb; }}
.metric-grid {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
.metric {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 12px 14px; min-width: 160px; }}
.metric b {{ display: block; font-size: 22px; }}
pre {{ background: #f6f8fb; border: 1px solid #d8dee9; border-radius: 8px; padding: 12px; overflow: auto; }}
code {{ white-space: pre-wrap; }}
.good {{ color: #047857; font-weight: 600; }}
.warn {{ color: #b45309; font-weight: 600; }}
</style>
</head>
<body>
<h1>{_e(report.title)}</h1>
<p>Baseline vs optimized run summary for speed, patch coverage, layer trace, and KV cache quality.</p>
{_benchmark_html(report.benchmarks)}
{_optimization_html(report.optimization_report)}
{_memory_policy_html(report.memory_policy)}
{_patch_html(report.patch_report)}
{_trace_compare_html(report.baseline_trace, report.optimized_trace)}
{_kv_quality_html(report.kv_quality)}
{_attention_stats_html(report.attention_stats)}
</body>
</html>"""


def write_compare_html(report: CompareReport, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(compare_report_to_html(report), encoding="utf-8")
    return output


def _benchmark_html(results: list[BenchmarkResult]) -> str:
    if not results:
        return ""
    rows = []
    base = results[0].mean_ms if results and results[0].mean_ms else 0.0
    for result in results:
        speedup = (base / result.mean_ms) if result.mean_ms else 0.0
        rows.append(
            f"<tr><td>{_e(result.name)}</td><td>{result.mean_ms:.3f}</td><td>{result.min_ms:.3f}</td><td>{result.max_ms:.3f}</td><td>{speedup:.2f}x</td><td>{_e(result.output_shape)}</td><td>{_e(_fmt_peak(result.peak_cuda_bytes))}</td></tr>"
        )
    return (
        """
<section>
<h2>Benchmark</h2>
<table><thead><tr><th>Name</th><th>Mean ms</th><th>Min ms</th><th>Max ms</th><th>Speed vs baseline</th><th>Output</th><th>Peak CUDA</th></tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _patch_html(patch_report: Any) -> str:
    if patch_report is None:
        return ""
    metrics = [
        ("RMSNorm", getattr(patch_report, "patched_norms", 0)),
        ("SwiGLU MLP", getattr(patch_report, "patched_mlps", 0)),
        ("Attention", getattr(patch_report, "patched_attentions", 0)),
        ("Skipped", len(getattr(patch_report, "skipped", []))),
    ]
    cards = "".join(f"<div class='metric'>{_e(name)}<b>{value}</b></div>" for name, value in metrics)
    details = _e(patch_report.to_text() if hasattr(patch_report, "to_text") else str(patch_report))
    return f"<section><h2>Patch report</h2><div class='metric-grid'>{cards}</div><pre>{details}</pre></section>"


def _trace_compare_html(baseline_trace: Any, optimized_trace: Any) -> str:
    if baseline_trace is None and optimized_trace is None:
        return ""
    rows = []
    for label, trace in (("baseline", baseline_trace), ("optimized", optimized_trace)):
        if trace is None:
            continue
        records = list(getattr(trace, "records", []))
        rows.append(
            f"<tr><td>{label}</td><td>{getattr(trace, 'total_ms', 0.0):.3f}</td><td>{len(records)}</td><td>{format_bytes(sum(getattr(r, 'output_bytes', 0) for r in records))}</td><td>{_e(_hot_layers(trace))}</td></tr>"
        )
    return (
        "<section><h2>Trace summary</h2><table><thead><tr><th>Run</th><th>Total ms</th><th>Layers</th><th>Output bytes</th><th>Hot layers</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )


def _kv_quality_html(kv_quality: Any) -> str:
    if kv_quality is None:
        return ""
    details = _e(kv_quality.to_text() if hasattr(kv_quality, "to_text") else str(kv_quality))
    ratio = getattr(kv_quality, "compression_ratio", None)
    ratio_card = f"<div class='metric'>KV compression<b>{ratio:.2f}x</b></div>" if ratio is not None else ""
    return f"<section><h2>KV quality</h2><div class='metric-grid'>{ratio_card}</div><pre>{details}</pre></section>"


def _hot_layers(trace: Any) -> str:
    if not hasattr(trace, "slowest"):
        return "n/a"
    return ", ".join(f"{record.name} {record.elapsed_ms:.3f} ms" for record in trace.slowest(3))


def _fmt_peak(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(value)


def _e(value: Any) -> str:
    return html.escape(str(value))


def _optimization_html(optimization_report: Any) -> str:
    if optimization_report is None:
        return ""
    findings = getattr(optimization_report, "findings", ())
    if not findings:
        try:
            from .optimization_report import infer_findings

            findings = infer_findings(optimization_report)
        except Exception:
            findings = ()
    rows = "".join(f"<tr><td>{_e(item.area)}</td><td>{_e(item.finding)}</td><td>{_e(item.impact)}</td></tr>" for item in findings)
    speed = optimization_report.speedup() if hasattr(optimization_report, "speedup") else None
    speed_card = f"<div class='metric'>Speed vs baseline<b>{speed:.2f}x</b></div>" if speed is not None else ""
    return f"<section><h2>Optimization summary</h2><div class='metric-grid'>{speed_card}</div><table><thead><tr><th>Area</th><th>Finding</th><th>Impact</th></tr></thead><tbody>{rows}</tbody></table></section>"


def _memory_policy_html(memory_policy: Any) -> str:
    if memory_policy is None:
        return ""
    details = _e(memory_policy.to_text() if hasattr(memory_policy, "to_text") else str(memory_policy))
    return f"<section><h2>Memory policy</h2><pre>{details}</pre></section>"


def _attention_stats_html(attention_stats: tuple[Any, ...]) -> str:
    if not attention_stats:
        return ""
    rows = "".join(
        f"<tr><td>{_e(item.name)}</td><td>{item.entropy:.4f}</td><td>{item.max_probability:.4f}</td><td>{item.dead_head_fraction:.1%}</td><td>{_e(item.shape)}</td></tr>"
        for item in attention_stats
    )
    return (
        "<section><h2>Attention debugger</h2><table><thead><tr><th>Layer</th><th>Entropy</th><th>Max prob</th><th>Dead heads</th><th>Shape</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></section>"
    )


def scoreboard_to_html(rows: list[dict[str, Any]], *, title: str = "llm-memlab scoreboard") -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{_e(row.get('model', ''))}</td>"
            f"<td>{_e(row.get('status', ''))}</td>"
            f"<td>{_fmt_num(row.get('baseline_ms'))}</td>"
            f"<td>{_fmt_num(row.get('optimized_ms'))}</td>"
            f"<td>{_fmt_speed(row.get('speedup'))}</td>"
            f"<td>{_e(row.get('patched', ''))}</td>"
            f"<td>{_e(row.get('params', ''))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{_e(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin:24px; color:#17202a; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ border-bottom:1px solid #e5e9f0; padding:8px; text-align:left; }}
th {{ background:#f6f8fb; }}
</style>
</head>
<body>
<h1>{_e(title)}</h1>
<table><thead><tr><th>Model</th><th>Status</th><th>Baseline ms</th><th>Optimized ms</th><th>Speed</th><th>Patched</th><th>Params</th></tr></thead><tbody>{"".join(body)}</tbody></table>
</body>
</html>"""


def write_scoreboard_html(rows: list[dict[str, Any]], path: str | Path, *, title: str = "llm-memlab scoreboard") -> Path:
    output = Path(path)
    output.write_text(scoreboard_to_html(rows, title=title), encoding="utf-8")
    return output


def _fmt_num(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return _e(value)


def _fmt_speed(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}x"
    except Exception:
        return _e(value)
