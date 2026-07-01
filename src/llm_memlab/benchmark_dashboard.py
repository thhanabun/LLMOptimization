from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .benchmark_store import BenchmarkRecord, read_benchmark_files
from .bytes import format_bytes

DASHBOARD_SCHEMA_VERSION = "llm_memlab.benchmark_dashboard.v2"


@dataclass(frozen=True)
class BenchmarkDashboard:
    records: tuple[BenchmarkRecord, ...]
    schema_version: str = DASHBOARD_SCHEMA_VERSION

    def to_html(self, *, title: str = "llm-memlab benchmark dashboard") -> str:
        rows = "".join(_record_row(record) for record in self.records)
        grouped = _group_summary(self.records)
        summary_rows = "".join(
            f"<tr><td>{_e(key)}</td><td>{len(items)}</td><td>{sum(r.mean_ms for r in items) / max(1, len(items)):.3f}</td><td>{_fmt_quality(items)}</td><td>{_fmt_peak(items)}</td></tr>"
            for key, items in grouped.items()
        )
        trend_rows = "".join(_trend_row(key, items) for key, items in grouped.items())
        regression_rows = "".join(_regression_row(key, items) for key, items in grouped.items())
        serving_rows = "".join(_serving_row(record) for record in self.records if record.kind.startswith("serving"))
        serving_section = (
            "<h2>Serving metrics</h2><table><thead><tr><th>Name</th><th>Backend selected</th><th>Available</th><th>First token ms</th><th>Tok/s</th><th>Cache hit</th><th>Prefix cache</th><th>Token match</th><th>Fallback</th></tr></thead><tbody>"
            + serving_rows
            + "</tbody></table>"
            if serving_rows
            else ""
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{_e(title)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}}.bad{{color:#b42318}}.good{{color:#067647}}.muted{{color:#667085}}</style>
</head><body><h1>{_e(title)}</h1><p>Schema: {_e(self.schema_version)}. Records: {len(self.records)}</p>
<h2>Per model/backend summary</h2><table><thead><tr><th>Group</th><th>Runs</th><th>Avg latency ms</th><th>Quality</th><th>Peak memory</th></tr></thead><tbody>{summary_rows}</tbody></table>
{serving_section}
<h2>Trend overview</h2><table><thead><tr><th>Group</th><th>Runs</th><th>Latency trend</th><th>Quality drift trend</th><th>Memory peak trend</th><th>GPU</th><th>Backend</th><th>Commit range</th></tr></thead><tbody>{trend_rows}</tbody></table>
<h2>Regression hints</h2><table><thead><tr><th>Group</th><th>Latency</th><th>Quality</th><th>Memory</th><th>Status</th></tr></thead><tbody>{regression_rows}</tbody></table>
<h2>Run history</h2><table><thead><tr><th>Name</th><th>Kind</th><th>Latency ms</th><th>First token ms</th><th>Tok/s</th><th>Peak</th><th>Quality</th><th>Mean abs</th><th>GPU</th><th>Backend</th><th>Commit</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>"""


def benchmark_dashboard_from_files(paths: list[str | Path] | tuple[str | Path, ...]) -> BenchmarkDashboard:
    return BenchmarkDashboard(tuple(read_benchmark_files(paths)))


def write_benchmark_dashboard_html(
    paths: list[str | Path] | tuple[str | Path, ...],
    output: str | Path,
    *,
    title: str = "llm-memlab benchmark dashboard",
) -> Path:
    dashboard = benchmark_dashboard_from_files(paths)
    path = Path(output)
    path.write_text(dashboard.to_html(title=title), encoding="utf-8")
    return path


def _record_row(record: BenchmarkRecord) -> str:
    extra = record.extra or {}
    metadata = record.metadata or {}
    quality = extra.get("quality_passed")
    css = "good" if quality is True else "bad" if quality is False else ""
    return (
        "<tr>"
        f"<td>{_e(record.name)}</td><td>{_e(record.kind)}</td><td>{record.mean_ms:.3f}</td>"
        f"<td>{_e(extra.get('first_token_ms', 'n/a'))}</td><td>{_e(extra.get('tokens_per_second', 'n/a'))}</td>"
        f"<td>{_fmt_bytes(record.peak_cuda_bytes)}</td><td class='{css}'>{_e(quality)}</td>"
        f"<td>{_e(extra.get('mean_abs', extra.get('prefill_mean_abs', 'n/a')))}</td>"
        f"<td>{_e(metadata.get('gpu', 'n/a'))}</td><td>{_e(metadata.get('backend', 'n/a'))}</td><td>{_e(metadata.get('commit', 'n/a'))}</td>"
        "</tr>"
    )


def _serving_row(record: BenchmarkRecord) -> str:
    extra = record.extra or {}
    token_match = extra.get("token_match")
    css = "good" if token_match is True else "bad" if token_match is False else ""
    return (
        "<tr>"
        f"<td>{_e(record.name)}</td><td>{_e(extra.get('backend_selected', record.metadata.get('backend', 'n/a')))}</td>"
        f"<td>{_e(extra.get('available', 'n/a'))}</td><td>{_e(extra.get('first_token_ms', 'n/a'))}</td>"
        f"<td>{_e(extra.get('tokens_per_second', 'n/a'))}</td><td>{_e(extra.get('cache_hit', 'n/a'))}</td>"
        f"<td>{_e(extra.get('prefix_cache', 'n/a'))}</td><td class='{css}'>{_e(token_match if token_match is not None else 'n/a')}</td>"
        f"<td>{_e(extra.get('fallback_reason') or '')}</td>"
        "</tr>"
    )


def _group_summary(records: tuple[BenchmarkRecord, ...]) -> dict[str, list[BenchmarkRecord]]:
    grouped: dict[str, list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        metadata = record.metadata or {}
        extra = record.extra or {}
        model = extra.get("model") or metadata.get("model") or record.kind
        backend = metadata.get("backend") or extra.get("cache_impl") or "unknown"
        grouped[f"{model} / {backend}"].append(record)
    return dict(sorted(grouped.items()))


def _fmt_quality(records: list[BenchmarkRecord]) -> str:
    values = [record.extra.get("quality_passed") for record in records if "quality_passed" in record.extra]
    if not values:
        return "n/a"
    return f"{sum(1 for item in values if item is True)}/{len(values)} pass"


def _fmt_peak(records: list[BenchmarkRecord]) -> str:
    peaks = [record.peak_cuda_bytes for record in records if record.peak_cuda_bytes is not None]
    return "n/a" if not peaks else format_bytes(max(peaks))


def _trend_row(key: str, records: list[BenchmarkRecord]) -> str:
    first = records[0]
    latest = records[-1]
    return (
        "<tr>"
        f"<td>{_e(key)}</td><td>{len(records)}</td>"
        f"<td>{_e(_latency_trend(first, latest))}</td>"
        f"<td>{_e(_quality_trend(first, latest))}</td>"
        f"<td>{_e(_memory_trend(first, latest))}</td>"
        f"<td>{_e(_latest_meta(records, 'gpu'))}</td><td>{_e(_latest_meta(records, 'backend'))}</td>"
        f"<td>{_e(_commit_range(first, latest))}</td>"
        "</tr>"
    )


def _regression_row(key: str, records: list[BenchmarkRecord]) -> str:
    first = records[0]
    latest = records[-1]
    latency_bad = latest.mean_ms > first.mean_ms * 1.10 if first.mean_ms > 0 else False
    quality_bad = _quality_value(latest) is False
    memory_bad = (
        latest.peak_cuda_bytes is not None and first.peak_cuda_bytes is not None and latest.peak_cuda_bytes > first.peak_cuda_bytes * 1.10
    )
    status = "FAIL" if latency_bad or quality_bad or memory_bad else "OK"
    css = "bad" if status == "FAIL" else "good"
    return (
        "<tr>"
        f"<td>{_e(key)}</td><td>{_e(_latency_trend(first, latest))}</td>"
        f"<td>{_e(_quality_trend(first, latest))}</td><td>{_e(_memory_trend(first, latest))}</td>"
        f"<td class='{css}'>{status}</td>"
        "</tr>"
    )


def _latency_trend(first: BenchmarkRecord, latest: BenchmarkRecord) -> str:
    if first.mean_ms <= 0:
        return f"{latest.mean_ms:.3f} ms"
    delta = (latest.mean_ms - first.mean_ms) / first.mean_ms * 100.0
    return f"{first.mean_ms:.3f} -> {latest.mean_ms:.3f} ms ({delta:+.1f}%)"


def _quality_trend(first: BenchmarkRecord, latest: BenchmarkRecord) -> str:
    first_mean = _mean_abs(first)
    latest_mean = _mean_abs(latest)
    quality = _quality_value(latest)
    if first_mean is None or latest_mean is None:
        return f"quality={quality}"
    return f"mean_abs {first_mean:.6f} -> {latest_mean:.6f}; quality={quality}"


def _memory_trend(first: BenchmarkRecord, latest: BenchmarkRecord) -> str:
    if first.peak_cuda_bytes is None or latest.peak_cuda_bytes is None:
        return _fmt_bytes(latest.peak_cuda_bytes)
    if first.peak_cuda_bytes <= 0:
        return _fmt_bytes(latest.peak_cuda_bytes)
    delta = (latest.peak_cuda_bytes - first.peak_cuda_bytes) / first.peak_cuda_bytes * 100.0
    return f"{_fmt_bytes(first.peak_cuda_bytes)} -> {_fmt_bytes(latest.peak_cuda_bytes)} ({delta:+.1f}%)"


def _latest_meta(records: list[BenchmarkRecord], key: str) -> str:
    for record in reversed(records):
        value = record.metadata.get(key)
        if value:
            return str(value)
    return "n/a"


def _commit_range(first: BenchmarkRecord, latest: BenchmarkRecord) -> str:
    start = first.metadata.get("commit", "n/a")
    end = latest.metadata.get("commit", "n/a")
    return str(start) if start == end else f"{start} -> {end}"


def _quality_value(record: BenchmarkRecord) -> Any:
    return record.extra.get("quality_passed")


def _mean_abs(record: BenchmarkRecord) -> float | None:
    value = record.extra.get("mean_abs", record.extra.get("prefill_mean_abs"))
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _fmt_bytes(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(value)


def _e(value: Any) -> str:
    import html

    return html.escape(str(value))
