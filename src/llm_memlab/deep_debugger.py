from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .attention_debugger import AttentionStats, attention_stats_to_text, collect_attention_stats
from .bytes import format_bytes
from .drift_debugger import DriftReport, compare_layer_drift
from .html_report import trace_interactive_to_html, trace_timeline_to_html
from .quality_metrics import LogitQualityResult, compare_logits
from .torch_debugger import TorchTrace


@dataclass(frozen=True)
class DeepDebugReport:
    baseline_trace: Any
    candidate_trace: Any | None = None
    drift: DriftReport | None = None
    quality: LogitQualityResult | None = None
    attention: tuple[AttentionStats, ...] = ()

    def to_html(self, *, title: str = "llm-memlab deep debug") -> str:
        quality_html = f"<pre>{_e(self.quality.to_text())}</pre>" if self.quality is not None else ""
        details = _layer_details(self.baseline_trace, self.candidate_trace, self.drift)
        details_json = json.dumps(details)
        top_changed = _top_changed_rows(self.drift)
        drift_html = _drift_table(self.drift)
        attention_html = f"<pre>{_e(attention_stats_to_text(self.attention))}</pre>" if self.attention else ""
        candidate_sections = ""
        if self.candidate_trace is not None:
            candidate_sections = f"""
<section><h2>Candidate interactive trace</h2>{trace_interactive_to_html(self.candidate_trace, title="candidate trace")}</section>
<section><h2>Candidate timeline</h2>{trace_timeline_to_html(self.candidate_trace, title="candidate timeline")}</section>
"""
        return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{_e(title)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left;vertical-align:top}}
.grid{{display:grid;grid-template-columns:1fr;gap:20px}}button{{padding:4px 8px;border:1px solid #cfd8e3;border-radius:6px;background:white;cursor:pointer}}
.panel{{border:1px solid #d8dee9;border-radius:8px;padding:12px;background:#fbfcfe}}pre{{white-space:pre-wrap;background:#f6f8fb;padding:12px;border-radius:8px;overflow:auto}}
.metric{{display:inline-block;border:1px solid #d8dee9;border-radius:8px;padding:10px 12px;margin:0 8px 8px 0}}
</style>
<script>
const layerDetails = {details_json};
function showLayer(name){{
  const data = layerDetails[name] || {{}};
  document.getElementById('selected').innerText = 'Selected layer: ' + name;
  document.getElementById('layer-json').innerText = JSON.stringify(data, null, 2);
}}
</script></head>
<body><h1>{_e(title)}</h1>
<div class='metric'>Baseline layers <b>{len(getattr(self.baseline_trace, "records", []))}</b></div>
<div class='metric'>Baseline time <b>{getattr(self.baseline_trace, "total_ms", 0.0):.3f} ms</b></div>
<div class='metric'>Baseline output <b>{format_bytes(sum(r.output_bytes for r in getattr(self.baseline_trace, "records", [])))}</b></div>
<div id='selected' class='panel'>Selected layer: none</div><pre id='layer-json'>Click a layer in the drift or changed-layer tables.</pre>
<div class='grid'>
<section><h2>Top changed tensors</h2>{top_changed}</section>
<section><h2>Output quality</h2>{quality_html}</section>
<section><h2>Layer drift</h2>{drift_html}</section>
<section><h2>Attention heads</h2>{attention_html}</section>
<section><h2>Baseline interactive trace</h2>{trace_interactive_to_html(self.baseline_trace, title="baseline trace")}</section>
<section><h2>Baseline timeline</h2>{trace_timeline_to_html(self.baseline_trace, title="baseline timeline")}</section>
{candidate_sections}
</div></body></html>"""


def build_deep_debug_report(baseline: Any, candidate: Any | None, *args, **kwargs) -> DeepDebugReport:
    torch = _import_torch()
    with torch.no_grad():
        with TorchTrace(baseline) as baseline_trace:
            baseline_out = baseline(*args, **kwargs)
        candidate_trace = None
        drift = None
        quality = None
        if candidate is not None:
            with TorchTrace(candidate) as trace:
                candidate_out = candidate(*args, **kwargs)
            candidate_trace = trace
            drift = compare_layer_drift(baseline, candidate, *args, **kwargs)
            quality = compare_logits(_first_tensor(baseline_out).float(), _first_tensor(candidate_out).float(), top_k=5)
        _, attention = collect_attention_stats(baseline, *args, **kwargs)
    return DeepDebugReport(baseline_trace, candidate_trace, drift, quality, tuple(attention))


def write_deep_debug_html(report: DeepDebugReport, path: str | Path, *, title: str = "llm-memlab deep debug") -> Path:
    output = Path(path)
    output.write_text(report.to_html(title=title), encoding="utf-8")
    return output


def _drift_table(drift: DriftReport | None) -> str:
    if drift is None:
        return ""
    rows = "".join(
        f"<tr><td><button onclick=\"showLayer('{_js(record.name)}')\">{_e(record.name)}</button></td><td>{record.status}</td><td>{_fmt(record.mean_abs_error)}</td><td>{_fmt(record.max_abs_error)}</td><td>{_fmt(record.cosine_similarity)}</td></tr>"
        for record in drift.records
    )
    return f"<table><thead><tr><th>Layer</th><th>Status</th><th>Mean abs</th><th>Max abs</th><th>Cosine</th></tr></thead><tbody>{rows}</tbody></table>"


def _top_changed_rows(drift: DriftReport | None, *, limit: int = 8) -> str:
    if drift is None:
        return ""
    records = [record for record in drift.records if record.mean_abs_error is not None]
    records.sort(key=lambda item: (item.mean_abs_error or 0.0, item.max_abs_error or 0.0), reverse=True)
    rows = "".join(
        f"<tr><td><button onclick=\"showLayer('{_js(record.name)}')\">{_e(record.name)}</button></td><td>{_fmt(record.mean_abs_error)}</td><td>{_fmt(record.max_abs_error)}</td><td>{_fmt(record.cosine_similarity)}</td></tr>"
        for record in records[:limit]
    )
    return f"<table><thead><tr><th>Layer</th><th>Mean abs</th><th>Max abs</th><th>Cosine</th></tr></thead><tbody>{rows}</tbody></table>"


def _layer_details(baseline_trace: Any, candidate_trace: Any | None, drift: DriftReport | None) -> dict[str, Any]:
    baseline = {record.name: _record_detail(record) for record in getattr(baseline_trace, "records", [])}
    candidate = (
        {record.name: _record_detail(record) for record in getattr(candidate_trace, "records", [])} if candidate_trace is not None else {}
    )
    drift_by_name = {record.name: record for record in drift.records} if drift is not None else {}
    names = sorted(set(baseline) | set(candidate) | set(drift_by_name))
    return {
        name: {
            "baseline": baseline.get(name),
            "candidate": candidate.get(name),
            "drift": _drift_detail(drift_by_name.get(name)),
        }
        for name in names
    }


def _record_detail(record: Any) -> dict[str, Any]:
    return {
        "type": record.type_name,
        "elapsed_ms": record.elapsed_ms,
        "input_bytes": record.input_bytes,
        "output_bytes": record.output_bytes,
        "parameter_bytes": record.parameter_bytes,
        "cuda_delta_bytes": record.cuda_delta_bytes,
        "input_shapes": record.input_shapes,
        "output_shapes": record.output_shapes,
        "output_stats": _stats_detail(record.output_stats),
        "has_nan": record.has_nan,
        "has_inf": record.has_inf,
    }


def _drift_detail(record: Any | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "status": record.status,
        "baseline_shape": record.baseline_shape,
        "candidate_shape": record.candidate_shape,
        "mean_abs_error": record.mean_abs_error,
        "max_abs_error": record.max_abs_error,
        "cosine_similarity": record.cosine_similarity,
    }


def _stats_detail(stats: Any | None) -> dict[str, Any] | None:
    if stats is None:
        return None
    return {
        "shape": stats.shape,
        "dtype": stats.dtype,
        "device": stats.device,
        "bytes": stats.bytes,
        "mean": stats.mean,
        "std": stats.std,
        "min": stats.min,
        "max": stats.max,
        "zero_pct": stats.zero_pct,
        "has_nan": stats.has_nan,
        "has_inf": stats.has_inf,
    }


def _first_tensor(value: Any):
    if hasattr(value, "detach"):
        return value.detach()
    if hasattr(value, "logits"):
        return value.logits.detach()
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _js(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def _e(value: Any) -> str:
    import html

    return html.escape(str(value))


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Deep debugger requires PyTorch") from exc
    return torch
