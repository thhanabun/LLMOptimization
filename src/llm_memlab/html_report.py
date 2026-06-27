from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .bytes import format_bytes


def trace_to_html(trace: Any, *, title: str = "llm-memlab trace") -> str:
    records = list(getattr(trace, "records", []))
    max_ms = max((record.elapsed_ms for record in records), default=1.0)
    max_bytes = max((record.output_bytes for record in records), default=1)
    rows = []
    for record in records:
        time_width = 2 + (record.elapsed_ms / max_ms) * 98
        byte_width = 2 + (record.output_bytes / max_bytes) * 98
        stats = getattr(record, "output_stats", None)
        stat_text = _stats_text(stats)
        rows.append(
            f"""
            <tr>
              <td><code>{_e(record.name)}</code></td>
              <td>{_e(record.type_name)}</td>
              <td>{record.elapsed_ms:.3f}<div class='bar'><span style='width:{time_width:.1f}%'></span></div></td>
              <td>{format_bytes(record.input_bytes)}</td>
              <td>{format_bytes(record.output_bytes)}<div class='bar mem'><span style='width:{byte_width:.1f}%'></span></div></td>
              <td>{format_bytes(record.parameter_bytes)}</td>
              <td><code>{_e(record.input_shapes)}</code></td>
              <td><code>{_e(record.output_shapes)}</code></td>
              <td>{_e(stat_text)}</td>
              <td>{'NaN' if record.has_nan else ''} {'Inf' if record.has_inf else ''}</td>
            </tr>
            """
        )
    hot = "".join(f"<li><code>{_e(r.name)}</code> {r.elapsed_ms:.3f} ms, output {format_bytes(r.output_bytes)}</li>" for r in trace.slowest(8))
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{_e(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17202a; }}
h1 {{ margin-bottom: 4px; }}
.summary {{ display: flex; gap: 16px; margin: 18px 0; flex-wrap: wrap; }}
.metric {{ border: 1px solid #d8dee9; border-radius: 8px; padding: 12px 14px; min-width: 160px; }}
.metric b {{ display: block; font-size: 22px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e9f0; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f8fb; position: sticky; top: 0; }}
code {{ white-space: pre-wrap; }}
.bar {{ margin-top: 4px; height: 5px; background: #edf1f7; border-radius: 4px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; background: #3b82f6; }}
.bar.mem span {{ background: #10b981; }}
.warning {{ color: #b42318; font-weight: 600; }}
</style>
</head>
<body>
<h1>{_e(title)}</h1>
<p>Layer-level runtime, memory, shape, and activation statistics.</p>
<div class='summary'>
  <div class='metric'>Total time<b>{trace.total_ms:.3f} ms</b></div>
  <div class='metric'>Layers traced<b>{len(records)}</b></div>
  <div class='metric'>Total output bytes<b>{format_bytes(sum(r.output_bytes for r in records))}</b></div>
</div>
<h2>Hot layers</h2>
<ul>{hot}</ul>
<h2>Layer table</h2>
<table>
<thead><tr><th>Module</th><th>Type</th><th>ms</th><th>Input</th><th>Output</th><th>Params</th><th>Input shape</th><th>Output shape</th><th>Stats</th><th>Flags</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>"""


def write_trace_html(trace: Any, path: str | Path, *, title: str = "llm-memlab trace") -> Path:
    output = Path(path)
    output.write_text(trace_to_html(trace, title=title), encoding="utf-8")
    return output


def _stats_text(stats: Any) -> str:
    if stats is None or getattr(stats, "mean", None) is None:
        return "n/a"
    return (
        f"mean={stats.mean:.4g} std={stats.std:.4g} min={stats.min:.4g} "
        f"max={stats.max:.4g} zero={stats.zero_pct:.2%}"
    )


def _e(value: Any) -> str:
    return html.escape(str(value))



def trace_timeline_to_html(trace: Any, *, title: str = "llm-memlab timeline") -> str:
    records = list(getattr(trace, "records", []))
    total = max(sum(record.elapsed_ms for record in records), 1e-9)
    cursor = 0.0
    rows = []
    max_bytes = max((record.output_bytes for record in records), default=1)
    for record in records:
        left = (cursor / total) * 100
        width = max((record.elapsed_ms / total) * 100, 0.5)
        mem_width = max((record.output_bytes / max_bytes) * 100, 1.0)
        cursor += record.elapsed_ms
        rows.append(
            f"""
            <div class='row'>
              <div class='label'><code>{_e(record.name)}</code><small>{_e(record.type_name)} · {record.elapsed_ms:.3f} ms · {format_bytes(record.output_bytes)}</small></div>
              <div class='track'><span class='time' style='left:{left:.2f}%;width:{width:.2f}%'></span></div>
              <div class='mem'><span style='width:{mem_width:.2f}%'></span></div>
            </div>
            """
        )
    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<title>{_e(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17202a; }}
.summary {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.metric {{ border:1px solid #d8dee9; border-radius:8px; padding:10px 12px; min-width:150px; }}
.metric b {{ display:block; font-size:20px; }}
.row {{ display:grid; grid-template-columns: 280px 1fr 160px; gap:12px; align-items:center; padding:8px 0; border-bottom:1px solid #e5e9f0; }}
.label small {{ display:block; color:#526070; margin-top:3px; }}
.track {{ position:relative; height:14px; background:#edf1f7; border-radius:7px; overflow:hidden; }}
.time {{ position:absolute; top:0; bottom:0; background:#3b82f6; border-radius:7px; }}
.mem {{ height:8px; background:#edf1f7; border-radius:6px; overflow:hidden; }}
.mem span {{ display:block; height:100%; background:#10b981; }}
code {{ white-space:pre-wrap; }}
</style>
</head>
<body>
<h1>{_e(title)}</h1>
<p>Sequential layer runtime timeline with output-memory bars.</p>
<div class='summary'>
  <div class='metric'>Total time<b>{getattr(trace, 'total_ms', 0.0):.3f} ms</b></div>
  <div class='metric'>Layers<b>{len(records)}</b></div>
  <div class='metric'>Output bytes<b>{format_bytes(sum(r.output_bytes for r in records))}</b></div>
</div>
{''.join(rows)}
</body>
</html>"""


def write_timeline_html(trace: Any, path: str | Path, *, title: str = "llm-memlab timeline") -> Path:
    output = Path(path)
    output.write_text(trace_timeline_to_html(trace, title=title), encoding="utf-8")
    return output
