from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bytes import format_bytes

DEBUGGER_V2_SCHEMA_VERSION = "llm_memlab.debugger_v2.v1"


@dataclass(frozen=True)
class DebuggerV2Report:
    title: str = "llm-memlab debugger v2"
    deep_debug: Any | None = None
    cache_certification: Any | None = None
    memory_profile: Any | None = None
    metadata: dict[str, Any] | None = None
    schema_version: str = DEBUGGER_V2_SCHEMA_VERSION

    def to_html(self) -> str:
        metadata = self.metadata or {}
        cache_html = _cache_certification_section(self.cache_certification)
        memory_html = _memory_profile_section(self.memory_profile)
        deep_html = self.deep_debug.to_html(title="deep debug") if self.deep_debug is not None else "<p>No deep debug report attached.</p>"
        meta_rows = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>" for k, v in sorted(metadata.items()))
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>{_e(self.title)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}
nav button{{margin-right:8px;padding:8px 10px;border:1px solid #cfd8e3;border-radius:6px;background:#fff;cursor:pointer}}
section{{display:none;margin-top:16px}}section.active{{display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left;vertical-align:top}}
.metric{{display:inline-block;border:1px solid #d8dee9;border-radius:8px;padding:10px 12px;margin:0 8px 8px 0}}
iframe{{width:100%;height:720px;border:1px solid #d8dee9;border-radius:8px}}
</style>
<script>function show(id){{for(const el of document.querySelectorAll('section'))el.classList.remove('active');document.getElementById(id).classList.add('active');}}</script>
</head><body><h1>{_e(self.title)}</h1>
<div class='metric'>Schema <b>{_e(self.schema_version)}</b></div>
<nav><button onclick="show('cache')">Cache certification</button><button onclick="show('memory')">Memory</button><button onclick="show('layers')">Layers / drift</button><button onclick="show('meta')">Metadata</button></nav>
<section id='cache' class='active'>{cache_html}</section>
<section id='memory'>{memory_html}</section>
<section id='layers'><iframe srcdoc="{_attr(deep_html)}"></iframe></section>
<section id='meta'><table><tbody>{meta_rows}</tbody></table></section>
</body></html>"""


def debugger_v2_to_html(
    *,
    deep_debug: Any | None = None,
    cache_certification: Any | None = None,
    memory_profile: Any | None = None,
    title: str = "llm-memlab debugger v2",
    metadata: dict[str, Any] | None = None,
) -> str:
    return DebuggerV2Report(title, deep_debug, cache_certification, memory_profile, metadata).to_html()


def write_debugger_v2_html(report: DebuggerV2Report, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(report.to_html(), encoding="utf-8")
    return output


def _cache_certification_section(report: Any | None) -> str:
    if report is None:
        return "<p>No cache certification report attached.</p>"
    rows = []
    for item in getattr(report, "results", []):
        rows.append(
            "<tr>"
            f"<td>{_e(getattr(getattr(item, 'case', None), 'name', 'case'))}</td>"
            f"<td>{'PASS' if getattr(item, 'passed', False) else 'FAIL'}</td>"
            f"<td>{_e(getattr(item, 'cache_impl', 'n/a'))}</td>"
            f"<td>{_e(getattr(item, 'requested_cache_impl', 'n/a'))}</td>"
            f"<td>{_e(getattr(item, 'fallback_reason', None) or 'n/a')}</td>"
            f"<td>{_fmt(getattr(item, 'prefill_mean_abs', None))}</td>"
            f"<td>{_fmt(getattr(item, 'generated_token_agreement', None))}</td>"
            f"<td>{format_bytes(getattr(item, 'peak_cuda_bytes', None)) if getattr(item, 'peak_cuda_bytes', None) is not None else 'n/a'}</td>"
            "</tr>"
        )
    return (
        "<h2>Cache certification</h2><table><thead><tr><th>Case</th><th>Status</th><th>Selected</th><th>Requested</th><th>Fallback</th><th>Logit drift</th><th>Token match</th><th>Peak</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _memory_profile_section(profile: Any | None) -> str:
    if profile is None:
        return "<p>No memory profile attached.</p>"
    attr_rows = []
    for item in getattr(profile, "attributions", []):
        attr_rows.append(
            f"<tr><td>{_e(item.name)}</td><td>{_e(item.kind)}</td><td>{format_bytes(item.bytes)}</td><td>{_e(item.reason)}</td></tr>"
        )
    sample_rows = []
    for sample in getattr(profile, "samples", []):
        sample_rows.append(
            f"<tr><td>{sample.step}</td><td>{_e(sample.phase)}</td><td>{_fmt(sample.elapsed_ms)}</td><td>{format_bytes(sample.allocated_bytes) if sample.allocated_bytes is not None else 'n/a'}</td><td>{format_bytes(sample.reserved_bytes) if sample.reserved_bytes is not None else 'n/a'}</td><td>{format_bytes(sample.cache_bytes) if sample.cache_bytes is not None else 'n/a'}</td></tr>"
        )
    return (
        "<h2>Memory timeline</h2><table><thead><tr><th>Step</th><th>Phase</th><th>ms</th><th>Allocated</th><th>Reserved</th><th>Cache</th></tr></thead><tbody>"
        + "".join(sample_rows)
        + "</tbody></table><h2>Attribution</h2><table><thead><tr><th>Name</th><th>Kind</th><th>Bytes</th><th>Reason</th></tr></thead><tbody>"
        + "".join(attr_rows)
        + "</tbody></table>"
    )


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _e(value: Any) -> str:
    import html

    return html.escape(str(value))


def _attr(value: Any) -> str:
    import html

    return html.escape(str(value), quote=True)
