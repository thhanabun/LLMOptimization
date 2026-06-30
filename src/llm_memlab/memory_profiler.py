from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bytes import format_bytes
from .kv_cache import sample_next_token
from .report import make_table

MEMORY_PROFILE_SCHEMA_VERSION = "llm_memlab.memory_profile.v1"


@dataclass(frozen=True)
class CUDAMemorySample:
    step: int
    phase: str
    allocated_bytes: int | None
    reserved_bytes: int | None
    peak_allocated_bytes: int | None
    elapsed_ms: float
    cache_bytes: int | None = None
    fragmentation_bytes: int | None = None


@dataclass(frozen=True)
class MemoryAttribution:
    name: str
    kind: str
    bytes: int
    reason: str


@dataclass
class DecodeMemoryProfile:
    name: str
    samples: list[CUDAMemorySample] = field(default_factory=list)
    attributions: list[MemoryAttribution] = field(default_factory=list)
    sequences: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MEMORY_PROFILE_SCHEMA_VERSION

    @property
    def peak_allocated_bytes(self) -> int | None:
        values = [item.peak_allocated_bytes for item in self.samples if item.peak_allocated_bytes is not None]
        return max(values, default=None)

    @property
    def final_allocated_bytes(self) -> int | None:
        for item in reversed(self.samples):
            if item.allocated_bytes is not None:
                return item.allocated_bytes
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "metadata": self.metadata,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "final_allocated_bytes": self.final_allocated_bytes,
            "sequence_shape": tuple(self.sequences.shape) if hasattr(self.sequences, "shape") else None,
            "attributions": [asdict(item) for item in self.attributions],
            "samples": [asdict(item) for item in self.samples],
        }

    def to_text(self) -> str:
        rows = [
            ("Schema", self.schema_version),
            ("Name", self.name),
            ("Samples", len(self.samples)),
            ("Peak allocated", _fmt_bytes(self.peak_allocated_bytes)),
            ("Final allocated", _fmt_bytes(self.final_allocated_bytes)),
            ("Attributed memory", _fmt_bytes(sum(item.bytes for item in self.attributions))),
            ("Device", self.metadata.get("device", "n/a")),
        ]
        return make_table(("Metric", "Value"), rows)


@dataclass(frozen=True)
class MemoryProfileComparison:
    baseline: DecodeMemoryProfile
    candidate: DecodeMemoryProfile

    @property
    def peak_delta_bytes(self) -> int | None:
        if self.baseline.peak_allocated_bytes is None or self.candidate.peak_allocated_bytes is None:
            return None
        return self.candidate.peak_allocated_bytes - self.baseline.peak_allocated_bytes

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Baseline peak", _fmt_bytes(self.baseline.peak_allocated_bytes)),
                ("Candidate peak", _fmt_bytes(self.candidate.peak_allocated_bytes)),
                ("Peak delta", _fmt_bytes(self.peak_delta_bytes)),
            ],
        )


def profile_decode_memory(
    model: Any,
    input_ids,
    *,
    max_new_tokens: int = 8,
    name: str = "decode",
    temperature: float = 0.0,
    top_k: int | None = None,
    **model_kwargs,
) -> DecodeMemoryProfile:
    torch = _import_torch()
    profile = DecodeMemoryProfile(name=name, metadata=_memory_metadata(torch))
    sequences = input_ids.clone()
    past = model_kwargs.pop("past_key_values", None)
    next_input = input_ids
    _reset_peak(torch)
    with torch.no_grad():
        for step in range(max_new_tokens):
            started = time.perf_counter()
            profile.samples.append(_sample(torch, step, "before", 0.0, past))
            outputs = model(next_input, past_key_values=past, use_cache=True, **model_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            logits = _get_logits(outputs)
            past = _get_past_key_values(outputs, default=past)
            profile.attributions = _attribute_cache_memory(past)
            profile.samples.append(_sample(torch, step, "after", elapsed_ms, past))
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k)
            sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
            next_input = next_token[:, None]
    profile.sequences = sequences
    return profile


def compare_memory_profiles(baseline: DecodeMemoryProfile, candidate: DecodeMemoryProfile) -> MemoryProfileComparison:
    return MemoryProfileComparison(baseline, candidate)


def memory_profile_to_html(profile: DecodeMemoryProfile, *, title: str | None = None) -> str:
    title = title or f"llm-memlab memory profile: {profile.name}"
    max_peak = max((item.peak_allocated_bytes or 0 for item in profile.samples), default=1)
    rows = []
    for item in profile.samples:
        width = 0.0 if not item.peak_allocated_bytes else max(1.0, item.peak_allocated_bytes / max_peak * 100)
        rows.append(
            f"<tr><td>{item.step}</td><td>{item.phase}</td><td>{item.elapsed_ms:.3f}</td><td>{_fmt_bytes(item.allocated_bytes)}</td><td>{_fmt_bytes(item.reserved_bytes)}</td><td>{_fmt_bytes(item.peak_allocated_bytes)}<div class='bar'><span style='width:{width:.1f}%'></span></div></td><td>{_fmt_bytes(item.cache_bytes)}</td><td>{_fmt_bytes(item.fragmentation_bytes)}</td></tr>"
        )
    attr_rows = "".join(
        f"<tr><td>{_e(item.name)}</td><td>{_e(item.kind)}</td><td>{_fmt_bytes(item.bytes)}</td><td>{_e(item.reason)}</td></tr>"
        for item in profile.attributions
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>{_e(title)}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}}.bar{{height:6px;background:#edf1f7;border-radius:6px;overflow:hidden}}.bar span{{display:block;height:100%;background:#10b981}}</style></head>
<body><h1>{_e(title)}</h1><pre>{_e(profile.to_text())}</pre><table><thead><tr><th>Step</th><th>Phase</th><th>ms</th><th>Allocated</th><th>Reserved</th><th>Peak</th><th>Cache</th><th>Fragmentation</th></tr></thead><tbody>{"".join(rows)}</tbody></table><h2>Layer/cache attribution</h2><table><thead><tr><th>Name</th><th>Kind</th><th>Bytes</th><th>Reason</th></tr></thead><tbody>{attr_rows}</tbody></table></body></html>"""


def write_memory_profile_html(profile: DecodeMemoryProfile, path: str | Path, *, title: str | None = None) -> Path:
    output = Path(path)
    output.write_text(memory_profile_to_html(profile, title=title), encoding="utf-8")
    return output


def write_memory_profile_json(profile: DecodeMemoryProfile, path: str | Path) -> Path:
    output = Path(path)
    output.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return output


def _sample(torch, step: int, phase: str, elapsed_ms: float, cache: Any) -> CUDAMemorySample:
    allocated: int | None
    reserved: int | None
    peak: int | None
    fragmentation: int | None
    if torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        peak = int(torch.cuda.max_memory_allocated())
        fragmentation = max(0, reserved - allocated)
    else:
        allocated = reserved = peak = None
        fragmentation = None
    cache_bytes = None
    if hasattr(cache, "stats"):
        try:
            cache_bytes = int(cache.stats().bytes_allocated)
        except Exception:
            cache_bytes = None
    return CUDAMemorySample(step, phase, allocated, reserved, peak, elapsed_ms, cache_bytes, fragmentation)


def _attribute_cache_memory(cache: Any) -> list[MemoryAttribution]:
    if cache is None:
        return []
    if hasattr(cache, "stats"):
        try:
            return [MemoryAttribution("cache", cache.__class__.__name__, int(cache.stats().bytes_allocated), "llm-memlab cache stats")]
        except Exception:
            return []
    attributions: list[MemoryAttribution] = []
    if isinstance(cache, (tuple, list)):
        for layer_idx, layer in enumerate(cache):
            if not isinstance(layer, (tuple, list)) or len(layer) < 2:
                continue
            key, value = layer[0], layer[1]
            bytes_used = _tensor_bytes(key) + _tensor_bytes(value)
            attributions.append(MemoryAttribution(f"layer_{layer_idx}.kv_cache", "past_key_values", bytes_used, "key + value tensors"))
    return attributions


def _tensor_bytes(tensor: Any) -> int:
    if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
        return 0
    return int(tensor.numel() * tensor.element_size())


def _memory_metadata(torch) -> dict[str, Any]:
    metadata = {"cuda_available": bool(torch.cuda.is_available()), "device": "cpu"}
    if torch.cuda.is_available():
        metadata.update(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": ".".join(str(item) for item in torch.cuda.get_device_capability(0)),
                "torch_cuda": torch.version.cuda,
            }
        )
    return metadata


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise TypeError("Model output must expose logits")


def _get_past_key_values(outputs, *, default=None):
    if isinstance(outputs, dict):
        return outputs.get("past_key_values", default)
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        return outputs[1]
    return default


def _fmt_bytes(value: int | None) -> str:
    return "n/a" if value is None else format_bytes(value)


def _e(value: Any) -> str:
    import html

    return html.escape(str(value))


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Memory profiling requires PyTorch") from exc
    return torch
