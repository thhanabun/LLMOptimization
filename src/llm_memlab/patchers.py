from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .modules import OptimizedRMSNorm, OptimizedSwiGLUMLP


@dataclass
class PatchReport:
    patched_norms: int = 0
    patched_mlps: int = 0
    attention_candidates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total_patched(self) -> int:
        return self.patched_norms + self.patched_mlps

    def to_text(self) -> str:
        from .report import make_table

        rows = [
            ("Patched RMSNorm", self.patched_norms),
            ("Patched SwiGLU MLP", self.patched_mlps),
            ("Attention candidates", len(self.attention_candidates)),
            ("Skipped", len(self.skipped)),
        ]
        text = [make_table(("Item", "Count"), rows)]
        if self.attention_candidates:
            text.append("")
            text.append("Attention candidates not replaced automatically")
            text.extend(f"- {name}" for name in self.attention_candidates[:20])
        if self.skipped:
            text.append("")
            text.append("Skipped")
            text.extend(f"- {item}" for item in self.skipped[:20])
        return "\n".join(text)


def optimize_hf_model(
    model: Any,
    *,
    patch_norms: bool = True,
    patch_mlps: bool = True,
    use_triton: bool = False,
    dry_run: bool = False,
) -> tuple[Any, PatchReport]:
    """Patch Hugging Face-style transformer modules with llm_memlab primitives.

    The patcher is deliberately conservative. It replaces modules whose local
    interface is stable across Llama/Qwen-like models:

    - RMSNorm-like modules with `weight` and `eps` or `variance_epsilon`.
    - SwiGLU MLP modules with `gate_proj`, `up_proj`, and `down_proj`.

    Attention modules are reported as candidates instead of replaced because HF
    attention signatures vary by model, cache class, masks, RoPE implementation,
    and grouped-query attention layout.
    """

    report = PatchReport()
    for name, module in list(model.named_modules()):
        if name == "":
            continue
        if _looks_like_attention(module):
            report.attention_candidates.append(name)
        parent, child_name = _parent_and_child(model, name)
        if parent is None:
            continue
        if patch_norms and _looks_like_rms_norm(module):
            replacement = _make_rms_norm(module)
            if replacement is None:
                report.skipped.append(f"{name}: unsupported RMSNorm shape")
                continue
            if not dry_run:
                setattr(parent, child_name, replacement)
            report.patched_norms += 1
            continue
        if patch_mlps and _looks_like_swiglu_mlp(module):
            replacement = _make_swiglu_mlp(module, use_triton=use_triton)
            if replacement is None:
                report.skipped.append(f"{name}: unsupported SwiGLU MLP")
                continue
            if not dry_run:
                setattr(parent, child_name, replacement)
            report.patched_mlps += 1
    return model, report


def _parent_and_child(root: Any, dotted_name: str) -> tuple[Any | None, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            return None, parts[-1]
        parent = getattr(parent, part)
    return parent, parts[-1]


def _looks_like_rms_norm(module: Any) -> bool:
    cls_name = module.__class__.__name__.lower()
    return "rmsnorm" in cls_name and hasattr(module, "weight")


def _make_rms_norm(module: Any):
    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(weight, "numel"):
        return None
    eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
    replacement = OptimizedRMSNorm(weight.numel(), eps=float(eps), bias=hasattr(module, "bias") and getattr(module, "bias") is not None)
    replacement = replacement.to(device=weight.device, dtype=weight.dtype)
    replacement.weight.data.copy_(weight.data)
    bias = getattr(module, "bias", None)
    if bias is not None and replacement.bias is not None:
        replacement.bias.data.copy_(bias.data)
    return replacement


def _looks_like_swiglu_mlp(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj"))


def _make_swiglu_mlp(module: Any, *, use_triton: bool):
    gate = getattr(module, "gate_proj", None)
    up = getattr(module, "up_proj", None)
    down = getattr(module, "down_proj", None)
    if not all(_linear_like(item) for item in (gate, up, down)):
        return None
    hidden_size = gate.in_features
    intermediate_size = gate.out_features
    has_bias = gate.bias is not None or up.bias is not None or down.bias is not None
    replacement = OptimizedSwiGLUMLP(hidden_size, intermediate_size, bias=has_bias, use_triton=use_triton)
    replacement = replacement.to(device=gate.weight.device, dtype=gate.weight.dtype)
    replacement.gate_proj.weight.data.copy_(gate.weight.data)
    replacement.up_proj.weight.data.copy_(up.weight.data)
    replacement.down_proj.weight.data.copy_(down.weight.data)
    _copy_bias(gate, replacement.gate_proj)
    _copy_bias(up, replacement.up_proj)
    _copy_bias(down, replacement.down_proj)
    return replacement


def _linear_like(module: Any) -> bool:
    return all(hasattr(module, attr) for attr in ("weight", "in_features", "out_features"))


def _copy_bias(source: Any, target: Any) -> None:
    if getattr(source, "bias", None) is not None and getattr(target, "bias", None) is not None:
        target.bias.data.copy_(source.bias.data)


def _looks_like_attention(module: Any) -> bool:
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    return all(hasattr(module, name) for name in names)
