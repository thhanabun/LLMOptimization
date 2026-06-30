from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .report import make_table


@dataclass(frozen=True)
class AttentionStats:
    name: str
    shape: tuple[int, ...]
    entropy: float
    max_probability: float
    dead_head_fraction: float
    has_nan: bool
    has_inf: bool

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Layer", self.name),
                ("Shape", self.shape),
                ("Entropy", f"{self.entropy:.6f}"),
                ("Max probability", f"{self.max_probability:.6f}"),
                ("Dead heads", f"{self.dead_head_fraction:.1%}"),
                ("NaN", self.has_nan),
                ("Inf", self.has_inf),
            ],
        )


def collect_attention_stats(model: Any, *args, **kwargs) -> tuple[Any, tuple[AttentionStats, ...]]:
    torch = _import_torch()
    records: list[AttentionStats] = []
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            hidden = _first_tensor(inputs)
            if hidden is None:
                return
            stat = analyze_qk_attention(module, hidden, name=name)
            if stat is not None:
                records.append(stat)

        return hook

    for name, module in model.named_modules():
        if _looks_like_qk_attention(module):
            handles.append(module.register_forward_hook(make_hook(name or module.__class__.__name__)))
    try:
        with torch.no_grad():
            output = model(*args, **kwargs)
    finally:
        while handles:
            handles.pop().remove()
    return output, tuple(records)


def analyze_qk_attention(module: Any, hidden_states, *, name: str = "attention") -> AttentionStats | None:
    torch = _import_torch()
    if not _looks_like_qk_attention(module) or hidden_states.dim() != 3:
        return None
    q_proj = module.q_proj
    k_proj = module.k_proj
    num_heads = int(getattr(module, "num_heads", getattr(module, "num_attention_heads", 1)))
    if q_proj.out_features % num_heads != 0 or k_proj.out_features != q_proj.out_features:
        return None
    head_dim = q_proj.out_features // num_heads
    batch, seq, _ = hidden_states.shape
    q = q_proj(hidden_states).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    k = k_proj(hidden_states).view(batch, seq, num_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
    max_probability = probs.max()
    per_head_energy = probs.abs().sum(dim=(-1, -2)).mean(dim=0)
    dead_head_fraction = (per_head_energy <= 1e-6).float().mean()
    return AttentionStats(
        name=name,
        shape=tuple(probs.shape),
        entropy=float(entropy.item()),
        max_probability=float(max_probability.item()),
        dead_head_fraction=float(dead_head_fraction.item()),
        has_nan=bool(torch.isnan(probs).any().item()),
        has_inf=bool(torch.isinf(probs).any().item()),
    )


def attention_stats_to_text(stats: tuple[AttentionStats, ...]) -> str:
    rows = [
        (item.name, item.shape, f"{item.entropy:.6f}", f"{item.max_probability:.6f}", f"{item.dead_head_fraction:.1%}") for item in stats
    ]
    return make_table(("Layer", "Shape", "Entropy", "Max prob", "Dead heads"), rows)


def _looks_like_qk_attention(module: Any) -> bool:
    return all(hasattr(module, name) for name in ("q_proj", "k_proj"))


def _first_tensor(value: Any):
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value
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


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Attention debugging requires PyTorch. Install with: pip install torch") from exc
    return torch
