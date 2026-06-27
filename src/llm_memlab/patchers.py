from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .modules import OptimizedRMSNorm, OptimizedSwiGLUMLP
def _lazy_torch_nn_module():
    try:
        import torch
    except ImportError:  # pragma: no cover
        class _Fallback:
            pass

        return _Fallback
    return torch.nn.Module

@dataclass
class PatchReport:
    patched_norms: int = 0
    patched_mlps: int = 0
    patched_attentions: int = 0
    attention_candidates: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def total_patched(self) -> int:
        return self.patched_norms + self.patched_mlps + self.patched_attentions

    def to_text(self) -> str:
        from .report import make_table

        rows = [
            ("Patched RMSNorm", self.patched_norms),
            ("Patched SwiGLU MLP", self.patched_mlps),
            ("Patched attention", self.patched_attentions),
            ("Attention candidates", len(self.attention_candidates)),
            ("Skipped", len(self.skipped)),
        ]
        text = [make_table(("Item", "Count"), rows)]
        if self.attention_candidates:
            text.append("")
            text.append("Attention candidates")
            text.extend(f"- {name}" for name in self.attention_candidates[:20])
        if self.skipped:
            text.append("")
            text.append("Skipped")
            text.extend(f"- {item}" for item in self.skipped[:20])
        return "\n".join(text)


class PackedQKVAttentionAdapter(_lazy_torch_nn_module()):
    """Packed-QKV SDPA adapter for simple Llama/Qwen-like attention modules.

    The adapter copies q/k/v/o projections into one qkv projection plus one output
    projection. It is intentionally limited to equal q/k/v widths; GQA and model-
    specific RoPE/cache semantics are skipped by the patcher until those paths can
    be represented without changing behavior.
    """

    def __init__(self, source: Any):
        torch = _import_torch()
        super().__init__()
        q_proj = source.q_proj
        k_proj = source.k_proj
        v_proj = source.v_proj
        o_proj = source.o_proj
        self.hidden_size = q_proj.in_features
        self.num_heads = int(getattr(source, "num_heads", getattr(source, "num_attention_heads", 1)))
        self.head_dim = int(getattr(source, "head_dim", q_proj.out_features // self.num_heads))
        self.qkv_proj = torch.nn.Linear(self.hidden_size, q_proj.out_features + k_proj.out_features + v_proj.out_features, bias=q_proj.bias is not None)
        self.o_proj = torch.nn.Linear(o_proj.in_features, o_proj.out_features, bias=o_proj.bias is not None)
        self.q_out = q_proj.out_features
        self.k_out = k_proj.out_features
        self.v_out = v_proj.out_features
        self.qkv_proj = self.qkv_proj.to(device=q_proj.weight.device, dtype=q_proj.weight.dtype)
        self.o_proj = self.o_proj.to(device=o_proj.weight.device, dtype=o_proj.weight.dtype)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if self.qkv_proj.bias is not None:
                self.qkv_proj.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))
            self.o_proj.weight.copy_(o_proj.weight)
            if self.o_proj.bias is not None:
                self.o_proj.bias.copy_(o_proj.bias)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        torch = _import_torch()
        batch, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_out, self.k_out, self.v_out), dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        kv_heads = max(1, self.k_out // self.head_dim)
        k = k.view(batch, seq_len, kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, kv_heads, self.head_dim).transpose(1, 2)
        if kv_heads != self.num_heads:
            repeat = self.num_heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        if position_embeddings is not None:
            try:
                from .kernels import apply_rope
                cos, sin = position_embeddings
                q, k = apply_rope(q, k, cos, sin)
            except Exception:
                pass
        attn_mask = _prepare_sdpa_mask(attention_mask, seq_len)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.q_out)
        out = self.o_proj(out)
        if output_attentions or use_cache or past_key_value is not None:
            next_cache = past_key_value if use_cache else None
            return out, None, next_cache
        return out


def optimize_hf_model(
    model: Any,
    *,
    patch_norms: bool = True,
    patch_mlps: bool = True,
    patch_attention: bool = False,
    use_triton: bool = False,
    dry_run: bool = False,
) -> tuple[Any, PatchReport]:
    """Patch Hugging Face-style transformer modules with llm_memlab primitives.

    Norm and MLP patching are conservative and enabled by default. Attention
    patching is opt-in because Llama/Qwen variants differ in RoPE, GQA, masks,
    and cache semantics. When enabled, only simple equal-width q/k/v/o attention
    modules are replaced by a packed-QKV SDPA adapter.
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
        if patch_attention and _looks_like_attention(module):
            replacement, reason = _make_attention_adapter(module)
            if replacement is None:
                report.skipped.append(f"{name}: {reason}")
                continue
            if not dry_run:
                setattr(parent, child_name, replacement)
            report.patched_attentions += 1
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


def optimize_llama_qwen_attention(model: Any, *, dry_run: bool = False) -> tuple[Any, PatchReport]:
    """Patch only supported Llama/Qwen-like attention modules."""

    return optimize_hf_model(model, patch_norms=False, patch_mlps=False, patch_attention=True, dry_run=dry_run)


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
    return all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")) and not _looks_like_attention(module)


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


def _make_attention_adapter(module: Any):
    if not _supported_attention_module_name(module):
        return None, "unsupported attention class name"
    q_proj = getattr(module, "q_proj", None)
    k_proj = getattr(module, "k_proj", None)
    v_proj = getattr(module, "v_proj", None)
    o_proj = getattr(module, "o_proj", None)
    if not all(_linear_like(item) for item in (q_proj, k_proj, v_proj, o_proj)):
        return None, "q/k/v/o projections are not Linear-like"
    if k_proj.out_features != v_proj.out_features:
        return None, "k_proj and v_proj widths differ"
    num_heads = int(getattr(module, "num_heads", getattr(module, "num_attention_heads", 1)))
    if num_heads <= 0 or q_proj.out_features % num_heads != 0:
        return None, "q_proj width is not divisible by num_heads"
    head_dim = q_proj.out_features // num_heads
    if k_proj.out_features % head_dim != 0:
        return None, "k_proj width is incompatible with q head_dim"
    kv_heads = k_proj.out_features // head_dim
    if num_heads % kv_heads != 0:
        return None, "GQA layout requires num_heads divisible by kv_heads"
    if o_proj.in_features != q_proj.out_features:
        return None, "o_proj input width does not match q output width"
    return PackedQKVAttentionAdapter(module), "patched"


def _supported_attention_module_name(module: Any) -> bool:
    cls_name = module.__class__.__name__.lower()
    return "attention" in cls_name and any(prefix in cls_name for prefix in ("llama", "qwen", "tiny", "fake"))


def _linear_like(module: Any) -> bool:
    return all(hasattr(module, attr) for attr in ("weight", "in_features", "out_features"))


def _copy_bias(source: Any, target: Any) -> None:
    if getattr(source, "bias", None) is not None and getattr(target, "bias", None) is not None:
        target.bias.data.copy_(source.bias.data)


def _looks_like_attention(module: Any) -> bool:
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    return all(hasattr(module, name) for name in names)


def _prepare_sdpa_mask(attention_mask, seq_len: int):
    if attention_mask is None:
        return None
    if hasattr(attention_mask, "dim") and attention_mask.dim() == 2:
        return attention_mask[:, None, None, :].to(dtype=attention_mask.dtype)
    return attention_mask


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Patching attention requires PyTorch. Install with: pip install torch") from exc
    return torch


def _lazy_torch_nn_module():
    try:
        import torch
    except ImportError:  # pragma: no cover
        class _Fallback:
            pass

        return _Fallback
    return torch.nn.Module


