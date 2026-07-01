from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bytes import format_bytes
from .patchers import optimize_hf_model
from .report import make_table


@dataclass(frozen=True)
class ModelArchitectureInfo:
    model_type: str = "unknown"
    num_layers: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    parameter_count: int = 0
    trainable_parameter_count: int = 0
    parameter_bytes: int = 0
    dtype_summary: tuple[str, ...] = ()
    device_summary: tuple[str, ...] = ()
    patchable_norms: int = 0
    patchable_mlps: int = 0
    attention_candidates: tuple[str, ...] = ()
    kv_cache_bytes_fp16: int | None = None
    kv_cache_bytes_int8: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_text(self) -> str:
        rows = [
            ("Model type", self.model_type),
            ("Layers", _fmt_optional(self.num_layers)),
            ("Hidden size", _fmt_optional(self.hidden_size)),
            ("Intermediate size", _fmt_optional(self.intermediate_size)),
            ("Attention heads", _fmt_optional(self.num_attention_heads)),
            ("KV heads", _fmt_optional(self.num_key_value_heads)),
            ("Head dim", _fmt_optional(self.head_dim)),
            ("Vocab size", _fmt_optional(self.vocab_size)),
            ("Max positions", _fmt_optional(self.max_position_embeddings)),
            ("Parameters", f"{self.parameter_count:,}"),
            ("Trainable parameters", f"{self.trainable_parameter_count:,}"),
            ("Parameter bytes", format_bytes(self.parameter_bytes)),
            ("Dtypes", ", ".join(self.dtype_summary) or "n/a"),
            ("Devices", ", ".join(self.device_summary) or "n/a"),
            ("Patchable RMSNorm", self.patchable_norms),
            ("Patchable SwiGLU MLP", self.patchable_mlps),
            ("Attention candidates", len(self.attention_candidates)),
        ]
        if self.kv_cache_bytes_fp16 is not None:
            rows.append(("KV cache fp16 estimate", format_bytes(self.kv_cache_bytes_fp16)))
        if self.kv_cache_bytes_int8 is not None:
            rows.append(("KV cache int8 estimate", format_bytes(self.kv_cache_bytes_int8)))
        text = [make_table(("Item", "Value"), rows)]
        if self.attention_candidates:
            text.extend(["", "Attention candidates"])
            text.extend(f"- {name}" for name in self.attention_candidates[:24])
        if self.notes:
            text.extend(["", "Notes"])
            text.extend(f"- {note}" for note in self.notes)
        return "\n".join(text)


def inspect_model(model: Any, *, max_seq_len: int | None = None, batch_size: int = 1) -> ModelArchitectureInfo:
    raw_config = getattr(model, "config", None)
    config = _language_config(raw_config)
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    param_count = sum(param.numel() for param in params)
    trainable_count = sum(param.numel() for param in params if getattr(param, "requires_grad", False))
    param_bytes = sum(param.numel() * param.element_size() for param in params)
    dtypes = tuple(sorted({str(param.dtype).replace("torch.", "") for param in params}))
    devices = tuple(sorted({str(param.device) for param in params}))
    patch_report = None
    if hasattr(model, "named_modules"):
        _, patch_report = optimize_hf_model(model, dry_run=True)

    hidden = _get_config_value(config, "hidden_size", "n_embd", "d_model")
    heads = _get_config_value(config, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _get_config_value(config, "num_key_value_heads", "num_kv_heads") or heads
    head_dim = _get_config_value(config, "head_dim")
    if head_dim is None and hidden is not None and heads:
        head_dim = hidden // heads
    layers = _get_config_value(config, "num_hidden_layers", "n_layer", "num_layers")
    max_positions = max_seq_len or _get_config_value(config, "max_position_embeddings", "n_positions", "seq_length")

    kv_fp16 = None
    kv_int8 = None
    if all(value is not None for value in (layers, kv_heads, head_dim, max_positions)):
        vectors = int(layers) * batch_size * int(kv_heads) * int(max_positions) * int(head_dim) * 2
        kv_fp16 = vectors * 2
        scale_vectors = int(layers) * batch_size * int(kv_heads) * int(max_positions) * 2
        kv_int8 = vectors + scale_vectors * 2

    notes: list[str] = []
    attention_candidates = () if patch_report is None else tuple(patch_report.attention_candidates)
    patched_norms = 0 if patch_report is None else patch_report.patched_norms
    patched_mlps = 0 if patch_report is None else patch_report.patched_mlps
    if not attention_candidates:
        notes.append("No Llama/Qwen-style attention candidates were detected by the conservative patcher.")
    if raw_config is None:
        notes.append("Model has no `.config`; architecture fields were inferred from module/parameter structure only.")
    elif config is not raw_config:
        notes.append(f"Architecture fields were read from nested language config under {getattr(raw_config, 'model_type', 'multimodal')}.")

    return ModelArchitectureInfo(
        model_type=str(_get_config_value(config, "model_type") or model.__class__.__name__),
        num_layers=layers,
        hidden_size=hidden,
        intermediate_size=_get_config_value(config, "intermediate_size", "n_inner", "ffn_dim"),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=_get_config_value(config, "vocab_size"),
        max_position_embeddings=max_positions,
        parameter_count=param_count,
        trainable_parameter_count=trainable_count,
        parameter_bytes=param_bytes,
        dtype_summary=dtypes,
        device_summary=devices,
        patchable_norms=patched_norms,
        patchable_mlps=patched_mlps,
        attention_candidates=attention_candidates,
        kv_cache_bytes_fp16=kv_fp16,
        kv_cache_bytes_int8=kv_int8,
        notes=tuple(notes),
    )


def load_hf_model(model_name_or_path: str, *, device: str | None = None, dtype: str | None = None, local_files_only: bool = False):
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("Loading Hugging Face models requires: pip install torch transformers") from exc
    torch_dtype = _resolve_torch_dtype(torch, dtype)
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _get_config_value(config: Any, *names: str):
    if config is None:
        return None
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
        if isinstance(config, dict) and name in config:
            return config[name]
    return None


def _language_config(config: Any):
    if config is None:
        return None
    for name in ("text_config", "language_config", "llm_config"):
        nested = getattr(config, name, None)
        if nested is not None:
            return nested
        if isinstance(config, dict) and config.get(name) is not None:
            return config[name]
    return config


def _resolve_torch_dtype(torch, dtype: str | None):
    if dtype is None or dtype == "auto":
        return None
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype {dtype!r}. Use auto, fp16, bf16, or fp32.")
    return mapping[dtype]


def _fmt_optional(value: Any) -> str:
    return "n/a" if value is None else str(value)
