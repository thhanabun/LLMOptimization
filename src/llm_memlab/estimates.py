from __future__ import annotations

from dataclasses import dataclass

from .bytes import dtype_size_bytes, format_bytes
from .report import make_table


@dataclass(frozen=True)
class TransformerConfig:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    sequence_length: int
    batch_size: int = 1
    dtype: str = "bf16"
    training: str = "inference"
    optimizer: str = "adamw"
    activation_checkpointing: str = "none"
    use_flash_attention: bool = True
    tie_embeddings: bool = True
    num_key_value_heads: int | None = None
    lora_rank: int = 0
    include_kv_cache: bool = True
    safety_factor: float = 1.08

    def normalized(self) -> "TransformerConfig":
        training = self.training.lower()
        if training not in {"inference", "lora", "full"}:
            raise ValueError("training must be one of: inference, lora, full")
        checkpointing = self.activation_checkpointing.lower()
        if checkpointing not in {"none", "selective", "full"}:
            raise ValueError("activation_checkpointing must be one of: none, selective, full")
        return TransformerConfig(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            training=training,
            optimizer=self.optimizer.lower(),
            activation_checkpointing=checkpointing,
            use_flash_attention=self.use_flash_attention,
            tie_embeddings=self.tie_embeddings,
            num_key_value_heads=self.num_key_value_heads or self.num_attention_heads,
            lora_rank=self.lora_rank,
            include_kv_cache=self.include_kv_cache,
            safety_factor=self.safety_factor,
        )


@dataclass(frozen=True)
class MemoryEstimate:
    config: TransformerConfig
    parameter_count: int
    trainable_parameter_count: int
    parameter_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    kv_cache_bytes: int
    temp_bytes: int
    peak_bytes: int
    notes: tuple[str, ...]

    def to_rows(self) -> list[tuple[str, str]]:
        return [
            ("Parameters", f"{self.parameter_count:,} ({format_bytes(self.parameter_bytes)})"),
            ("Trainable parameters", f"{self.trainable_parameter_count:,}"),
            ("Gradients", format_bytes(self.gradient_bytes)),
            ("Optimizer states", format_bytes(self.optimizer_bytes)),
            ("Saved activations", format_bytes(self.activation_bytes)),
            ("KV cache", format_bytes(self.kv_cache_bytes)),
            ("Temporary/workspace", format_bytes(self.temp_bytes)),
            ("Estimated peak", format_bytes(self.peak_bytes)),
        ]

    def to_text(self) -> str:
        cfg = self.config
        header = (
            f"{cfg.num_layers}L h={cfg.hidden_size} ffn={cfg.intermediate_size} "
            f"heads={cfg.num_attention_heads} seq={cfg.sequence_length} batch={cfg.batch_size} "
            f"dtype={cfg.dtype} mode={cfg.training}"
        )
        text = [header, "", make_table(("Item", "Estimate"), self.to_rows())]
        if self.notes:
            text.append("")
            text.append("Notes")
            text.extend(f"- {note}" for note in self.notes)
        return "\n".join(text)


def preset_config(name: str, *, sequence_length: int = 2048, batch_size: int = 1, dtype: str = "bf16") -> TransformerConfig:
    normalized = name.lower().replace("_", "-")
    presets = {
        "tiny": dict(num_layers=2, hidden_size=128, intermediate_size=384, num_attention_heads=4, vocab_size=4096),
        "1b-like": dict(num_layers=24, hidden_size=2048, intermediate_size=5504, num_attention_heads=16, vocab_size=32000),
        "3b-like": dict(num_layers=28, hidden_size=2560, intermediate_size=6912, num_attention_heads=20, vocab_size=32000),
        "7b-like": dict(num_layers=32, hidden_size=4096, intermediate_size=11008, num_attention_heads=32, vocab_size=32000),
        "13b-like": dict(num_layers=40, hidden_size=5120, intermediate_size=13824, num_attention_heads=40, vocab_size=32000),
    }
    if normalized not in presets:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(sorted(presets))}")
    return TransformerConfig(
        **presets[normalized],
        sequence_length=sequence_length,
        batch_size=batch_size,
        dtype=dtype,
    )


def estimate_transformer_memory(config: TransformerConfig) -> MemoryEstimate:
    cfg = config.normalized()
    dtype_bytes = dtype_size_bytes(cfg.dtype)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_width = head_dim * (cfg.num_key_value_heads or cfg.num_attention_heads)

    embedding_params = cfg.vocab_size * cfg.hidden_size
    q_params = cfg.hidden_size * cfg.hidden_size
    k_params = cfg.hidden_size * kv_width
    v_params = cfg.hidden_size * kv_width
    o_params = cfg.hidden_size * cfg.hidden_size
    mlp_params = (cfg.hidden_size * cfg.intermediate_size * 2) + (cfg.intermediate_size * cfg.hidden_size)
    norm_params = cfg.hidden_size * 2
    layer_params = q_params + k_params + v_params + o_params + mlp_params + norm_params
    lm_head_params = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.hidden_size
    parameter_count = int(embedding_params + cfg.num_layers * layer_params + cfg.hidden_size + lm_head_params)

    lora_params = 0
    if cfg.lora_rank > 0:
        lora_linear_shapes = [
            (cfg.hidden_size, cfg.hidden_size),
            (cfg.hidden_size, kv_width),
            (cfg.hidden_size, kv_width),
            (cfg.hidden_size, cfg.hidden_size),
            (cfg.hidden_size, cfg.intermediate_size),
            (cfg.hidden_size, cfg.intermediate_size),
            (cfg.intermediate_size, cfg.hidden_size),
        ]
        lora_params = cfg.num_layers * sum(cfg.lora_rank * (inp + out) for inp, out in lora_linear_shapes)

    if cfg.training == "full":
        trainable_params = parameter_count
    elif cfg.training == "lora":
        trainable_params = lora_params
    else:
        trainable_params = 0

    parameter_bytes = int(parameter_count * dtype_bytes)
    gradient_bytes = int(trainable_params * dtype_bytes)
    optimizer_bytes = _optimizer_bytes(trainable_params, cfg.optimizer)

    tokens = cfg.batch_size * cfg.sequence_length
    hidden_act = tokens * cfg.hidden_size * dtype_bytes
    mlp_act = tokens * cfg.intermediate_size * dtype_bytes
    attention_scores = 0
    if not cfg.use_flash_attention:
        attention_scores = cfg.batch_size * cfg.num_attention_heads * cfg.sequence_length * cfg.sequence_length * dtype_bytes

    per_layer_saved = (6 * hidden_act) + (2 * mlp_act) + attention_scores
    checkpoint_factor = {"none": 1.0, "selective": 0.55, "full": 0.22}[cfg.activation_checkpointing]
    activation_bytes = 0
    if cfg.training != "inference":
        activation_bytes = int(cfg.num_layers * per_layer_saved * checkpoint_factor)

    kv_cache_bytes = 0
    if cfg.include_kv_cache:
        kv_cache_bytes = int(cfg.num_layers * 2 * cfg.batch_size * cfg.sequence_length * kv_width * dtype_bytes)

    temp_bytes = int(max(hidden_act * 8, mlp_act * 2, attention_scores * 0.15))
    base_peak = parameter_bytes + gradient_bytes + optimizer_bytes + activation_bytes + kv_cache_bytes + temp_bytes
    peak_bytes = int(base_peak * cfg.safety_factor)

    notes = _notes(cfg, parameter_bytes, activation_bytes, kv_cache_bytes, optimizer_bytes)
    return MemoryEstimate(
        config=cfg,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_params,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        kv_cache_bytes=kv_cache_bytes,
        temp_bytes=temp_bytes,
        peak_bytes=peak_bytes,
        notes=notes,
    )


def _optimizer_bytes(trainable_params: int, optimizer: str) -> int:
    if trainable_params == 0 or optimizer in {"none", "false", "off"}:
        return 0
    if optimizer in {"adam", "adamw"}:
        return int(trainable_params * 8)
    if optimizer in {"sgd", "momentum"}:
        return int(trainable_params * 4)
    if optimizer in {"8bit-adam", "adam8bit", "paged-adamw-8bit"}:
        return int(trainable_params * 2)
    raise ValueError("optimizer must be one of: none, adamw, adam, sgd, momentum, 8bit-adam, paged-adamw-8bit")


def _notes(
    cfg: TransformerConfig, parameter_bytes: int, activation_bytes: int, kv_cache_bytes: int, optimizer_bytes: int
) -> tuple[str, ...]:
    notes: list[str] = []
    if cfg.training == "lora" and cfg.lora_rank <= 0:
        notes.append("LoRA mode has rank 0, so only base model memory is counted.")
    if cfg.training == "full" and optimizer_bytes > parameter_bytes:
        notes.append("Adam-style optimizer states dominate memory; sharding or offload should be considered.")
    if activation_bytes > parameter_bytes:
        notes.append("Saved activations exceed weights; checkpointing and sequence packing will matter.")
    if kv_cache_bytes > parameter_bytes:
        notes.append("KV cache exceeds weights; use grouped-query attention, quantized cache, or shorter active context.")
    if not cfg.use_flash_attention:
        notes.append("Attention score tensors are materialized; FlashAttention-style kernels can remove most of this.")
    if cfg.activation_checkpointing == "full":
        notes.append("Full checkpointing lowers activation memory but increases recompute work.")
    return tuple(notes)
