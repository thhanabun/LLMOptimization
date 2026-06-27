from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .bytes import format_bytes
from .report import make_table


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("KV cache inference requires PyTorch. Install with: pip install torch") from exc
    return torch


@dataclass(frozen=True)
class KVCacheConfig:
    num_layers: int
    batch_size: int
    num_heads: int
    head_dim: int
    max_seq_len: int
    dtype: Any = None
    device: Any = None


@dataclass
class KVCacheStats:
    tokens_used: int
    capacity: int
    bytes_allocated: int
    bytes_used: int

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.tokens_used / self.capacity

    def to_text(self) -> str:
        rows = [
            ("Tokens used", self.tokens_used),
            ("Capacity", self.capacity),
            ("Allocated", format_bytes(self.bytes_allocated)),
            ("Used", format_bytes(self.bytes_used)),
            ("Utilization", f"{self.utilization:.1%}"),
        ]
        return make_table(("Metric", "Value"), rows)


class StaticKVCache:
    """Preallocated per-layer KV cache for decode-time attention.

    Layout is [layer, batch, heads, seq, head_dim]. The cache avoids allocating
    a fresh K/V tensor every token. `append_layer` writes new keys/values for one
    layer and returns the currently valid slice for attention.
    """

    def __init__(self, config: KVCacheConfig):
        torch = require_torch()
        self.config = config
        dtype = config.dtype or torch.float16
        self.keys = torch.empty(
            config.num_layers,
            config.batch_size,
            config.num_heads,
            config.max_seq_len,
            config.head_dim,
            device=config.device,
            dtype=dtype,
        )
        self.values = torch.empty_like(self.keys)
        self.length = 0

    @property
    def capacity(self) -> int:
        return self.config.max_seq_len

    @property
    def nbytes(self) -> int:
        return (self.keys.numel() + self.values.numel()) * self.keys.element_size()

    def reset(self) -> None:
        self.length = 0

    def append_layer(self, layer_idx: int, key, value, position: int | None = None):
        pos = self.length if position is None else position
        step = key.shape[-2]
        end = pos + step
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is outside 0..{self.config.num_layers - 1}")
        if end > self.capacity:
            raise ValueError(f"KV cache capacity exceeded: requested {end}, capacity {self.capacity}")
        self.keys[layer_idx, :, :, pos:end, :].copy_(key)
        self.values[layer_idx, :, :, pos:end, :].copy_(value)
        self.length = max(self.length, end)
        return self.get_layer(layer_idx)

    def get_layer(self, layer_idx: int, end: int | None = None):
        stop = self.length if end is None else end
        return (
            self.keys[layer_idx, :, :, :stop, :],
            self.values[layer_idx, :, :, :stop, :],
        )

    def as_legacy_cache(self) -> tuple[tuple[Any, Any], ...]:
        return tuple(self.get_layer(layer_idx) for layer_idx in range(self.config.num_layers))

    def stats(self) -> KVCacheStats:
        used_per_layer = self.config.batch_size * self.config.num_heads * self.length * self.config.head_dim
        used = used_per_layer * self.config.num_layers * 2 * self.keys.element_size()
        return KVCacheStats(
            tokens_used=self.length,
            capacity=self.capacity,
            bytes_allocated=self.nbytes,
            bytes_used=used,
        )


@dataclass(frozen=True)
class DecodeConfig:
    max_new_tokens: int
    eos_token_id: int | None = None
    temperature: float = 0.0
    top_k: int | None = None
    use_cache: bool = True
    return_token_scores: bool = True


@dataclass
class DecodeStep:
    index: int
    token_id: int | list[int]
    elapsed_ms: float
    cache_tokens: int | None = None
    max_logit: float | None = None


@dataclass
class DecodeResult:
    sequences: Any
    steps: list[DecodeStep] = field(default_factory=list)
    past_key_values: Any = None

    @property
    def total_ms(self) -> float:
        return sum(step.elapsed_ms for step in self.steps)

    @property
    def tokens_per_second(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return len(self.steps) * 1000 / self.total_ms

    def to_text(self) -> str:
        rows = [
            ("New tokens", len(self.steps)),
            ("Total decode time", f"{self.total_ms:.3f} ms"),
            ("Throughput", f"{self.tokens_per_second:.2f} tok/s"),
            ("Sequence shape", tuple(self.sequences.shape) if hasattr(self.sequences, "shape") else type(self.sequences).__name__),
        ]
        text = [make_table(("Metric", "Value"), rows)]
        if self.steps:
            step_rows = [
                (step.index, step.token_id, f"{step.elapsed_ms:.3f}", step.cache_tokens if step.cache_tokens is not None else "")
                for step in self.steps
            ]
            text.extend(["", "Decode steps", make_table(("Step", "Token", "ms", "Cache tokens"), step_rows)])
        return "\n".join(text)


def greedy_decode(model, input_ids, config: DecodeConfig, **model_kwargs) -> DecodeResult:
    """Generic HuggingFace-style greedy/sampling decode loop.

    The model may return a dict, an object with `.logits`, or a tuple whose
    first item is logits. If `past_key_values` is returned, the loop feeds only
    the newest token on later steps.
    """

    torch = require_torch()
    sequences = input_ids.clone()
    past_key_values = model_kwargs.pop("past_key_values", None)
    steps: list[DecodeStep] = []
    next_input = input_ids

    with torch.no_grad():
        for index in range(config.max_new_tokens):
            started = time.perf_counter()
            outputs = model(next_input, past_key_values=past_key_values, use_cache=config.use_cache, **model_kwargs)
            logits = _get_logits(outputs)
            past_key_values = _get_past_key_values(outputs, default=past_key_values)
            next_token = sample_next_token(logits[:, -1, :], temperature=config.temperature, top_k=config.top_k)
            elapsed_ms = (time.perf_counter() - started) * 1000
            sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
            cache_tokens = _cache_length(past_key_values)
            max_logit = float(logits[:, -1, :].max().item()) if config.return_token_scores else None
            token_payload: int | list[int]
            token_payload = int(next_token.item()) if next_token.numel() == 1 else [int(item) for item in next_token.tolist()]
            steps.append(DecodeStep(index=index, token_id=token_payload, elapsed_ms=elapsed_ms, cache_tokens=cache_tokens, max_logit=max_logit))
            if config.eos_token_id is not None and bool((next_token == config.eos_token_id).all().item()):
                break
            next_input = next_token[:, None] if config.use_cache and past_key_values is not None else sequences

    return DecodeResult(sequences=sequences, steps=steps, past_key_values=past_key_values)


def sample_next_token(logits, *, temperature: float = 0.0, top_k: int | None = None):
    torch = require_torch()
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scores = logits / temperature
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1)
        probs = torch.softmax(values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return indices.gather(-1, sampled[:, None]).squeeze(-1)
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise TypeError("Model output must be dict-like, object with .logits, or tuple(logits, ...)")


def _get_past_key_values(outputs, *, default=None):
    if isinstance(outputs, dict):
        return outputs.get("past_key_values", default)
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        return outputs[1]
    return default


def _cache_length(past_key_values) -> int | None:
    if past_key_values is None:
        return None
    if isinstance(past_key_values, StaticKVCache):
        return past_key_values.length
    try:
        first = past_key_values[0][0]
    except Exception:
        return None
    if hasattr(first, "shape") and len(first.shape) >= 3:
        return int(first.shape[-2])
    return None
