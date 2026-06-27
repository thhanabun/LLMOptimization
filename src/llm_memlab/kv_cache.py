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
    reference_bytes: int | None = None
    storage_dtype: str | None = None

    @property
    def utilization(self) -> float:
        if self.capacity == 0:
            return 0.0
        return self.tokens_used / self.capacity

    @property
    def compression_ratio(self) -> float | None:
        if self.reference_bytes is None or self.bytes_allocated == 0:
            return None
        return self.reference_bytes / self.bytes_allocated

    def to_text(self) -> str:
        rows = [
            ("Tokens used", self.tokens_used),
            ("Capacity", self.capacity),
            ("Allocated", format_bytes(self.bytes_allocated)),
            ("Used", format_bytes(self.bytes_used)),
            ("Utilization", f"{self.utilization:.1%}"),
        ]
        if self.storage_dtype is not None:
            rows.append(("Storage dtype", self.storage_dtype))
        if self.reference_bytes is not None:
            rows.append(("Reference fp cache", format_bytes(self.reference_bytes)))
        if self.compression_ratio is not None:
            rows.append(("Compression", f"{self.compression_ratio:.2f}x"))
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
        pos, end = self._validate_append(layer_idx, key, value, position)
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
            storage_dtype=str(self.keys.dtype).replace("torch.", ""),
        )

    def _validate_append(self, layer_idx: int, key, value, position: int | None) -> tuple[int, int]:
        pos = self.length if position is None else position
        step = key.shape[-2]
        end = pos + step
        if layer_idx < 0 or layer_idx >= self.config.num_layers:
            raise IndexError(f"layer_idx {layer_idx} is outside 0..{self.config.num_layers - 1}")
        if key.shape != value.shape:
            raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
        expected = (self.config.batch_size, self.config.num_heads, step, self.config.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(f"expected K/V shape {expected}, got {tuple(key.shape)}")
        if end > self.capacity:
            raise ValueError(f"KV cache capacity exceeded: requested {end}, capacity {self.capacity}")
        return pos, end


class QuantizedStaticKVCache(StaticKVCache):
    """Preallocated KV cache with int/float compressed storage modes.

    `quant_dtype` supports the common inference cache choices: int8, uint8,
    fp16, bf16, fp32, and fp8_e4m3fn when the installed PyTorch exposes it.
    Quantized modes keep per-token/per-head metadata and dequantize reads to
    `config.dtype` (or fp16 by default) so existing attention code can keep the
    same math path.
    """

    def __init__(self, config: KVCacheConfig, *, quant_dtype: str | Any = "int8", scale_dtype=None, eps: float = 1e-6):
        torch = require_torch()
        self.config = config
        self.output_dtype = config.dtype or torch.float16
        self.quant_dtype_name, self.storage_dtype = resolve_kv_storage_dtype(quant_dtype)
        self.scale_dtype = scale_dtype or torch.float16
        self.eps = eps
        shape = (config.num_layers, config.batch_size, config.num_heads, config.max_seq_len, config.head_dim)
        scale_shape = (config.num_layers, config.batch_size, config.num_heads, config.max_seq_len, 1)
        self.keys = torch.empty(shape, device=config.device, dtype=self.storage_dtype)
        self.values = torch.empty_like(self.keys)
        self.length = 0
        self.key_scales = None
        self.value_scales = None
        self.key_zero_points = None
        self.value_zero_points = None
        if self.quant_dtype_name == "int8":
            self.key_scales = torch.empty(scale_shape, device=config.device, dtype=self.scale_dtype)
            self.value_scales = torch.empty_like(self.key_scales)
        elif self.quant_dtype_name == "uint8":
            self.key_scales = torch.empty(scale_shape, device=config.device, dtype=self.scale_dtype)
            self.value_scales = torch.empty_like(self.key_scales)
            self.key_zero_points = torch.empty(scale_shape, device=config.device, dtype=torch.uint8)
            self.value_zero_points = torch.empty_like(self.key_zero_points)

    @property
    def nbytes(self) -> int:
        tensors = [self.keys, self.values]
        tensors.extend(tensor for tensor in (self.key_scales, self.value_scales, self.key_zero_points, self.value_zero_points) if tensor is not None)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def reference_nbytes(self) -> int:
        torch = require_torch()
        dtype = self.output_dtype or torch.float16
        element_size = torch.empty((), dtype=dtype).element_size()
        elements = self.config.num_layers * self.config.batch_size * self.config.num_heads * self.config.max_seq_len * self.config.head_dim * 2
        return elements * element_size

    def append_layer(self, layer_idx: int, key, value, position: int | None = None):
        pos, end = self._validate_append(layer_idx, key, value, position)
        if self.quant_dtype_name == "int8":
            q_key, key_scale = quantize_int8_per_token(key, eps=self.eps)
            q_value, value_scale = quantize_int8_per_token(value, eps=self.eps)
            self.keys[layer_idx, :, :, pos:end, :].copy_(q_key)
            self.values[layer_idx, :, :, pos:end, :].copy_(q_value)
            self.key_scales[layer_idx, :, :, pos:end, :].copy_(key_scale.to(dtype=self.scale_dtype))
            self.value_scales[layer_idx, :, :, pos:end, :].copy_(value_scale.to(dtype=self.scale_dtype))
        elif self.quant_dtype_name == "uint8":
            q_key, key_scale, key_zp = quantize_uint8_per_token(key, eps=self.eps)
            q_value, value_scale, value_zp = quantize_uint8_per_token(value, eps=self.eps)
            self.keys[layer_idx, :, :, pos:end, :].copy_(q_key)
            self.values[layer_idx, :, :, pos:end, :].copy_(q_value)
            self.key_scales[layer_idx, :, :, pos:end, :].copy_(key_scale.to(dtype=self.scale_dtype))
            self.value_scales[layer_idx, :, :, pos:end, :].copy_(value_scale.to(dtype=self.scale_dtype))
            self.key_zero_points[layer_idx, :, :, pos:end, :].copy_(key_zp)
            self.value_zero_points[layer_idx, :, :, pos:end, :].copy_(value_zp)
        else:
            self.keys[layer_idx, :, :, pos:end, :].copy_(key.to(dtype=self.storage_dtype))
            self.values[layer_idx, :, :, pos:end, :].copy_(value.to(dtype=self.storage_dtype))
        self.length = max(self.length, end)
        return self.get_layer(layer_idx)

    def get_layer(self, layer_idx: int, end: int | None = None):
        stop = self.length if end is None else end
        key_store = self.keys[layer_idx, :, :, :stop, :]
        value_store = self.values[layer_idx, :, :, :stop, :]
        if self.quant_dtype_name == "int8":
            key = dequantize_int8_per_token(key_store, self.key_scales[layer_idx, :, :, :stop, :], dtype=self.output_dtype)
            value = dequantize_int8_per_token(value_store, self.value_scales[layer_idx, :, :, :stop, :], dtype=self.output_dtype)
        elif self.quant_dtype_name == "uint8":
            key = dequantize_uint8_per_token(
                key_store,
                self.key_scales[layer_idx, :, :, :stop, :],
                self.key_zero_points[layer_idx, :, :, :stop, :],
                dtype=self.output_dtype,
            )
            value = dequantize_uint8_per_token(
                value_store,
                self.value_scales[layer_idx, :, :, :stop, :],
                self.value_zero_points[layer_idx, :, :, :stop, :],
                dtype=self.output_dtype,
            )
        else:
            key = key_store.to(dtype=self.output_dtype)
            value = value_store.to(dtype=self.output_dtype)
        return key, value

    def stats(self) -> KVCacheStats:
        used_vectors = self.config.num_layers * self.config.batch_size * self.config.num_heads * self.length
        used = used_vectors * self.config.head_dim * 2 * self.keys.element_size()
        if self.key_scales is not None:
            used += used_vectors * 2 * self.key_scales.element_size()
        if self.key_zero_points is not None:
            used += used_vectors * 2 * self.key_zero_points.element_size()
        return KVCacheStats(
            tokens_used=self.length,
            capacity=self.capacity,
            bytes_allocated=self.nbytes,
            bytes_used=used,
            reference_bytes=self.reference_nbytes,
            storage_dtype=self.quant_dtype_name,
        )


def resolve_kv_storage_dtype(quant_dtype: str | Any) -> tuple[str, Any]:
    torch = require_torch()
    if not isinstance(quant_dtype, str):
        name = str(quant_dtype).replace("torch.", "")
        return name, quant_dtype
    normalized = quant_dtype.lower().replace("torch.", "").replace("float16", "fp16").replace("float32", "fp32").replace("bfloat16", "bf16")
    aliases = {
        "int8": torch.int8,
        "uint8": torch.uint8,
        "fp16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if normalized in aliases:
        canonical = {"half": "fp16", "float": "fp32"}.get(normalized, normalized)
        return canonical, aliases[normalized]
    if normalized in {"fp8", "fp8_e4m3fn", "float8_e4m3fn"} and hasattr(torch, "float8_e4m3fn"):
        return "fp8_e4m3fn", torch.float8_e4m3fn
    raise ValueError("quant_dtype must be one of: int8, uint8, fp16, bf16, fp32, fp8_e4m3fn(if supported)")


def quantize_int8_per_token(x, *, eps: float = 1e-6):
    """Quantize last-dimension vectors to int8 with one absmax scale per vector."""

    torch = require_torch()
    scale = x.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(eps) / 127.0
    q = torch.round(x.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_int8_per_token(q, scale, *, dtype=None):
    torch = require_torch()
    out = q.float() * scale.float()
    return out.to(dtype=dtype or torch.float16)


def quantize_uint8_per_token(x, *, eps: float = 1e-6):
    """Asymmetric uint8 quantization with per-vector min/max metadata."""

    torch = require_torch()
    x_float = x.detach().float()
    x_min = x_float.amin(dim=-1, keepdim=True)
    x_max = x_float.amax(dim=-1, keepdim=True)
    scale = ((x_max - x_min).clamp_min(eps)) / 255.0
    zero_point = torch.round((-x_min / scale).clamp(0, 255)).to(torch.uint8)
    q = torch.round(x_float / scale + zero_point.float()).clamp(0, 255).to(torch.uint8)
    return q, scale, zero_point


def dequantize_uint8_per_token(q, scale, zero_point, *, dtype=None):
    torch = require_torch()
    out = (q.float() - zero_point.float()) * scale.float()
    return out.to(dtype=dtype or torch.float16)


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
    """Generic HuggingFace-style greedy/sampling decode loop."""

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
