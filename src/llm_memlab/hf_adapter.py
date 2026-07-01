from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Any

from .hf_adapter_contract import MemoryAdapterProtocol
from .hf_cache_policy import HFCachePolicy, select_hf_cache_policy
from .hf_cache_profiles import QuantizedCacheCertificationProfile
from .kv_cache import DecodeConfig, KVCacheConfig, PagedKVCache, QuantizedStaticKVCache, sample_next_token
from .report import make_table


def _transformers_cache_base():
    try:
        from transformers.cache_utils import Cache
    except Exception:  # pragma: no cover
        return object
    return Cache


@dataclass(frozen=True)
class MemoryFirstHFConfig:
    cache: str = "quantized"
    quant_dtype: str = "int8"
    model_name: str | None = None
    page_size: int = 16
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_k: int | None = None
    allow_experimental_direct_cache: bool = False
    allow_experimental_quantized_cache: bool = False
    quantized_prefill_token_limit: int = 1
    quantized_profiles: tuple[QuantizedCacheCertificationProfile, ...] | None = None


@dataclass(frozen=True)
class FamilyAdapterCapabilities:
    family: str
    cache_positions: bool = True
    attention_mask: bool = True
    rotary_position_embeddings: bool = True
    grouped_query_attention: bool = True
    sliding_window: bool = False
    quirks: tuple[str, ...] = field(default_factory=tuple)

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Family", self.family),
                ("Cache positions", self.cache_positions),
                ("Attention mask", self.attention_mask),
                ("RoPE", self.rotary_position_embeddings),
                ("GQA/MQA", self.grouped_query_attention),
                ("Sliding window", self.sliding_window),
                ("Quirks", "; ".join(self.quirks)),
            ],
        )


@dataclass
class MemoryFirstGenerateResult:
    sequences: Any
    cache: Any
    steps: int
    cache_impl: str
    fallback_reason: str | None = None
    direct_cache: bool = False
    requested_cache_impl: str | None = None

    def to_text(self) -> str:
        rows = [
            ("Steps", self.steps),
            ("Cache impl", self.cache_impl),
            ("Requested cache", self.requested_cache_impl or self.cache_impl),
            ("Direct cache", self.direct_cache),
            ("Fallback reason", self.fallback_reason or "n/a"),
            ("Sequence shape", tuple(self.sequences.shape) if hasattr(self.sequences, "shape") else type(self.sequences).__name__),
        ]
        if hasattr(self.cache, "stats"):
            rows.append(("Cache bytes", self.cache.stats().bytes_allocated))
        return make_table(("Metric", "Value"), rows)


@dataclass(frozen=True)
class HFAdapterInfo:
    family: str
    supports_cache_api: bool
    strategy: str

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Family", self.family),
                ("Transformers Cache API", self.supports_cache_api),
                ("Strategy", self.strategy),
            ],
        )


@dataclass(frozen=True)
class HFGenerateIntegrationPlan:
    family: str
    adapter: str
    cache_position: bool
    attention_mask: bool
    sliding_window: bool
    grouped_query_attention: bool
    fallback_reason: str | None = None

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Family", self.family),
                ("Adapter", self.adapter),
                ("Cache position", self.cache_position),
                ("Attention mask", self.attention_mask),
                ("Sliding window", self.sliding_window),
                ("GQA/MQA", self.grouped_query_attention),
                ("Fallback", self.fallback_reason or "n/a"),
            ],
        )


class MemoryFirstTransformersCache(_transformers_cache_base()):  # type: ignore[misc]
    """Small Transformers Cache-compatible facade backed by llm-memlab caches."""

    def __init__(self, config: KVCacheConfig, *, cache: str = "quantized", quant_dtype: str = "int8", page_size: int = 16):
        self.config = config
        self.cache_impl = cache
        self.quant_dtype = quant_dtype
        self.page_size = page_size
        self.storage = (
            PagedKVCache(config, page_size=page_size) if cache == "paged" else QuantizedStaticKVCache(config, quant_dtype=quant_dtype)
        )
        self._max_batch_size = int(config.batch_size)
        self._max_cache_len = int(config.max_seq_len)
        self._layer_lengths = [0 for _ in range(int(config.num_layers))]

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_initialized(self) -> bool:
        return True

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def max_cache_len(self) -> int:
        return self._max_cache_len

    @property
    def is_sliding(self):
        return tuple(False for _ in range(self.config.num_layers))

    def update(self, key_states, value_states, layer_idx: int, cache_kwargs: dict[str, Any] | None = None):
        position = _cache_position_from_kwargs(cache_kwargs)
        if position is None:
            position = self._layer_lengths[layer_idx]
        result = self.storage.append_layer(layer_idx, key_states, value_states, position=position)
        self._layer_lengths[layer_idx] = max(self._layer_lengths[layer_idx], int(position) + int(key_states.shape[-2]))
        return result

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx is None:
            return max(self._layer_lengths, default=0)
        if int(layer_idx) >= len(self._layer_lengths):
            return 0
        return int(self._layer_lengths[int(layer_idx)])

    def get_max_length(self) -> int:
        return int(self.config.max_seq_len)

    def get_max_cache_shape(self) -> int:
        return int(self.config.max_seq_len)

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return max(0, min(self.get_seq_length(layer_idx), self.get_max_length() - int(new_seq_length)))

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.get_seq_length(layer_idx) + int(query_length), 0

    def has_previous_state(self) -> bool:
        return self.get_seq_length() > 0

    def reset(self) -> None:
        self.storage.reset()
        self._layer_lengths = [0 for _ in range(int(self.config.num_layers))]

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = max(0, self.get_seq_length() + int(max_length))
        self.storage.length = min(self.storage.length, int(max_length))
        self._layer_lengths = [min(length, int(max_length)) for length in self._layer_lengths]

    def batch_select_indices(self, indices) -> None:
        index = indices.to(self.storage.keys.device) if hasattr(indices, "to") else indices
        self.storage.keys = self.storage.keys.index_select(1, index)
        self.storage.values = self.storage.values.index_select(1, index)

    def batch_repeat_interleave(self, repeats: int) -> None:
        self.storage.keys = self.storage.keys.repeat_interleave(repeats, dim=1)
        self.storage.values = self.storage.values.repeat_interleave(repeats, dim=1)

    def to_legacy_cache(self):
        return self.storage.as_legacy_cache()

    def reorder_cache(self, beam_idx) -> None:
        if not hasattr(beam_idx, "to"):
            return
        index = beam_idx.to(self.storage.keys.device)
        self.storage.keys = self.storage.keys.index_select(1, index)
        self.storage.values = self.storage.values.index_select(1, index)
        for name in ("key_scales", "value_scales", "key_zero_points", "value_zero_points"):
            tensor = getattr(self.storage, name, None)
            if tensor is not None:
                setattr(self.storage, name, tensor.index_select(1, index))

    def stats(self):
        return self.storage.stats()


class BaseFamilyMemoryAdapter:
    family: str = "generic"
    capabilities = FamilyAdapterCapabilities(
        family="generic", rotary_position_embeddings=False, grouped_query_attention=False, quirks=("generic fallback",)
    )

    def __init__(self, model: Any, config: MemoryFirstHFConfig | None = None):
        self.model = model
        self.config = config or MemoryFirstHFConfig()

    @classmethod
    def detect(cls, model: Any) -> bool:
        family = _model_family(model)
        return family.startswith(cls.family)

    def make_cache(self, input_ids, *, max_new_tokens: int | None = None) -> MemoryFirstTransformersCache:
        return make_transformers_cache_from_model(
            self.model,
            self.config,
            batch_size=int(input_ids.shape[0]),
            max_seq_len=int(input_ids.shape[-1] + (max_new_tokens or self.config.max_new_tokens) + 64),
            dtype=getattr(input_ids, "dtype", None),
            device=getattr(input_ids, "device", None),
        )

    def prepare_generate_kwargs(self, input_ids, **generate_kwargs) -> dict[str, Any]:
        kwargs = dict(generate_kwargs)
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.max_new_tokens))
        kwargs.setdefault("max_new_tokens", max_new_tokens)
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("past_key_values", self.make_cache(input_ids, max_new_tokens=max_new_tokens))
        kwargs = self._prepare_family_kwargs(input_ids, kwargs)
        return _filter_generate_kwargs(self.model, kwargs)

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        return kwargs

    def certify(self, input_ids: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        return {
            "family": self.family,
            "adapter": self.__class__.__name__,
            "capabilities": self.capabilities,
            "integration_plan": self.integration_plan(input_ids).to_text(),
            "fallback_reason": self.fallback_reason(input_ids, **kwargs),
        }

    def integration_plan(self, input_ids: Any | None = None) -> HFGenerateIntegrationPlan:
        return HFGenerateIntegrationPlan(
            family=self.family,
            adapter=self.__class__.__name__,
            cache_position=self.capabilities.cache_positions and _generate_has_named_kwarg(self.model, "cache_position"),
            attention_mask=self.capabilities.attention_mask and _generate_accepts_kwarg(self.model, "attention_mask"),
            sliding_window=self.capabilities.sliding_window and _generate_has_named_kwarg(self.model, "sliding_window"),
            grouped_query_attention=self.capabilities.grouped_query_attention,
            fallback_reason=self.fallback_reason(input_ids),
        )

    def fallback_reason(self, input_ids: Any | None = None, **kwargs: Any) -> str | None:
        del input_ids, kwargs
        return None

    def generate(self, input_ids, **generate_kwargs) -> MemoryFirstGenerateResult:
        kwargs = self.prepare_generate_kwargs(input_ids, **generate_kwargs)
        cache = kwargs.get("past_key_values")
        max_new_tokens = int(kwargs.get("max_new_tokens", self.config.max_new_tokens))
        try:
            sequences = self.model.generate(input_ids=input_ids, **kwargs)
            return MemoryFirstGenerateResult(
                sequences,
                cache,
                max_new_tokens,
                f"{self.family}:{self.config.cache}",
                direct_cache=True,
                requested_cache_impl=f"{self.family}:{self.config.cache}",
            )
        except TypeError as exc:
            if cache is not None or _is_generate_injection_error(exc):
                fallback_kwargs = _fallback_generate_kwargs(generate_kwargs, max_new_tokens)
                sequences = self.model.generate(input_ids=input_ids, **fallback_kwargs)
                return MemoryFirstGenerateResult(
                    sequences, None, max_new_tokens, f"{self.family}:original-fallback", fallback_reason=str(exc)[:240]
                )
            return memory_first_generate(self.model, input_ids, self.config, **_safe_model_kwargs(generate_kwargs))
        except (ValueError, AttributeError, RuntimeError) as exc:
            if not _is_generate_injection_error(exc):
                raise
            fallback_kwargs = _fallback_generate_kwargs(generate_kwargs, max_new_tokens)
            sequences = self.model.generate(input_ids=input_ids, **fallback_kwargs)
            return MemoryFirstGenerateResult(
                sequences, None, max_new_tokens, f"{self.family}:original-fallback", fallback_reason=str(exc)[:240]
            )


class LlamaMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "llama"
    capabilities = FamilyAdapterCapabilities(family="llama", quirks=("cache_position-aware", "standard RoPE"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        return kwargs


class QwenMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "qwen"
    capabilities = FamilyAdapterCapabilities(family="qwen", quirks=("qwen cache api", "GQA common"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        return kwargs


class Qwen3MemoryAdapter(QwenMemoryAdapter):
    family = "qwen3"
    capabilities = FamilyAdapterCapabilities(
        family="qwen3",
        quirks=(
            "Transformers Cache API compatible",
            "GQA common",
            "top-level cache_position rejected by generate",
            "direct cache must pass per-model certification before production rollout",
        ),
    )

    def generate(self, input_ids, **generate_kwargs) -> MemoryFirstGenerateResult:
        if not self.config.allow_experimental_direct_cache:
            max_new_tokens = int(generate_kwargs.get("max_new_tokens", self.config.max_new_tokens))
            fallback_kwargs = _fallback_generate_kwargs(generate_kwargs, max_new_tokens)
            sequences = self.model.generate(input_ids=input_ids, **fallback_kwargs)
            return MemoryFirstGenerateResult(
                sequences,
                None,
                max_new_tokens,
                "qwen3:quality-gated-fallback",
                fallback_reason="qwen3 direct cache is not quality-certified; use explicit certification or --allow-experimental-direct-cache",
                requested_cache_impl=f"qwen3:{self.config.cache}",
            )
        if self._should_fallback_quantized_to_paged(input_ids):
            fallback_config = replace(self.config, cache="paged")
            result = Qwen3MemoryAdapter(self.model, fallback_config).generate(input_ids, **generate_kwargs)
            result.requested_cache_impl = f"qwen3:{self.config.cache}:{self.config.quant_dtype}"
            result.fallback_reason = (
                "qwen3 quantized direct cache is not production-certified for multi-token prefill; used paged direct cache"
            )
            result.cache_impl = "qwen3:paged-policy-fallback"
            return result
        return super().generate(input_ids, **generate_kwargs)

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        return kwargs

    def _should_fallback_quantized_to_paged(self, input_ids) -> bool:
        if self.config.cache != "quantized":
            return False
        policy = HFCachePolicy(
            requested_cache="quantized",
            quant_dtype=self.config.quant_dtype,
            model=self.config.model_name,
            qwen3_quantized_prefill_limit=self.config.quantized_prefill_token_limit,
            allow_experimental_quantized=self.config.allow_experimental_quantized_cache,
            quantized_profiles=self.config.quantized_profiles,
        )
        decision = select_hf_cache_policy(
            family=self.family, prompt_tokens=int(input_ids.shape[-1]), device=input_ids.device, policy=policy
        )
        return decision.cache != "quantized"

    def fallback_reason(self, input_ids: Any | None = None, **kwargs: Any) -> str | None:
        del kwargs
        if input_ids is not None and self._should_fallback_quantized_to_paged(input_ids):
            return "qwen3 quantized direct cache is not certified for this prompt/profile"
        if not self.config.allow_experimental_direct_cache:
            return "qwen3 direct cache requires explicit certification or experimental opt-in"
        return None


class MistralMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "mistral"
    capabilities = FamilyAdapterCapabilities(family="mistral", sliding_window=True, quirks=("sliding-window attention", "GQA common"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        sliding_window = getattr(getattr(self.model, "config", None), "sliding_window", None)
        if sliding_window is not None:
            kwargs.setdefault("sliding_window", sliding_window)
        return kwargs


class MixtralMemoryAdapter(MistralMemoryAdapter):
    family = "mixtral"
    capabilities = FamilyAdapterCapabilities(
        family="mixtral", sliding_window=True, quirks=("mixture-of-experts", "mistral-compatible cache semantics")
    )


class GemmaMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "gemma"
    capabilities = FamilyAdapterCapabilities(family="gemma", quirks=("gemma/gemma2/gemma3 cache API", "RoPE variant"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        return kwargs


class PhiMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "phi"
    capabilities = FamilyAdapterCapabilities(family="phi", quirks=("phi3/phi4 cache API", "small-model friendly"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        return kwargs


class DeepSeekMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "deepseek"
    capabilities = FamilyAdapterCapabilities(family="deepseek", quirks=("deepseek cache API", "GQA/MLA variants require certification"))

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        _ensure_cache_position(input_ids, kwargs)
        return kwargs


class GPTNeoXMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "gpt_neox"
    capabilities = FamilyAdapterCapabilities(family="gpt_neox", quirks=("legacy GPT-NeoX cache layout",), grouped_query_attention=False)

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        return kwargs


class FalconMemoryAdapter(BaseFamilyMemoryAdapter):
    family = "falcon"
    capabilities = FamilyAdapterCapabilities(family="falcon", quirks=("falcon multi-query attention",), grouped_query_attention=True)

    def _prepare_family_kwargs(self, input_ids, kwargs: dict[str, Any]) -> dict[str, Any]:
        _ensure_attention_mask(input_ids, kwargs)
        return kwargs


_ADAPTER_REGISTRY: list[tuple[str, type[BaseFamilyMemoryAdapter]]] = []


def register_memory_adapter(prefix: str, adapter_cls: type[BaseFamilyMemoryAdapter]) -> None:
    if not issubclass(adapter_cls, BaseFamilyMemoryAdapter):
        raise TypeError("adapter_cls must inherit BaseFamilyMemoryAdapter")
    normalized = prefix.lower()
    _ADAPTER_REGISTRY[:] = [(item, cls) for item, cls in _ADAPTER_REGISTRY if item != normalized]
    _ADAPTER_REGISTRY.append((normalized, adapter_cls))
    _ADAPTER_REGISTRY.sort(key=lambda item: len(item[0]), reverse=True)


def list_memory_adapters() -> tuple[tuple[str, type[BaseFamilyMemoryAdapter]], ...]:
    return tuple(_ADAPTER_REGISTRY)


class MemoryFirstHFAdapter:
    """Custom generate loop that stores HF legacy past_key_values in llm-memlab caches."""

    def __init__(self, model: Any, config: MemoryFirstHFConfig | None = None):
        self.model = model
        self.config = config or MemoryFirstHFConfig()

    def generate(self, input_ids, **model_kwargs) -> MemoryFirstGenerateResult:
        torch = _import_torch()
        cfg = self.config
        sequences = input_ids.clone()
        cache = None
        next_input = input_ids
        steps = 0
        with torch.no_grad():
            for index in range(cfg.max_new_tokens):
                kwargs = dict(model_kwargs)
                kwargs["use_cache"] = True
                if cache is not None:
                    kwargs["past_key_values"] = cache.as_legacy_cache()
                outputs = self.model(next_input, **kwargs)
                logits = _get_logits(outputs)
                past = _get_past_key_values(outputs)
                if past is not None:
                    cache = _cache_from_past(past, cache, cfg, max_seq_len=input_ids.shape[-1] + cfg.max_new_tokens)
                next_token = sample_next_token(logits[:, -1, :], temperature=cfg.temperature, top_k=cfg.top_k)
                sequences = torch.cat([sequences, next_token[:, None]], dim=-1)
                next_input = next_token[:, None]
                steps = index + 1
        return MemoryFirstGenerateResult(sequences, cache, steps, cfg.cache)


def memory_first_generate(model: Any, input_ids, config: MemoryFirstHFConfig | None = None, **model_kwargs) -> MemoryFirstGenerateResult:
    return MemoryFirstHFAdapter(model, config).generate(input_ids, **model_kwargs)


def select_memory_adapter(model: Any, config: MemoryFirstHFConfig | None = None) -> BaseFamilyMemoryAdapter:
    family = _model_family(model)
    for prefix, adapter_cls in _ADAPTER_REGISTRY:
        if family.startswith(prefix):
            return adapter_cls(model, config)
    return BaseFamilyMemoryAdapter(model, config)


def detect_hf_adapter_info(model: Any) -> HFAdapterInfo:
    family = _model_family(model)
    direct = any(family.startswith(prefix) for prefix, _ in _ADAPTER_REGISTRY)
    strategy = "family-cache-api" if direct else "generic-cache-api"
    return HFAdapterInfo(family=family, supports_cache_api=supports_transformers_cache_api(), strategy=strategy)


def supports_transformers_cache_api() -> bool:
    try:
        from transformers.cache_utils import Cache  # noqa: F401
    except Exception:
        return False
    return True


def make_transformers_cache_from_model(
    model: Any, config: MemoryFirstHFConfig, *, batch_size: int, max_seq_len: int, dtype=None, device=None
) -> MemoryFirstTransformersCache:
    model_config = getattr(model, "config", None)
    num_layers = int(getattr(model_config, "num_hidden_layers", getattr(model_config, "n_layer", 1)))
    num_heads = int(
        getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", getattr(model_config, "n_head", 1)))
    )
    hidden_size = int(getattr(model_config, "hidden_size", num_heads * 64))
    head_dim = int(getattr(model_config, "head_dim", hidden_size // max(num_heads, 1)))
    if dtype is None or str(dtype).endswith("int64") or str(dtype).endswith("long"):
        dtype = _model_dtype(model)
    kv_config = KVCacheConfig(
        num_layers=num_layers,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )
    return MemoryFirstTransformersCache(kv_config, cache=config.cache, quant_dtype=config.quant_dtype, page_size=config.page_size)


def memory_first_generate_hf(
    model: Any, input_ids, config: MemoryFirstHFConfig | None = None, **generate_kwargs
) -> MemoryFirstGenerateResult:
    return select_memory_adapter(model, config).generate(input_ids, **generate_kwargs)


def install_memory_first_generate(model: Any, config: MemoryFirstHFConfig | None = None):
    """Patch model.generate to inject a family-specific MemoryFirstTransformersCache when possible."""

    cfg = config or MemoryFirstHFConfig()
    original_generate = model.generate
    adapter = select_memory_adapter(model, cfg)

    def generate_with_memory_first(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is not None and kwargs.get("use_cache", True) and "past_key_values" not in kwargs:
            kwargs.update(adapter.prepare_generate_kwargs(input_ids, **kwargs))
        try:
            return original_generate(*args, **kwargs)
        except (TypeError, ValueError, AttributeError) as exc:
            if not _is_generate_injection_error(exc):
                raise
            kwargs.pop("past_key_values", None)
            kwargs.pop("cache_position", None)
            kwargs.pop("sliding_window", None)
            return original_generate(*args, **kwargs)

    model.generate = generate_with_memory_first
    return model


def _cache_from_past(past, existing, cfg: MemoryFirstHFConfig, *, max_seq_len: int):
    first_k, _ = past[0]
    config = KVCacheConfig(
        num_layers=len(past),
        batch_size=first_k.shape[0],
        num_heads=first_k.shape[1],
        head_dim=first_k.shape[-1],
        max_seq_len=max_seq_len,
        dtype=first_k.dtype,
        device=first_k.device,
    )
    if existing is None:
        existing = (
            PagedKVCache(config, page_size=cfg.page_size)
            if cfg.cache == "paged"
            else QuantizedStaticKVCache(config, quant_dtype=cfg.quant_dtype)
        )
    length = int(first_k.shape[-2])
    existing.reset()
    for layer_idx, (key, value) in enumerate(past):
        existing.append_layer(layer_idx, key[:, :, :length, :], value[:, :, :length, :], position=0)
    return existing


def _ensure_attention_mask(input_ids, kwargs: dict[str, Any]) -> None:
    if "attention_mask" not in kwargs and hasattr(input_ids, "new_ones"):
        kwargs["attention_mask"] = input_ids.new_ones(input_ids.shape)


def _ensure_cache_position(input_ids, kwargs: dict[str, Any]) -> None:
    if "cache_position" in kwargs or not hasattr(input_ids, "device"):
        return
    torch = _import_torch()
    kwargs["cache_position"] = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long)


def _cache_position_from_kwargs(cache_kwargs: dict[str, Any] | None) -> int | None:
    if not cache_kwargs:
        return None
    cache_position = cache_kwargs.get("cache_position")
    if cache_position is not None and hasattr(cache_position, "numel") and cache_position.numel() > 0:
        return int(cache_position.flatten()[0].item())
    return None


def _fallback_generate_kwargs(kwargs: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    payload = _safe_model_kwargs(kwargs)
    payload.setdefault("max_new_tokens", max_new_tokens)
    payload.setdefault("use_cache", True)
    return payload


def _is_generate_injection_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "past_key_values",
            "past key",
            "cache_position",
            "sliding_window",
            "model_kwargs",
            "cache object",
            "cache",
            "device",
            "shape",
            "size mismatch",
            "expected all tensors",
            "same device",
            "indices should be",
            "invalid for input",
        )
    )


def _safe_model_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in kwargs.items() if key not in {"max_new_tokens", "past_key_values", "cache_position", "sliding_window"}
    }


def _filter_generate_kwargs(model: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    if _generate_accepts_var_kwargs(model):
        payload = dict(kwargs)
        for key in ("cache_position", "sliding_window"):
            if key in payload and not _generate_has_named_kwarg(model, key):
                payload.pop(key, None)
        return payload
    payload = dict(kwargs)
    for key in ("cache_position", "sliding_window", "attention_mask"):
        if key in payload and not _generate_accepts_kwarg(model, key):
            payload.pop(key, None)
    return payload


def _generate_has_named_kwarg(model: Any, name: str) -> bool:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError, AttributeError):
        return False
    return name in signature.parameters


def _generate_accepts_kwarg(model: Any, name: str) -> bool:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError, AttributeError):
        return True
    return name in signature.parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _generate_accepts_var_kwargs(model: Any) -> bool:
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError, AttributeError):
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise TypeError("Model output must expose logits")


def _get_past_key_values(outputs):
    if isinstance(outputs, dict):
        return outputs.get("past_key_values")
    if hasattr(outputs, "past_key_values"):
        return outputs.past_key_values
    if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
        return outputs[1]
    return None


def _model_family(model: Any) -> str:
    return str(getattr(getattr(model, "config", None), "model_type", model.__class__.__name__)).lower()


def _model_dtype(model: Any):
    try:
        return next(model.parameters()).dtype
    except Exception:
        torch = _import_torch()
        return torch.float16 if torch.cuda.is_available() else torch.float32


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MemoryFirstHFAdapter requires PyTorch") from exc
    return torch


for _prefix, _adapter in (
    ("qwen3", Qwen3MemoryAdapter),
    ("qwen2", QwenMemoryAdapter),
    ("qwen", QwenMemoryAdapter),
    ("llama", LlamaMemoryAdapter),
    ("mistral", MistralMemoryAdapter),
    ("mixtral", MixtralMemoryAdapter),
    ("gemma3", GemmaMemoryAdapter),
    ("gemma2", GemmaMemoryAdapter),
    ("gemma", GemmaMemoryAdapter),
    ("phi4", PhiMemoryAdapter),
    ("phi3", PhiMemoryAdapter),
    ("phi", PhiMemoryAdapter),
    ("deepseek", DeepSeekMemoryAdapter),
    ("gpt_neox", GPTNeoXMemoryAdapter),
    ("gpt-neox", GPTNeoXMemoryAdapter),
    ("falcon", FalconMemoryAdapter),
):
    register_memory_adapter(_prefix, _adapter)


def adapter_satisfies_contract(adapter: Any) -> bool:
    return isinstance(adapter, MemoryAdapterProtocol)
