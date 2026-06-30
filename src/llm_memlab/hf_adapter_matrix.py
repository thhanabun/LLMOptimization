from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hf_adapter import (
    BaseFamilyMemoryAdapter,
    DeepSeekMemoryAdapter,
    FalconMemoryAdapter,
    GemmaMemoryAdapter,
    GPTNeoXMemoryAdapter,
    LlamaMemoryAdapter,
    MistralMemoryAdapter,
    MixtralMemoryAdapter,
    PhiMemoryAdapter,
    Qwen3MemoryAdapter,
    QwenMemoryAdapter,
    supports_transformers_cache_api,
)
from .report import make_table

HF_ADAPTER_MATRIX_SCHEMA_VERSION = "llm_memlab.hf_adapter_matrix.v1"


@dataclass(frozen=True)
class HFAdapterMatrixEntry:
    family: str
    adapter: str
    cache_api: bool
    cache_position: bool
    attention_mask: bool
    gqa: bool
    sliding_window: bool
    production_default_cache: str
    quantized_direct: str
    notes: tuple[str, ...] = ()
    schema_version: str = HF_ADAPTER_MATRIX_SCHEMA_VERSION


def production_hf_adapter_matrix(*, transformers_version: str | None = None) -> tuple[HFAdapterMatrixEntry, ...]:
    cache_api = supports_transformers_cache_api()
    version_note = (f"transformers={transformers_version}",) if transformers_version else ()
    return (
        _entry("llama", LlamaMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("qwen", QwenMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("qwen3", Qwen3MemoryAdapter, cache_api, "paged", "fallback-until-certified", version_note),
        _entry("mistral", MistralMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("mixtral", MixtralMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("gemma", GemmaMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("phi", PhiMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("deepseek", DeepSeekMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("gpt_neox", GPTNeoXMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("falcon", FalconMemoryAdapter, cache_api, "paged", "certification-required", version_note),
        _entry("generic", BaseFamilyMemoryAdapter, cache_api, "paged", "disabled-by-default", version_note),
    )


def hf_adapter_matrix_to_text(entries: tuple[HFAdapterMatrixEntry, ...] | list[HFAdapterMatrixEntry]) -> str:
    return make_table(
        ("Family", "Adapter", "Cache API", "Default", "Quantized", "GQA", "Sliding", "Notes"),
        [
            (
                item.family,
                item.adapter,
                item.cache_api,
                item.production_default_cache,
                item.quantized_direct,
                item.gqa,
                item.sliding_window,
                "; ".join(item.notes),
            )
            for item in entries
        ],
    )


def _entry(
    family: str,
    adapter_cls: type[Any],
    cache_api: bool,
    default_cache: str,
    quantized_direct: str,
    notes: tuple[str, ...],
) -> HFAdapterMatrixEntry:
    caps = getattr(adapter_cls, "capabilities")
    return HFAdapterMatrixEntry(
        family=family,
        adapter=adapter_cls.__name__,
        cache_api=cache_api,
        cache_position=bool(caps.cache_positions),
        attention_mask=bool(caps.attention_mask),
        gqa=bool(caps.grouped_query_attention),
        sliding_window=bool(caps.sliding_window),
        production_default_cache=default_cache,
        quantized_direct=quantized_direct,
        notes=tuple(caps.quirks) + tuple(notes),
    )
