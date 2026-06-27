from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .bytes import parse_bytes
from .report import make_table


@dataclass(frozen=True)
class MemoryPolicy:
    max_vram_bytes: int
    kv_dtype: str
    cache_impl: str
    use_quantized_cache: bool
    use_paged_cache: bool
    page_size: int
    use_chunked_lm_head: bool
    recommended_chunk_size: int
    attention_backend: str
    notes: tuple[str, ...]

    def to_text(self) -> str:
        rows = [
            ("Max VRAM", self.max_vram_bytes),
            ("KV dtype", self.kv_dtype),
            ("Cache impl", self.cache_impl),
            ("Quantized cache", self.use_quantized_cache),
            ("Paged cache", self.use_paged_cache),
            ("Page size", self.page_size),
            ("Chunked LM head", self.use_chunked_lm_head),
            ("Chunk size", self.recommended_chunk_size),
            ("Attention backend", self.attention_backend),
        ]
        text = [make_table(("Policy", "Value"), rows)]
        if self.notes:
            text.extend(["", "Notes"])
            text.extend(f"- {note}" for note in self.notes)
        return "\n".join(text)


def choose_memory_policy(
    *,
    max_vram: str | int,
    model_info: Any | None = None,
    sequence_length: int | None = None,
    prefer_speed: bool = True,
) -> MemoryPolicy:
    max_bytes = parse_bytes(max_vram) if isinstance(max_vram, str) else int(max_vram)
    kv_fp16 = getattr(model_info, "kv_cache_bytes_fp16", None) if model_info is not None else None
    notes: list[str] = []
    kv_dtype = "fp16"
    cache_impl = "static"
    use_quantized = False
    use_paged = False
    page_size = 16
    chunk_size = 1024
    attention_backend = "sdpa"

    if kv_fp16 is not None:
        budget_for_kv = max_bytes * 0.35
        if kv_fp16 > budget_for_kv:
            kv_dtype = "int8"
            use_quantized = True
            notes.append("KV cache fp16 estimate exceeds 35% of the VRAM budget; int8 KV is recommended.")
        if kv_fp16 > max_bytes * 0.2 or (sequence_length or 0) >= 4096:
            cache_impl = "paged"
            use_paged = True
            page_size = 32 if (sequence_length or 0) >= 4096 else 16
            notes.append("Paged cache is recommended to make cache allocation explicit and reusable.")
    else:
        notes.append("Model KV estimate is unavailable; using conservative fp16 static cache defaults.")

    if max_bytes <= parse_bytes("8GB"):
        chunk_size = 256
        notes.append("Small VRAM budget: prefer chunked LM-head/loss operations.")
    elif max_bytes <= parse_bytes("16GB"):
        chunk_size = 512

    if prefer_speed and not use_quantized:
        attention_backend = "sdpa/flash-if-available"
    elif use_quantized:
        attention_backend = "dequant+sdpa"
        notes.append("Quantized KV saves memory but may trade some decode latency until fused kernels are available.")

    return MemoryPolicy(
        max_vram_bytes=max_bytes,
        kv_dtype=kv_dtype,
        cache_impl=cache_impl,
        use_quantized_cache=use_quantized,
        use_paged_cache=use_paged,
        page_size=page_size,
        use_chunked_lm_head=True,
        recommended_chunk_size=chunk_size,
        attention_backend=attention_backend,
        notes=tuple(notes),
    )
