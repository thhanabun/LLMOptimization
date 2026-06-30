from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_policy import MemoryPolicy
from .report import make_table


@dataclass(frozen=True)
class HFCachePlan:
    use_cache: bool
    cache_implementation: str | None
    notes: tuple[str, ...]

    def generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"use_cache": self.use_cache}
        if self.cache_implementation is not None:
            kwargs["cache_implementation"] = self.cache_implementation
        return kwargs

    def to_text(self) -> str:
        rows = [("use_cache", self.use_cache), ("cache_implementation", self.cache_implementation or "default")]
        text = [make_table(("HF cache", "Value"), rows)]
        if self.notes:
            text.extend(["", "Notes"])
            text.extend(f"- {note}" for note in self.notes)
        return "\n".join(text)


def plan_hf_cache(policy: MemoryPolicy, model: Any | None = None) -> HFCachePlan:
    notes: list[str] = []
    cache_impl = None
    use_cache = True
    if policy.use_paged_cache:
        cache_impl = "dynamic"
        notes.append("HF native paged cache is model/version dependent; using dynamic cache hint when accepted.")
    if policy.use_quantized_cache:
        notes.append(
            "llm_memlab quantized KV cache is tracked as a policy recommendation; HF generate may ignore it without a model-specific cache adapter."
        )
    if model is not None and not hasattr(model, "generate"):
        use_cache = False
        notes.append("Model has no generate(); disabling cache hints.")
    return HFCachePlan(use_cache=use_cache, cache_implementation=cache_impl, notes=tuple(notes))
