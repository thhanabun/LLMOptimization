from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryAdapterProtocol(Protocol):
    family: str
    capabilities: Any

    @classmethod
    def detect(cls, model: Any) -> bool:
        """Return True when this adapter can handle the model."""

    def make_cache(self, input_ids: Any, *, max_new_tokens: int | None = None) -> Any:
        """Create a cache object compatible with the target model family."""

    def prepare_generate_kwargs(self, input_ids: Any, **generate_kwargs: Any) -> dict[str, Any]:
        """Return generate kwargs with cache/mask/position arguments prepared."""

    def generate(self, input_ids: Any, **generate_kwargs: Any) -> Any:
        """Run generation through this adapter."""

    def certify(self, input_ids: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        """Return adapter-level certification hints for policy/reporting."""

    def fallback_reason(self, input_ids: Any | None = None, **kwargs: Any) -> str | None:
        """Explain why the adapter would fall back for this input, if known."""
