from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .report import make_table


@dataclass(frozen=True)
class OOMStrategy:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OOMRunResult:
    value: Any
    strategy: OOMStrategy
    attempts: tuple[str, ...]

    def to_text(self) -> str:
        return make_table(("Metric", "Value"), [
            ("Selected strategy", self.strategy.name),
            ("Attempts", ", ".join(self.attempts)),
        ])


def run_with_oom_fallback(fn: Callable[..., Any], strategies: list[OOMStrategy]) -> OOMRunResult:
    errors: list[str] = []
    for strategy in strategies:
        try:
            return OOMRunResult(fn(**strategy.kwargs), strategy, tuple([*errors, strategy.name]))
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            errors.append(f"{strategy.name}: OOM")
    raise RuntimeError("All OOM fallback strategies failed: " + "; ".join(errors))


def is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def default_generation_strategies(policy=None) -> list[OOMStrategy]:
    strategies = [OOMStrategy("policy-default", {})]
    if policy is not None and getattr(policy, "use_quantized_cache", False):
        strategies.append(OOMStrategy("disable-cache", {"use_cache": False}))
    strategies.extend([
        OOMStrategy("no-cache", {"use_cache": False}),
        OOMStrategy("cpu-friendly", {"use_cache": False}),
    ])
    return strategies
