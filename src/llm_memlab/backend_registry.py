from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Callable

from .report import make_table


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    priority: int
    reason: str = ""
    kind: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendPlugin:
    name: str
    check: Callable[[], tuple[bool, str]]
    priority: int = 0
    kind: str = "plugin"
    metadata: dict[str, Any] = field(default_factory=dict)


class BackendRegistry:
    def __init__(self):
        self._checks: dict[str, BackendPlugin] = {}

    def register(
        self,
        name: str,
        check: Callable[[], tuple[bool, str]],
        *,
        priority: int = 0,
        kind: str = "runtime",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.register_plugin(BackendPlugin(name=name, check=check, priority=priority, kind=kind, metadata=metadata or {}))

    def register_plugin(self, plugin: BackendPlugin) -> None:
        self._checks[plugin.name] = plugin

    def get(self, name: str) -> BackendInfo:
        plugin = self._checks[name]
        try:
            available, reason = plugin.check()
        except Exception as exc:
            available, reason = False, str(exc)
        return BackendInfo(
            name=name,
            available=available,
            priority=plugin.priority,
            reason=reason,
            kind=plugin.kind,
            metadata=dict(plugin.metadata),
        )

    def list(self) -> tuple[BackendInfo, ...]:
        return tuple(sorted((self.get(name) for name in self._checks), key=lambda item: item.priority, reverse=True))

    def best(self, *names: str) -> BackendInfo:
        candidates = [self.get(name) for name in names if name in self._checks]
        available = [item for item in candidates if item.available]
        if available:
            return sorted(available, key=lambda item: item.priority, reverse=True)[0]
        if candidates:
            return sorted(candidates, key=lambda item: item.priority, reverse=True)[0]
        raise KeyError(f"No registered backend among {names!r}")

    def to_text(self) -> str:
        rows = [(item.name, item.available, item.priority, item.reason) for item in self.list()]
        return make_table(("Backend", "Available", "Priority", "Reason"), rows)


def default_backend_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register("torch", _check_torch, priority=10)
    registry.register("triton", _check_triton, priority=30)
    registry.register("cuda", _check_cuda, priority=20)
    registry.register("cutile", _check_cutile, priority=25)
    registry.register("flash-attn", lambda: _check_import("flash_attn", "flash-attn"), priority=28, kind="optional")
    registry.register("xformers", lambda: _check_import("xformers", "xformers"), priority=24, kind="optional")
    registry.register("vllm", lambda: _check_import("vllm", "vllm"), priority=22, kind="optional")
    registry.register("bitsandbytes", lambda: _check_import("bitsandbytes", "bitsandbytes"), priority=18, kind="optional")
    load_backend_entrypoints(registry)
    return registry


def load_backend_entrypoints(registry: BackendRegistry, *, group: str = "llm_memlab.backends") -> BackendRegistry:
    try:
        entry_points = metadata.entry_points()
        selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, ())
    except Exception:
        return registry
    for entry_point in selected:
        try:
            plugin = entry_point.load()
            if isinstance(plugin, BackendPlugin):
                registry.register_plugin(plugin)
            elif callable(plugin):
                produced = plugin()
                if isinstance(produced, BackendPlugin):
                    registry.register_plugin(produced)
        except Exception:
            continue
    return registry


def _check_torch() -> tuple[bool, str]:
    try:
        import torch

        return True, f"torch {torch.__version__}"
    except Exception as exc:
        return False, str(exc)


def _check_triton() -> tuple[bool, str]:
    try:
        from .triton_kernels import triton_available

        ok = triton_available()
        return ok, "triton import ok" if ok else "triton is not installed"
    except Exception as exc:
        return False, str(exc)


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch

        ok = torch.cuda.is_available()
        return ok, torch.cuda.get_device_name(0) if ok else "CUDA is not available"
    except Exception as exc:
        return False, str(exc)


def _check_cutile() -> tuple[bool, str]:
    try:
        from .backends.cutile import detect_cutile_runtime

        info = detect_cutile_runtime()
        return info.available, "; ".join(info.reasons)
    except Exception as exc:
        return False, str(exc)


def _check_import(module: str, label: str) -> tuple[bool, str]:
    try:
        imported = __import__(module)
        version = getattr(imported, "__version__", "unknown")
        return True, f"{label} {version}"
    except Exception as exc:
        return False, str(exc)
