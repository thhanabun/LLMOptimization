from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .report import make_table


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    priority: int
    reason: str = ""


class BackendRegistry:
    def __init__(self):
        self._checks: dict[str, tuple[int, Callable[[], tuple[bool, str]]]] = {}

    def register(self, name: str, check: Callable[[], tuple[bool, str]], *, priority: int = 0) -> None:
        self._checks[name] = (priority, check)

    def get(self, name: str) -> BackendInfo:
        priority, check = self._checks[name]
        try:
            available, reason = check()
        except Exception as exc:
            available, reason = False, str(exc)
        return BackendInfo(name=name, available=available, priority=priority, reason=reason)

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
