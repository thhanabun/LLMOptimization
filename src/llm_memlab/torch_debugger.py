from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .bytes import format_bytes
from .report import make_table


@dataclass
class ModuleRecord:
    name: str
    type_name: str
    depth: int
    elapsed_ms: float
    output_bytes: int
    parameter_bytes: int
    cuda_before_bytes: int | None
    cuda_after_bytes: int | None
    cuda_delta_bytes: int | None
    input_shapes: str
    output_shapes: str
    has_nan: bool = False
    has_inf: bool = False


@dataclass
class GradientRecord:
    name: str
    norm: float
    max_abs: float
    has_nan: bool
    has_inf: bool


@dataclass
class TorchTrace:
    model: Any
    record_leaf_only: bool = True
    records: list[ModuleRecord] = field(default_factory=list)
    gradients: list[GradientRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._torch = _import_torch()
        self._handles: list[Any] = []
        self._starts: dict[int, tuple[float, int | None]] = {}
        self._names: dict[int, str] = {}
        self._depths: dict[int, int] = {}

    def __enter__(self) -> "TorchTrace":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._handles:
            return
        for name, module in self.model.named_modules():
            if self.record_leaf_only and any(True for _ in module.children()):
                continue
            module_name = name or "<root>"
            self._names[id(module)] = module_name
            self._depths[id(module)] = module_name.count(".")
            self._handles.append(module.register_forward_pre_hook(self._pre_hook))
            self._handles.append(module.register_forward_hook(self._post_hook))

    def stop(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def attach_gradient_monitor(self) -> None:
        for name, param in self.model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            self._handles.append(param.register_hook(self._make_grad_hook(name)))

    def to_text(self, *, limit: int | None = None) -> str:
        records = self.records[:limit] if limit is not None else self.records
        rows = [
            (
                rec.name,
                rec.type_name,
                f"{rec.elapsed_ms:.3f}",
                format_bytes(rec.output_bytes),
                format_bytes(rec.parameter_bytes),
                _fmt_optional_bytes(rec.cuda_delta_bytes),
                "yes" if rec.has_nan else "",
                "yes" if rec.has_inf else "",
            )
            for rec in records
        ]
        text = [
            make_table(
                ("Module", "Type", "ms", "Output", "Params", "CUDA delta", "NaN", "Inf"),
                rows,
            )
        ]
        if self.gradients:
            grad_rows = [
                (
                    grad.name,
                    f"{grad.norm:.4g}",
                    f"{grad.max_abs:.4g}",
                    "yes" if grad.has_nan else "",
                    "yes" if grad.has_inf else "",
                )
                for grad in self.gradients
            ]
            text.extend(["", "Gradients", make_table(("Param", "L2 norm", "Max abs", "NaN", "Inf"), grad_rows)])
        return "\n".join(text)

    def _pre_hook(self, module, inputs) -> None:
        self._starts[id(module)] = (time.perf_counter(), _cuda_allocated(self._torch))

    def _post_hook(self, module, inputs, output) -> None:
        start, cuda_before = self._starts.pop(id(module), (time.perf_counter(), _cuda_allocated(self._torch)))
        cuda_after = _cuda_allocated(self._torch)
        cuda_delta = None
        if cuda_before is not None and cuda_after is not None:
            cuda_delta = cuda_after - cuda_before
        elapsed_ms = (time.perf_counter() - start) * 1000
        has_nan, has_inf = _nonfinite_flags(self._torch, output)
        self.records.append(
            ModuleRecord(
                name=self._names.get(id(module), module.__class__.__name__),
                type_name=module.__class__.__name__,
                depth=self._depths.get(id(module), 0),
                elapsed_ms=elapsed_ms,
                output_bytes=_tree_nbytes(output),
                parameter_bytes=sum(param.numel() * param.element_size() for param in module.parameters(recurse=False)),
                cuda_before_bytes=cuda_before,
                cuda_after_bytes=cuda_after,
                cuda_delta_bytes=cuda_delta,
                input_shapes=_shape_summary(inputs),
                output_shapes=_shape_summary(output),
                has_nan=has_nan,
                has_inf=has_inf,
            )
        )

    def _make_grad_hook(self, name: str):
        def hook(grad):
            detached = grad.detach()
            finite = self._torch.isfinite(detached)
            self.gradients.append(
                GradientRecord(
                    name=name,
                    norm=float(detached.float().norm().item()),
                    max_abs=float(detached.float().abs().max().item()),
                    has_nan=bool(self._torch.isnan(detached).any().item()),
                    has_inf=bool(self._torch.isinf(detached).any().item()),
                )
            )
            return grad

        return hook


def trace_forward(model, *args, record_leaf_only: bool = True, **kwargs) -> tuple[Any, TorchTrace]:
    with TorchTrace(model, record_leaf_only=record_leaf_only) as trace:
        output = model(*args, **kwargs)
    return output, trace


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for llm_memlab.torch_debugger. Install with: pip install torch") from exc
    return torch


def _cuda_allocated(torch_module) -> int | None:
    if not torch_module.cuda.is_available():
        return None
    return int(torch_module.cuda.memory_allocated())


def _tree_nbytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_nbytes(item) for item in value)
    return 0


def _shape_summary(value: Any) -> str:
    if hasattr(value, "shape"):
        dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
        return f"{tuple(value.shape)}:{dtype}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key}: {_shape_summary(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_shape_summary(item) for item in value) + "]"
    return type(value).__name__


def _nonfinite_flags(torch_module, value: Any) -> tuple[bool, bool]:
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        return (
            bool(torch_module.isnan(value).any().item()),
            bool(torch_module.isinf(value).any().item()),
        )
    if isinstance(value, dict):
        flags = [_nonfinite_flags(torch_module, item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        flags = [_nonfinite_flags(torch_module, item) for item in value]
    else:
        flags = []
    return any(item[0] for item in flags), any(item[1] for item in flags)


def _fmt_optional_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return format_bytes(value)
