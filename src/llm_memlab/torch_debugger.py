from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .bytes import format_bytes
from .report import make_table


@dataclass
class TensorStats:
    shape: str
    dtype: str
    device: str
    bytes: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    zero_pct: float | None = None
    has_nan: bool = False
    has_inf: bool = False


@dataclass
class ModuleRecord:
    name: str
    type_name: str
    depth: int
    elapsed_ms: float
    input_bytes: int
    output_bytes: int
    parameter_bytes: int
    parameter_count: int
    trainable_parameter_count: int
    cuda_before_bytes: int | None
    cuda_after_bytes: int | None
    cuda_delta_bytes: int | None
    input_shapes: str
    output_shapes: str
    output_stats: TensorStats | None = None
    has_nan: bool = False
    has_inf: bool = False


@dataclass
class GradientRecord:
    name: str
    norm: float
    max_abs: float
    mean: float
    std: float
    has_nan: bool
    has_inf: bool


@dataclass
class TorchTrace:
    model: Any
    record_leaf_only: bool = True
    collect_stats: bool = True
    records: list[ModuleRecord] = field(default_factory=list)
    gradients: list[GradientRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._torch = _import_torch()
        self._handles: list[Any] = []
        self._started = False
        self._starts: dict[int, tuple[float, int | None]] = {}
        self._names: dict[int, str] = {}
        self._depths: dict[int, int] = {}

    def __enter__(self) -> "TorchTrace":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        self._started = True
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
        self._started = False

    def attach_gradient_monitor(self) -> None:
        for name, param in self.model.named_parameters():
            if not getattr(param, "requires_grad", False):
                continue
            self._handles.append(param.register_hook(self._make_grad_hook(name)))

    @property
    def total_ms(self) -> float:
        return sum(record.elapsed_ms for record in self.records)

    def slowest(self, limit: int = 8) -> list[ModuleRecord]:
        return sorted(self.records, key=lambda record: record.elapsed_ms, reverse=True)[:limit]

    def largest_outputs(self, limit: int = 8) -> list[ModuleRecord]:
        return sorted(self.records, key=lambda record: record.output_bytes, reverse=True)[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_ms": self.total_ms,
            "records": [asdict(record) for record in self.records],
            "gradients": [asdict(record) for record in self.gradients],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self, *, limit: int | None = None, show_shapes: bool = False, show_stats: bool = False) -> str:
        records = self.records[:limit] if limit is not None else self.records
        rows = []
        for rec in records:
            row = [
                rec.name,
                rec.type_name,
                f"{rec.elapsed_ms:.3f}",
                format_bytes(rec.input_bytes),
                format_bytes(rec.output_bytes),
                format_bytes(rec.parameter_bytes),
                _fmt_optional_bytes(rec.cuda_delta_bytes),
                "yes" if rec.has_nan else "",
                "yes" if rec.has_inf else "",
            ]
            if show_shapes:
                row.extend([rec.input_shapes, rec.output_shapes])
            if show_stats:
                row.append(_stats_summary(rec.output_stats))
            rows.append(tuple(row))

        headers = ["Module", "Type", "ms", "Input", "Output", "Params", "CUDA delta", "NaN", "Inf"]
        if show_shapes:
            headers.extend(["Input shape", "Output shape"])
        if show_stats:
            headers.append("Output stats")

        text = [make_table(tuple(headers), rows)]
        if records:
            text.append("")
            text.append("Hot layers")
            hot_rows = [(rec.name, rec.type_name, f"{rec.elapsed_ms:.3f} ms", format_bytes(rec.output_bytes)) for rec in self.slowest()]
            text.append(make_table(("Module", "Type", "Time", "Output"), hot_rows))
        if self.gradients:
            grad_rows = [
                (
                    grad.name,
                    f"{grad.norm:.4g}",
                    f"{grad.max_abs:.4g}",
                    f"{grad.mean:.4g}",
                    f"{grad.std:.4g}",
                    "yes" if grad.has_nan else "",
                    "yes" if grad.has_inf else "",
                )
                for grad in self.gradients
            ]
            text.extend(["", "Gradients", make_table(("Param", "L2 norm", "Max abs", "Mean", "Std", "NaN", "Inf"), grad_rows)])
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
        output_stats = _first_tensor_stats(self._torch, output) if self.collect_stats else None
        has_nan, has_inf = _nonfinite_flags(self._torch, output)
        params = list(module.parameters(recurse=False))
        self.records.append(
            ModuleRecord(
                name=self._names.get(id(module), module.__class__.__name__),
                type_name=module.__class__.__name__,
                depth=self._depths.get(id(module), 0),
                elapsed_ms=elapsed_ms,
                input_bytes=_tree_nbytes(inputs),
                output_bytes=_tree_nbytes(output),
                parameter_bytes=sum(param.numel() * param.element_size() for param in params),
                parameter_count=sum(param.numel() for param in params),
                trainable_parameter_count=sum(param.numel() for param in params if getattr(param, "requires_grad", False)),
                cuda_before_bytes=cuda_before,
                cuda_after_bytes=cuda_after,
                cuda_delta_bytes=cuda_delta,
                input_shapes=_shape_summary(inputs),
                output_shapes=_shape_summary(output),
                output_stats=output_stats,
                has_nan=has_nan,
                has_inf=has_inf,
            )
        )

    def _make_grad_hook(self, name: str):
        def hook(grad):
            detached = grad.detach().float()
            self.gradients.append(
                GradientRecord(
                    name=name,
                    norm=float(detached.norm().item()),
                    max_abs=float(detached.abs().max().item()),
                    mean=float(detached.mean().item()),
                    std=float(detached.std(unbiased=False).item()),
                    has_nan=bool(self._torch.isnan(detached).any().item()),
                    has_inf=bool(self._torch.isinf(detached).any().item()),
                )
            )
            return grad

        return hook


def trace_forward(model, *args, record_leaf_only: bool = True, collect_stats: bool = True, **kwargs) -> tuple[Any, TorchTrace]:
    with TorchTrace(model, record_leaf_only=record_leaf_only, collect_stats=collect_stats) as trace:
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


def _first_tensor_stats(torch_module, value: Any) -> TensorStats | None:
    tensor = _first_tensor(value)
    if tensor is None:
        return None
    stats = TensorStats(
        shape=str(tuple(tensor.shape)),
        dtype=str(tensor.dtype).replace("torch.", ""),
        device=str(tensor.device),
        bytes=int(tensor.numel() * tensor.element_size()),
    )
    if hasattr(tensor, "is_floating_point") and tensor.is_floating_point() and tensor.numel() > 0:
        detached = tensor.detach().float()
        stats.mean = float(detached.mean().item())
        stats.std = float(detached.std(unbiased=False).item())
        stats.min = float(detached.min().item())
        stats.max = float(detached.max().item())
        stats.zero_pct = float((detached == 0).float().mean().item())
        stats.has_nan = bool(torch_module.isnan(detached).any().item())
        stats.has_inf = bool(torch_module.isinf(detached).any().item())
    return stats


def _first_tensor(value: Any):
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


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


def _stats_summary(stats: TensorStats | None) -> str:
    if stats is None or stats.mean is None:
        return "n/a"
    return f"mean={stats.mean:.3g} std={stats.std:.3g} min={stats.min:.3g} max={stats.max:.3g} zero={stats.zero_pct:.1%}"


def _fmt_optional_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return format_bytes(value)
