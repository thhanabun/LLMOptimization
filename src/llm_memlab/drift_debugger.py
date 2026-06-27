from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .report import make_table


@dataclass(frozen=True)
class LayerDriftRecord:
    name: str
    baseline_shape: str
    candidate_shape: str
    mean_abs_error: float | None
    max_abs_error: float | None
    cosine_similarity: float | None
    status: str = "ok"


@dataclass(frozen=True)
class DriftReport:
    records: tuple[LayerDriftRecord, ...]

    @property
    def worst(self) -> LayerDriftRecord | None:
        valid = [record for record in self.records if record.mean_abs_error is not None]
        return max(valid, key=lambda record: record.mean_abs_error, default=None)

    def to_text(self, *, limit: int | None = None) -> str:
        records = self.records[:limit] if limit is not None else self.records
        rows = [
            (
                record.name,
                record.status,
                record.baseline_shape,
                record.candidate_shape,
                _fmt(record.mean_abs_error),
                _fmt(record.max_abs_error),
                _fmt(record.cosine_similarity),
            )
            for record in records
        ]
        text = [make_table(("Layer", "Status", "Base shape", "Candidate shape", "Mean abs", "Max abs", "Cosine"), rows)]
        if self.worst is not None:
            text.extend(["", f"Worst drift: {self.worst.name} mean_abs={self.worst.mean_abs_error:.6f}"])
        return "\n".join(text)


def compare_layer_drift(baseline: Any, candidate: Any, *args, record_leaf_only: bool = True, **kwargs) -> DriftReport:
    torch = _import_torch()
    baseline_outputs: dict[str, Any] = {}
    candidate_outputs: dict[str, Any] = {}
    handles = []
    handles.extend(_attach_hooks(baseline, baseline_outputs, record_leaf_only=record_leaf_only))
    handles.extend(_attach_hooks(candidate, candidate_outputs, record_leaf_only=record_leaf_only))
    try:
        with torch.no_grad():
            baseline(*args, **kwargs)
            candidate(*args, **kwargs)
    finally:
        while handles:
            handles.pop().remove()
    names = sorted(set(baseline_outputs) | set(candidate_outputs))
    records = [
        _compare_tensors(torch, name, baseline_outputs.get(name), candidate_outputs.get(name))
        for name in names
    ]
    return DriftReport(tuple(records))


def _attach_hooks(model: Any, outputs: dict[str, Any], *, record_leaf_only: bool):
    handles = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if record_leaf_only and any(True for _ in module.children()):
            continue
        handles.append(module.register_forward_hook(_make_hook(name, outputs)))
    return handles


def _make_hook(name: str, outputs: dict[str, Any]):
    def hook(module, inputs, output):
        tensor = _first_tensor(output)
        if tensor is not None:
            outputs[name] = tensor.detach().float().cpu()

    return hook


def _compare_tensors(torch, name: str, baseline, candidate) -> LayerDriftRecord:
    if baseline is None or candidate is None:
        return LayerDriftRecord(name, _shape(baseline), _shape(candidate), None, None, None, status="missing")
    if tuple(baseline.shape) != tuple(candidate.shape):
        return LayerDriftRecord(name, _shape(baseline), _shape(candidate), None, None, None, status="shape-mismatch")
    diff = (baseline - candidate).abs()
    cosine = torch.nn.functional.cosine_similarity(baseline.flatten(), candidate.flatten(), dim=0)
    return LayerDriftRecord(
        name=name,
        baseline_shape=_shape(baseline),
        candidate_shape=_shape(candidate),
        mean_abs_error=float(diff.mean().item()),
        max_abs_error=float(diff.max().item()),
        cosine_similarity=float(cosine.item()),
    )


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


def _shape(value: Any) -> str:
    return "missing" if value is None else str(tuple(value.shape))


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Layer drift debugging requires PyTorch. Install with: pip install torch") from exc
    return torch
