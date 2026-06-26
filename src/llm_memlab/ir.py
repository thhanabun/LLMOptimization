from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .bytes import tensor_nbytes


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str = "bf16"
    device: str = "cuda"
    role: str = "activation"

    @classmethod
    def from_shape(
        cls,
        name: str,
        shape: Iterable[int],
        *,
        dtype: str = "bf16",
        device: str = "cuda",
        role: str = "activation",
    ) -> "TensorSpec":
        return cls(name=name, shape=tuple(int(dim) for dim in shape), dtype=dtype, device=device, role=role)

    @property
    def nbytes(self) -> int:
        return int(tensor_nbytes(self.shape, self.dtype))


@dataclass(frozen=True)
class OperationSpec:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        name: str,
        op_type: str,
        inputs: Iterable[str],
        outputs: Iterable[str],
        **attrs: Any,
    ) -> "OperationSpec":
        return cls(
            name=name,
            op_type=op_type,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            attrs=dict(attrs),
        )


@dataclass
class GraphSpec:
    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    operations: list[OperationSpec] = field(default_factory=list)
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def add_tensor(self, tensor: TensorSpec) -> TensorSpec:
        if tensor.name in self.tensors:
            raise ValueError(f"Tensor {tensor.name!r} already exists")
        self.tensors[tensor.name] = tensor
        return tensor

    def add_op(self, op: OperationSpec) -> OperationSpec:
        missing = [name for name in (*op.inputs, *op.outputs) if name not in self.tensors]
        if missing:
            raise ValueError(f"Operation {op.name!r} references unknown tensors: {missing}")
        self.operations.append(op)
        return op

    def validate(self) -> None:
        seen = set(self.inputs)
        seen.update(name for name, tensor in self.tensors.items() if tensor.role == "parameter")
        unknown_inputs = [name for name in self.inputs if name not in self.tensors]
        unknown_outputs = [name for name in self.outputs if name not in self.tensors]
        if unknown_inputs or unknown_outputs:
            raise ValueError(f"Unknown graph endpoints: inputs={unknown_inputs}, outputs={unknown_outputs}")
        for index, op in enumerate(self.operations):
            missing_inputs = [name for name in op.inputs if name not in self.tensors]
            missing_outputs = [name for name in op.outputs if name not in self.tensors]
            if missing_inputs or missing_outputs:
                raise ValueError(f"Op {op.name!r} references unknown tensors")
            produced_late = [name for name in op.inputs if name not in seen]
            if produced_late:
                raise ValueError(f"Op {op.name!r} at index {index} consumes tensors before production: {produced_late}")
            seen.update(op.outputs)

    def tensor_lifetimes(self):
        from .planner import TensorLifetime

        self.validate()
        first_seen: dict[str, int] = {}
        last_seen: dict[str, int] = {}

        for name in self.inputs:
            first_seen[name] = 0
            last_seen[name] = 0

        for index, op in enumerate(self.operations, start=1):
            for name in op.outputs:
                first_seen.setdefault(name, index)
                last_seen[name] = max(last_seen.get(name, index), index)
            for name in op.inputs:
                first_seen.setdefault(name, 0)
                last_seen[name] = max(last_seen.get(name, 0), index)

        terminal = len(self.operations) + 1
        for name in self.outputs:
            last_seen[name] = terminal

        lifetimes = []
        for name, tensor in self.tensors.items():
            if tensor.role == "parameter":
                start = 0
                end = terminal
            else:
                start = first_seen.get(name, 0)
                end = max(last_seen.get(name, start + 1), start + 1)
            lifetimes.append(
                TensorLifetime(
                    name=name,
                    size_bytes=tensor.nbytes,
                    start=start,
                    end=end,
                    kind=tensor.role,
                    device=tensor.device,
                )
            )
        return lifetimes
