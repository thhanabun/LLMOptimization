from __future__ import annotations

from dataclasses import dataclass, field

from .bytes import format_bytes


@dataclass(frozen=True)
class TensorLifetime:
    name: str
    size_bytes: int
    start: int
    end: int
    kind: str = "activation"
    device: str = "cuda"

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)

    @property
    def pressure_score(self) -> int:
        return self.size_bytes * max(1, self.duration)


@dataclass(frozen=True)
class BufferAssignment:
    tensor_name: str
    buffer_id: int
    buffer_size_bytes: int
    start: int
    end: int


@dataclass
class BufferPlan:
    lifetimes: list[TensorLifetime]
    assignments: list[BufferAssignment]
    eager_peak_bytes: int
    pooled_bytes: int
    total_allocated_bytes: int
    timeline: list[tuple[int, int]] = field(default_factory=list)

    @property
    def reuse_saving_bytes(self) -> int:
        return max(0, self.total_allocated_bytes - self.pooled_bytes)

    @property
    def fragmentation_bytes(self) -> int:
        return max(0, self.pooled_bytes - self.eager_peak_bytes)

    def top_pressure(self, limit: int = 8, *, kind: str | None = None) -> list[TensorLifetime]:
        values = self.lifetimes
        if kind is not None:
            values = [item for item in values if item.kind == kind]
        return sorted(values, key=lambda item: item.pressure_score, reverse=True)[:limit]

    def to_text(self) -> str:
        from .report import make_table

        rows = [
            ("Peak live memory", format_bytes(self.eager_peak_bytes)),
            ("Planned pool size", format_bytes(self.pooled_bytes)),
            ("Total allocation traffic", format_bytes(self.total_allocated_bytes)),
            ("Reuse saving", format_bytes(self.reuse_saving_bytes)),
            ("Fragmentation overhead", format_bytes(self.fragmentation_bytes)),
        ]
        text = [make_table(("Metric", "Value"), rows)]
        pressure_rows = [
            (
                item.name,
                item.kind,
                item.duration,
                format_bytes(item.size_bytes),
                format_bytes(item.pressure_score),
            )
            for item in self.top_pressure()
        ]
        if pressure_rows:
            text.append("")
            text.append("Top memory-pressure tensors")
            text.append(make_table(("Tensor", "Kind", "Life", "Size", "Pressure"), pressure_rows))
        return "\n".join(text)


class MemoryPlanner:
    """Greedy lifetime planner for reusable tensor buffers."""

    def __init__(self, lifetimes: list[TensorLifetime]):
        self.lifetimes = sorted(lifetimes, key=lambda item: (item.start, -item.size_bytes, item.end))

    def plan(self, *, reusable_kinds: set[str] | None = None) -> BufferPlan:
        reusable_kinds = reusable_kinds or {"activation", "temporary", "kv_cache"}
        timeline = self._timeline()
        eager_peak = max((live for _, live in timeline), default=0)
        total_allocated = sum(item.size_bytes for item in self.lifetimes)

        persistent = [item for item in self.lifetimes if item.kind not in reusable_kinds]
        reusable = [item for item in self.lifetimes if item.kind in reusable_kinds]

        assignments: list[BufferAssignment] = []
        buffers: list[dict[str, int]] = []

        for item in sorted(reusable, key=lambda value: (value.start, -value.size_bytes)):
            chosen_index = None
            chosen_capacity = None
            for index, buf in enumerate(buffers):
                if buf["free_at"] <= item.start and buf["capacity"] >= item.size_bytes:
                    if chosen_capacity is None or buf["capacity"] < chosen_capacity:
                        chosen_index = index
                        chosen_capacity = buf["capacity"]
            if chosen_index is None:
                chosen_index = len(buffers)
                buffers.append({"capacity": item.size_bytes, "free_at": item.end})
            else:
                buffers[chosen_index]["free_at"] = item.end
            assignments.append(
                BufferAssignment(
                    tensor_name=item.name,
                    buffer_id=chosen_index,
                    buffer_size_bytes=buffers[chosen_index]["capacity"],
                    start=item.start,
                    end=item.end,
                )
            )

        persistent_bytes = sum(item.size_bytes for item in persistent)
        pooled_bytes = persistent_bytes + sum(buf["capacity"] for buf in buffers)
        return BufferPlan(
            lifetimes=self.lifetimes,
            assignments=assignments,
            eager_peak_bytes=eager_peak,
            pooled_bytes=pooled_bytes,
            total_allocated_bytes=total_allocated,
            timeline=timeline,
        )

    def _timeline(self) -> list[tuple[int, int]]:
        events: list[tuple[int, int]] = []
        for item in self.lifetimes:
            events.append((item.start, item.size_bytes))
            events.append((item.end, -item.size_bytes))
        live = 0
        timeline: list[tuple[int, int]] = []
        for step, delta in sorted(events, key=lambda value: (value[0], value[1])):
            live += delta
            timeline.append((step, live))
        return timeline
