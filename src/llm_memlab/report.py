from __future__ import annotations

from collections.abc import Iterable, Sequence


def make_table(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> str:
    materialized = [tuple(str(cell) for cell in row) for row in rows]
    header = tuple(str(cell) for cell in headers)
    widths = [len(cell) for cell in header]
    for row in materialized:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def fmt(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()

    lines = [fmt(header), fmt(tuple("-" * width for width in widths))]
    lines.extend(fmt(row) for row in materialized)
    return "\n".join(lines)
