from __future__ import annotations

import re
from functools import reduce
from operator import mul
from typing import Iterable


DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "fp8": 1,
    "float8": 1,
    "int16": 2,
    "float16": 2,
    "fp16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "fp32": 4,
    "tf32": 4,
    "int64": 8,
    "float64": 8,
    "fp64": 8,
    "int4": 0.5,
    "nf4": 0.5,
}


def dtype_size_bytes(dtype: str) -> float:
    key = dtype.lower().replace("torch.", "")
    if key not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype {dtype!r}. Known dtypes: {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[key]


def numel(shape: Iterable[int]) -> int:
    values = tuple(int(dim) for dim in shape)
    if not values:
        return 1
    return reduce(mul, values, 1)


def tensor_nbytes(shape: Iterable[int], dtype: str) -> float:
    return numel(shape) * dtype_size_bytes(dtype)


def format_bytes(value: float | int, *, precision: int = 2) -> str:
    amount = float(value)
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit = units[0]
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            break
        amount /= 1024
    if unit == "B":
        return f"{sign}{int(amount)} {unit}"
    return f"{sign}{amount:.{precision}f} {unit}"


_BYTE_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[kmgtp]?i?b?|bytes?)?\s*$", re.I)


def parse_bytes(value: str) -> int:
    match = _BYTE_RE.match(value)
    if not match:
        raise ValueError(f"Could not parse byte value: {value!r}")
    amount = float(match.group("num"))
    unit = (match.group("unit") or "b").lower()
    unit = unit.replace("bytes", "b").replace("byte", "b")
    multipliers = {
        "b": 1,
        "": 1,
        "kb": 1024,
        "kib": 1024,
        "mb": 1024**2,
        "mib": 1024**2,
        "gb": 1024**3,
        "gib": 1024**3,
        "tb": 1024**4,
        "tib": 1024**4,
        "pb": 1024**5,
        "pib": 1024**5,
    }
    if unit not in multipliers:
        raise ValueError(f"Unknown byte unit {unit!r}")
    return int(amount * multipliers[unit])
