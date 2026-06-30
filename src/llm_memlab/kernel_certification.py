from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .benchmark_store import BenchmarkRecord, collect_run_metadata, write_benchmark_csv, write_benchmark_json
from .decode_benchmarks import benchmark_fused_decode_attention
from .report import make_table

CERTIFICATION_SCHEMA_VERSION = "llm_memlab.kernel_certification.v1"
SUPPORTED_COMPUTE_DTYPES = ("fp16", "bf16")
SUPPORTED_QUANT_DTYPES = ("int8", "uint8")


@dataclass(frozen=True)
class KernelCertificationCase:
    batch: int = 1
    q_heads: int = 4
    kv_heads: int = 2
    head_dim: int = 32
    sequence_length: int = 64
    page_size: int = 16
    compute_dtype: str = "fp16"
    quant_dtype: str = "int8"
    backend: str = "triton-experimental"

    @property
    def name(self) -> str:
        return (
            f"cert/b{self.batch}/q{self.q_heads}/kv{self.kv_heads}/t{self.sequence_length}/"
            f"d{self.head_dim}/{self.compute_dtype}/{self.quant_dtype}/p{self.page_size}"
        )

    def validate(self) -> None:
        if min(self.batch, self.q_heads, self.kv_heads, self.head_dim, self.sequence_length, self.page_size) <= 0:
            raise ValueError("kernel certification shapes must be positive")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads for GQA/MQA certification")
        if self.compute_dtype not in SUPPORTED_COMPUTE_DTYPES:
            raise ValueError(f"compute_dtype must be one of {SUPPORTED_COMPUTE_DTYPES}")
        if self.quant_dtype not in SUPPORTED_QUANT_DTYPES:
            raise ValueError(f"quant_dtype must be one of {SUPPORTED_QUANT_DTYPES}")


@dataclass(frozen=True)
class KernelCertificationResult:
    case: KernelCertificationCase
    passed: bool
    skipped: bool = False
    skipped_reason: str | None = None
    fused_ms: float | None = None
    reference_ms: float | None = None
    speedup: float | None = None
    mean_abs_error: float | None = None
    max_abs_error: float | None = None
    top1_agreement: float | None = None
    topk_agreement: float | None = None
    tolerance_mean_abs: float = 0.03
    tolerance_top1: float = 0.90
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CERTIFICATION_SCHEMA_VERSION

    def to_record(self) -> BenchmarkRecord:
        mean_ms = self.fused_ms if self.fused_ms is not None else 0.0
        ref_ms = self.reference_ms if self.reference_ms is not None else 0.0
        extra = {
            "certification_schema_version": self.schema_version,
            "case": asdict(self.case),
            "quality_passed": self.passed and not self.skipped,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
            "reference_ms": ref_ms,
            "speedup": self.speedup,
            "mean_abs": self.mean_abs_error,
            "max_abs": self.max_abs_error,
            "top1": self.top1_agreement,
            "topk": self.topk_agreement,
            "tolerance_mean_abs": self.tolerance_mean_abs,
            "tolerance_top1": self.tolerance_top1,
        }
        return BenchmarkRecord(
            name=self.case.name,
            kind="kernel-certification",
            mean_ms=mean_ms,
            min_ms=mean_ms,
            max_ms=mean_ms,
            extra=extra,
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class KernelCertificationReport:
    results: tuple[KernelCertificationResult, ...]
    schema_version: str = CERTIFICATION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return all(item.passed or item.skipped for item in self.results)

    @property
    def skipped(self) -> bool:
        return bool(self.results) and all(item.skipped for item in self.results)

    def to_records(self) -> list[BenchmarkRecord]:
        return [item.to_record() for item in self.results]

    def write_json(self, path: str | Path) -> Path:
        return write_benchmark_json(self.to_records(), path)

    def write_csv(self, path: str | Path) -> Path:
        return write_benchmark_csv(self.to_records(), path)

    def to_text(self) -> str:
        rows = []
        for item in self.results:
            if item.skipped:
                rows.append((item.case.name, "SKIP", "-", "-", "-", item.skipped_reason or "skipped"))
            else:
                status = "PASS" if item.passed else "FAIL"
                rows.append(
                    (
                        item.case.name,
                        status,
                        f"{item.fused_ms:.3f}" if item.fused_ms is not None else "-",
                        f"{item.reference_ms:.3f}" if item.reference_ms is not None else "-",
                        f"{item.mean_abs_error:.6f}" if item.mean_abs_error is not None else "-",
                        f"{item.speedup:.3f}x" if item.speedup is not None else "-",
                    )
                )
        return make_table(("Case", "Status", "Fused ms", "Ref ms", "Mean abs", "Speedup"), rows)


def default_kernel_certification_cases(*, quick: bool = False) -> list[KernelCertificationCase]:
    if quick:
        return [
            KernelCertificationCase(
                q_heads=2, kv_heads=1, head_dim=16, sequence_length=32, page_size=16, compute_dtype="fp16", quant_dtype="int8"
            ),
            KernelCertificationCase(
                q_heads=2, kv_heads=1, head_dim=16, sequence_length=32, page_size=16, compute_dtype="fp16", quant_dtype="uint8"
            ),
        ]
    cases: list[KernelCertificationCase] = []
    for batch in (1, 2):
        for q_heads, kv_heads in ((4, 4), (4, 2), (8, 1)):
            for head_dim in (32, 64):
                for sequence_length in (64, 256):
                    for page_size in (16, 32):
                        for compute_dtype in SUPPORTED_COMPUTE_DTYPES:
                            for quant_dtype in SUPPORTED_QUANT_DTYPES:
                                cases.append(
                                    KernelCertificationCase(
                                        batch, q_heads, kv_heads, head_dim, sequence_length, page_size, compute_dtype, quant_dtype
                                    )
                                )
    return cases


def certify_quantized_attention(
    cases: list[KernelCertificationCase] | None = None,
    *,
    quick: bool = False,
    repeats: int = 5,
    warmup: int = 2,
    seed: int = 0,
    max_mean_abs: float = 0.03,
    min_top1: float = 0.90,
) -> KernelCertificationReport:
    selected = cases or default_kernel_certification_cases(quick=quick)
    torch = _import_torch()
    if not torch.cuda.is_available():
        metadata = collect_run_metadata(warmup=warmup, repeats=repeats, seed=seed, backend="torch-skip").__dict__
        return KernelCertificationReport(
            tuple(
                KernelCertificationResult(case=case, passed=True, skipped=True, skipped_reason="CUDA is not available", metadata=metadata)
                for case in selected
            )
        )

    results: list[KernelCertificationResult] = []
    for index, case in enumerate(selected):
        case.validate()
        metadata = collect_run_metadata(
            dtype=case.compute_dtype,
            sequence_length=case.sequence_length,
            warmup=warmup,
            repeats=repeats,
            seed=seed + index,
            backend=case.backend,
        ).__dict__
        try:
            bench = benchmark_fused_decode_attention(
                batch=case.batch,
                q_heads=case.q_heads,
                kv_heads=case.kv_heads,
                tokens=case.sequence_length,
                head_dim=case.head_dim,
                dtype=case.compute_dtype,
                quant_dtype=case.quant_dtype,
                repeats=repeats,
                warmup=warmup,
                seed=seed + index,
            )
            quality_passed = bench.quality.mean_abs_error <= max_mean_abs and bench.quality.top1_agreement >= min_top1
            results.append(
                KernelCertificationResult(
                    case=case,
                    passed=bool(bench.quality.passed and quality_passed),
                    fused_ms=bench.fused.mean_ms,
                    reference_ms=bench.dequant_sdpa.mean_ms,
                    speedup=bench.speedup,
                    mean_abs_error=bench.quality.mean_abs_error,
                    max_abs_error=bench.quality.max_abs_error,
                    top1_agreement=bench.quality.top1_agreement,
                    topk_agreement=bench.quality.topk_agreement,
                    tolerance_mean_abs=max_mean_abs,
                    tolerance_top1=min_top1,
                    metadata=metadata,
                )
            )
        except Exception as exc:
            results.append(KernelCertificationResult(case=case, passed=False, skipped=False, skipped_reason=str(exc), metadata=metadata))
    return KernelCertificationReport(tuple(results))


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Kernel certification requires PyTorch") from exc
    return torch
