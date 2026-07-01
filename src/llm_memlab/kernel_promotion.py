from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hardware import HardwareProfile, detect_hardware_profile
from .kernel_certification import KernelCertificationReport
from .report import make_table

KERNEL_PROMOTION_SCHEMA_VERSION = "llm_memlab.kernel_promotion.v1"


@dataclass(frozen=True)
class KernelPromotionRequirements:
    required_batches: tuple[int, ...] = (1, 2)
    required_q_heads: tuple[int, ...] = (4, 8)
    required_kv_heads: tuple[int, ...] = (1, 2, 4)
    required_head_dims: tuple[int, ...] = (32, 64)
    required_sequence_lengths: tuple[int, ...] = (64, 256)
    required_page_sizes: tuple[int, ...] = (16, 32)
    required_quant_dtypes: tuple[str, ...] = ("int8", "uint8")
    required_compute_dtypes: tuple[str, ...] = ("fp16", "bf16")
    min_cases: int = 16
    require_gqa: bool = True
    require_mqa: bool = True
    require_long_context: bool = True
    long_context_tokens: int = 4096


@dataclass(frozen=True)
class KernelPromotionDecision:
    backend: str
    tier: str
    promoted: bool
    reasons: tuple[str, ...]
    schema_version: str = KERNEL_PROMOTION_SCHEMA_VERSION

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Backend", self.backend),
                ("Tier", self.tier),
                ("Promoted", self.promoted),
                ("Reasons", "; ".join(self.reasons)),
            ],
        )


def decide_kernel_promotion(
    report: KernelCertificationReport,
    *,
    backend: str = "triton",
    hardware: HardwareProfile | None = None,
    require_long_context: bool | None = None,
    requirements: KernelPromotionRequirements | None = None,
) -> KernelPromotionDecision:
    hw = hardware or detect_hardware_profile()
    req = requirements or KernelPromotionRequirements()
    if require_long_context is not None:
        req = KernelPromotionRequirements(
            required_batches=req.required_batches,
            required_q_heads=req.required_q_heads,
            required_kv_heads=req.required_kv_heads,
            required_head_dims=req.required_head_dims,
            required_sequence_lengths=req.required_sequence_lengths,
            required_page_sizes=req.required_page_sizes,
            required_quant_dtypes=req.required_quant_dtypes,
            required_compute_dtypes=req.required_compute_dtypes,
            min_cases=req.min_cases,
            require_gqa=req.require_gqa,
            require_mqa=req.require_mqa,
            require_long_context=bool(require_long_context),
            long_context_tokens=req.long_context_tokens,
        )
    reasons: list[str] = []
    passed_results = [item for item in report.results if item.passed and not item.skipped]
    if not report.results:
        reasons.append("no certification results")
    if not report.passed or report.skipped:
        reasons.append("certification did not pass on this hardware")
    if len(passed_results) < req.min_cases:
        reasons.append(f"passed certification cases {len(passed_results)} < required {req.min_cases}")
    reasons.extend(_missing_requirements(passed_results, req))
    if backend == "cutile" and hw.architecture not in {"hopper", "blackwell"}:
        reasons.append(f"CuTile promotion requires Hopper/Blackwell, got {hw.architecture}")
    tier = "production" if not reasons else "experimental"
    return KernelPromotionDecision(
        backend=backend, tier=tier, promoted=not reasons, reasons=tuple(reasons or ["all promotion gates passed"])
    )


def _missing_requirements(results, requirements: KernelPromotionRequirements) -> list[str]:
    reasons: list[str] = []
    cases = [item.case for item in results]
    _require_values(reasons, "batch", {case.batch for case in cases}, requirements.required_batches)
    _require_values(reasons, "q_heads", {case.q_heads for case in cases}, requirements.required_q_heads)
    _require_values(reasons, "kv_heads", {case.kv_heads for case in cases}, requirements.required_kv_heads)
    _require_values(reasons, "head_dim", {case.head_dim for case in cases}, requirements.required_head_dims)
    _require_values(reasons, "sequence_length", {case.sequence_length for case in cases}, requirements.required_sequence_lengths)
    _require_values(reasons, "page_size", {case.page_size for case in cases}, requirements.required_page_sizes)
    _require_values(reasons, "quant_dtype", {case.quant_dtype for case in cases}, requirements.required_quant_dtypes)
    _require_values(reasons, "compute_dtype", {case.compute_dtype for case in cases}, requirements.required_compute_dtypes)
    if requirements.require_gqa and not any(case.q_heads > case.kv_heads > 1 for case in cases):
        reasons.append("GQA coverage requires at least one q_heads > kv_heads > 1 case")
    if requirements.require_mqa and not any(case.kv_heads == 1 and case.q_heads > 1 for case in cases):
        reasons.append("MQA coverage requires at least one kv_heads == 1 case")
    if requirements.require_long_context and max((case.sequence_length for case in cases), default=0) < requirements.long_context_tokens:
        reasons.append(f"long-context certification requires seq >= {requirements.long_context_tokens}")
    return reasons


def _require_values(reasons: list[str], label: str, actual: set[Any], required: tuple[Any, ...]) -> None:
    missing = [item for item in required if item not in actual]
    if missing:
        reasons.append(f"{label} coverage missing {missing}")
