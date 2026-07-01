from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hf_cache_certification import HFCacheCertificationReport, certify_hf_cache
from .hf_cache_profiles import QuantizedCacheCertificationProfile, write_quantized_cache_profiles
from .report import make_table

CERTIFICATION_MATRIX_SCHEMA_VERSION = "llm_memlab.certification_matrix.v1"


@dataclass(frozen=True)
class ModelCertificationTarget:
    name: str
    family: str
    model: str
    local_files_only: bool = True
    production: bool = True


@dataclass(frozen=True)
class ModelCertificationOutcome:
    target: ModelCertificationTarget
    status: str
    report: HFCacheCertificationReport | None = None
    profile: QuantizedCacheCertificationProfile | None = None
    error: str | None = None


@dataclass(frozen=True)
class CertificationMatrixGateResult:
    passed: bool
    real_certified_count: int
    required_real_certified_count: int
    skipped_count: int
    failed_count: int
    reasons: tuple[str, ...]

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Gate passed", self.passed),
                ("Real certified", f"{self.real_certified_count}/{self.required_real_certified_count}"),
                ("Skipped", self.skipped_count),
                ("Failed", self.failed_count),
                ("Reasons", "; ".join(self.reasons) if self.reasons else "all certification gates passed"),
            ],
        )


@dataclass(frozen=True)
class CertificationMatrixReport:
    outcomes: tuple[ModelCertificationOutcome, ...]
    schema_version: str = CERTIFICATION_MATRIX_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(item.status in {"passed", "skipped"} for item in self.outcomes)

    @property
    def profiles(self) -> tuple[QuantizedCacheCertificationProfile, ...]:
        return tuple(item.profile for item in self.outcomes if item.profile is not None)

    @property
    def real_certified_count(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "passed" and item.report is not None)

    @property
    def skipped_count(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "skipped")

    @property
    def failed_count(self) -> int:
        return sum(1 for item in self.outcomes if item.status == "failed")

    @property
    def production_target_count(self) -> int:
        return sum(1 for item in self.outcomes if item.target.production)

    def evaluate_gate(
        self,
        *,
        require_real_models: bool = False,
        min_certified_models: int = 0,
        strict: bool = False,
    ) -> CertificationMatrixGateResult:
        required = max(0, int(min_certified_models))
        if strict:
            require_real_models = True
            required = max(required, self.production_target_count)
        elif require_real_models:
            required = max(required, 1)
        reasons: list[str] = []
        if self.failed_count:
            reasons.append(f"{self.failed_count} model certification case(s) failed")
        if require_real_models and self.skipped_count:
            reasons.append(f"{self.skipped_count} model certification case(s) skipped")
        if self.real_certified_count < required:
            reasons.append(f"real certified models {self.real_certified_count} < required {required}")
        return CertificationMatrixGateResult(
            passed=not reasons,
            real_certified_count=self.real_certified_count,
            required_real_certified_count=required,
            skipped_count=self.skipped_count,
            failed_count=self.failed_count,
            reasons=tuple(reasons),
        )

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        payload = {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "real_certified_count": self.real_certified_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "metadata": self.metadata,
            "outcomes": [_outcome_to_dict(item) for item in self.outcomes],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def write_profiles(self, path: str | Path) -> Path:
        return write_quantized_cache_profiles(self.profiles, path)

    def to_text(self) -> str:
        rows = [
            (
                item.target.name,
                item.target.family,
                item.status,
                "n/a" if item.profile is None else item.profile.certified_backend,
                "n/a" if item.profile is None else item.profile.safe_prompt_tokens,
                item.error or "",
            )
            for item in self.outcomes
        ]
        return (
            make_table(("Target", "Family", "Status", "Backend", "Safe tokens", "Error"), rows)
            + "\n\n"
            + make_table(
                ("Metric", "Value"),
                [
                    ("Real certified", self.real_certified_count),
                    ("Skipped", self.skipped_count),
                    ("Failed", self.failed_count),
                ],
            )
        )


def default_model_certification_targets(*, local_root: str | Path | None = None) -> tuple[ModelCertificationTarget, ...]:
    root = Path(local_root) if local_root is not None else None
    specs = (
        ("tinyllama", "llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        ("qwen2.5", "qwen", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("qwen3", "qwen3", "Qwen/Qwen3-1.7B"),
        ("mistral", "mistral", "mistralai/Mistral-7B-Instruct-v0.3"),
        ("gemma", "gemma", "google/gemma-2-2b-it"),
        ("phi", "phi", "microsoft/Phi-3-mini-4k-instruct"),
        ("deepseek", "deepseek", "deepseek-ai/deepseek-coder-1.3b-instruct"),
    )
    targets = []
    for name, family, model in specs:
        resolved = _resolve_local_model(root, name, model)
        targets.append(ModelCertificationTarget(name=name, family=family, model=resolved, local_files_only=True))
    return tuple(targets)


def certify_model_matrix(
    targets: tuple[ModelCertificationTarget, ...] | list[ModelCertificationTarget] | None = None,
    *,
    prompts: tuple[str, ...] = ("hello", "Explain KV cache briefly."),
    device: str = "auto",
    dtype: str = "auto",
    allow_remote: bool = False,
) -> CertificationMatrixReport:
    selected = tuple(targets or default_model_certification_targets())
    outcomes: list[ModelCertificationOutcome] = []
    for target in selected:
        if target.local_files_only and not allow_remote and not Path(target.model).exists():
            outcomes.append(
                ModelCertificationOutcome(
                    target=target,
                    status="skipped",
                    profile=_conservative_profile(target, "model is not available locally"),
                    error="model is not available locally",
                )
            )
            continue
        try:
            report = certify_hf_cache(
                target.model,
                prompts=list(prompts),
                token_counts=[1],
                caches=["paged"],
                experimental_caches=["quantized"],
                quant_dtypes=["int8", "uint8"],
                device=device,
                dtype=dtype,
                local_files_only=target.local_files_only and not allow_remote,
                allow_experimental_direct_cache=True,
            )
            profile = _profile_from_report(target, report)
            outcomes.append(
                ModelCertificationOutcome(target=target, status="passed" if report.passed else "failed", report=report, profile=profile)
            )
        except Exception as exc:
            outcomes.append(
                ModelCertificationOutcome(
                    target=target, status="failed", profile=_conservative_profile(target, str(exc)), error=str(exc)[:500]
                )
            )
    return CertificationMatrixReport(tuple(outcomes), metadata={"device": device, "dtype": dtype, "allow_remote": allow_remote})


def _profile_from_report(target: ModelCertificationTarget, report: HFCacheCertificationReport) -> QuantizedCacheCertificationProfile:
    quantized = [item for item in report.results if item.case.cache == "quantized"]
    direct_ok = [
        item
        for item in quantized
        if item.direct_cache and item.fallback_reason is None and item.profile_passed is not False and item.token_quality.exact_match
    ]
    if direct_ok:
        safe_tokens = max(item.prompt_tokens for item in direct_ok)
        production = target.production
        notes = ("derived from real model certification matrix",)
        backend = "quantized"
    else:
        safe_tokens = 0
        production = False
        notes = ("quantized direct cache did not pass certification; use paged fallback",)
        backend = "paged"
    return QuantizedCacheCertificationProfile(
        family=target.family,
        model=target.model,
        model_architecture=report.family,
        transformers_version=str(report.metadata.get("transformers_version") or "") or None,
        torch_version=str(report.metadata.get("torch") or "") or None,
        gpu_arch=_gpu_arch_from_metadata(report.metadata),
        certified_backend=backend,
        quant_dtype="int8",
        safe_prompt_tokens=safe_tokens,
        production=production,
        notes=notes,
    )


def _conservative_profile(target: ModelCertificationTarget, reason: str) -> QuantizedCacheCertificationProfile:
    return QuantizedCacheCertificationProfile(
        family=target.family,
        model=target.model,
        certified_backend="paged",
        quant_dtype="int8",
        safe_prompt_tokens=0,
        production=False,
        notes=(reason,),
    )


def _resolve_local_model(root: Path | None, name: str, fallback: str) -> str:
    if root is None:
        return fallback
    candidates = (
        root / name,
        root / fallback.split("/")[-1],
        root / fallback.replace("/", "_"),
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _gpu_arch_from_metadata(metadata: dict[str, Any]) -> str | None:
    capability = str(metadata.get("gpu_capability") or "")
    if not capability:
        return None
    major = capability.split(".", 1)[0]
    return {
        "7": "turing",
        "8": "ampere",
        "9": "hopper",
        "10": "blackwell",
        "12": "blackwell",
    }.get(major, capability)


def _outcome_to_dict(outcome: ModelCertificationOutcome) -> dict[str, Any]:
    return {
        "target": asdict(outcome.target),
        "status": outcome.status,
        "profile": None if outcome.profile is None else outcome.profile.to_dict(),
        "error": outcome.error,
        "report": None if outcome.report is None else outcome.report.to_dict(),
    }
