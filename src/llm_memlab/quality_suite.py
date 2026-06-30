from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .kv_cache import DecodeConfig, greedy_decode
from .quality_metrics import LogitQualityResult, TokenQualityResult, compare_logits, compare_token_sequences
from .report import make_table

QUALITY_SCHEMA_VERSION = "llm_memlab.quality.v1"


@dataclass(frozen=True)
class QualityThresholds:
    top_k: int = 5
    max_mean_abs: float = 0.02
    min_top1: float = 0.98
    min_topk: float = 0.95
    min_token_agreement: float = 1.0
    max_loss_delta: float = 0.05

    def validate(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= self.min_top1 <= 1.0 or not 0.0 <= self.min_topk <= 1.0 or not 0.0 <= self.min_token_agreement <= 1.0:
            raise ValueError("agreement thresholds must be between 0 and 1")
        if self.max_mean_abs < 0 or self.max_loss_delta < 0:
            raise ValueError("error/loss thresholds must be non-negative")


@dataclass(frozen=True)
class QualityRegressionResult:
    logits: LogitQualityResult
    tokens: TokenQualityResult
    baseline_loss: float | None
    candidate_loss: float | None
    thresholds: QualityThresholds
    schema_version: str = QUALITY_SCHEMA_VERSION

    @property
    def max_loss_delta(self) -> float:
        return self.thresholds.max_loss_delta

    @property
    def loss_delta(self) -> float | None:
        if self.baseline_loss is None or self.candidate_loss is None:
            return None
        return self.candidate_loss - self.baseline_loss

    @property
    def passed(self) -> bool:
        delta = self.loss_delta
        loss_ok = True if delta is None else delta <= self.thresholds.max_loss_delta
        return bool(
            self.logits.passed
            and self.logits.topk_agreement >= self.thresholds.min_topk
            and self.tokens.token_agreement >= self.thresholds.min_token_agreement
            and loss_ok
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logits": asdict(self.logits),
            "tokens": asdict(self.tokens),
            "baseline_loss": self.baseline_loss,
            "candidate_loss": self.candidate_loss,
            "loss_delta": self.loss_delta,
            "thresholds": asdict(self.thresholds),
            "passed": self.passed,
        }

    def to_text(self) -> str:
        rows = [
            ("Schema", self.schema_version),
            ("Passed", self.passed),
            ("Logit mean abs", f"{self.logits.mean_abs_error:.6f}"),
            ("Top-1", f"{self.logits.top1_agreement:.1%}"),
            ("Top-k", f"{self.logits.topk_agreement:.1%}"),
            ("Token agreement", f"{self.tokens.token_agreement:.1%}"),
            ("Baseline loss", _fmt(self.baseline_loss)),
            ("Candidate loss", _fmt(self.candidate_loss)),
            ("Loss delta", _fmt(self.loss_delta)),
            ("Thresholds", asdict(self.thresholds)),
        ]
        return make_table(("Metric", "Value"), rows)


def run_quality_regression(
    baseline_model: Any,
    candidate_model: Any,
    input_ids,
    *,
    max_new_tokens: int = 4,
    top_k: int = 5,
    max_mean_abs: float = 0.02,
    min_top1: float = 0.98,
    max_loss_delta: float = 0.05,
    thresholds: QualityThresholds | None = None,
    **model_kwargs,
) -> QualityRegressionResult:
    torch = _import_torch()
    cfg = thresholds or QualityThresholds(top_k=top_k, max_mean_abs=max_mean_abs, min_top1=min_top1, max_loss_delta=max_loss_delta)
    cfg.validate()
    with torch.no_grad():
        baseline_out = baseline_model(input_ids, **model_kwargs)
        candidate_out = candidate_model(input_ids, **model_kwargs)
    baseline_logits = _get_logits(baseline_out)
    candidate_logits = _get_logits(candidate_out)
    logits = compare_logits(baseline_logits, candidate_logits, top_k=cfg.top_k, max_mean_abs=cfg.max_mean_abs, min_top1=cfg.min_top1)
    decode_cfg = DecodeConfig(max_new_tokens=max_new_tokens, use_cache=True)
    baseline_decode = greedy_decode(baseline_model, input_ids, decode_cfg, **model_kwargs)
    candidate_decode = greedy_decode(candidate_model, input_ids, decode_cfg, **model_kwargs)
    tokens = compare_token_sequences(baseline_decode.sequences, candidate_decode.sequences)
    baseline_loss = _next_token_loss(torch, baseline_logits, input_ids)
    candidate_loss = _next_token_loss(torch, candidate_logits, input_ids)
    return QualityRegressionResult(logits, tokens, baseline_loss, candidate_loss, cfg)


def assert_quality_regression(result: QualityRegressionResult) -> None:
    if not result.passed:
        raise AssertionError("Quality regression failed:\n" + result.to_text())


def _next_token_loss(torch, logits, input_ids) -> float | None:
    if input_ids.shape[-1] < 2:
        return None
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction="mean")
    return float(loss.item())


def _get_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    raise TypeError("Model output must expose logits")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Quality regression suite requires PyTorch") from exc
    return torch
