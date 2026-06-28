from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .report import make_table


@dataclass(frozen=True)
class LogitQualityResult:
    mean_abs_error: float
    max_abs_error: float
    cosine_similarity: float
    kl_divergence: float
    top1_agreement: float
    topk_agreement: float
    passed: bool

    def to_text(self) -> str:
        return make_table(("Metric", "Value"), [
            ("Mean abs error", f"{self.mean_abs_error:.6f}"),
            ("Max abs error", f"{self.max_abs_error:.6f}"),
            ("Cosine similarity", f"{self.cosine_similarity:.6f}"),
            ("KL divergence", f"{self.kl_divergence:.6f}"),
            ("Top-1 agreement", f"{self.top1_agreement:.1%}"),
            ("Top-k agreement", f"{self.topk_agreement:.1%}"),
            ("Passed", self.passed),
        ])


@dataclass(frozen=True)
class TokenQualityResult:
    exact_match: bool
    token_agreement: float
    baseline_length: int
    candidate_length: int

    def to_text(self) -> str:
        return make_table(("Metric", "Value"), [
            ("Exact match", self.exact_match),
            ("Token agreement", f"{self.token_agreement:.1%}"),
            ("Baseline length", self.baseline_length),
            ("Candidate length", self.candidate_length),
        ])


def compare_logits(baseline, candidate, *, top_k: int = 5, max_mean_abs: float = 0.02, min_top1: float = 0.98) -> LogitQualityResult:
    torch = _import_torch()
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if baseline.shape != candidate.shape:
        raise ValueError(f"logit shapes must match, got {tuple(baseline.shape)} and {tuple(candidate.shape)}")
    base = baseline.detach().float()
    cand = candidate.detach().float()
    diff = (base - cand).abs()
    cosine = torch.nn.functional.cosine_similarity(base.flatten(), cand.flatten(), dim=0)
    base_logp = torch.nn.functional.log_softmax(base, dim=-1)
    cand_logp = torch.nn.functional.log_softmax(cand, dim=-1)
    base_p = base_logp.exp()
    kl = torch.nn.functional.kl_div(cand_logp, base_p, reduction="batchmean")
    top1 = (base.argmax(dim=-1) == cand.argmax(dim=-1)).float().mean()
    k = min(top_k, base.shape[-1])
    base_topk = base.topk(k, dim=-1).indices
    cand_topk = cand.topk(k, dim=-1).indices
    overlap = (base_topk[..., :, None] == cand_topk[..., None, :]).any(dim=-1).float().mean()
    mean_abs = float(diff.mean().item())
    top1_value = float(top1.item())
    return LogitQualityResult(
        mean_abs_error=mean_abs,
        max_abs_error=float(diff.max().item()),
        cosine_similarity=float(cosine.item()),
        kl_divergence=float(kl.item()),
        top1_agreement=top1_value,
        topk_agreement=float(overlap.item()),
        passed=mean_abs <= max_mean_abs and top1_value >= min_top1,
    )


def compare_token_sequences(baseline: Any, candidate: Any) -> TokenQualityResult:
    base = _to_list(baseline)
    cand = _to_list(candidate)
    limit = min(len(base), len(cand))
    matches = sum(1 for idx in range(limit) if base[idx] == cand[idx])
    denom = max(len(base), len(cand), 1)
    return TokenQualityResult(
        exact_match=base == cand,
        token_agreement=matches / denom,
        baseline_length=len(base),
        candidate_length=len(cand),
    )


def _to_list(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        return [int(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, (list, tuple)) or hasattr(item, "detach"):
                out.extend(_to_list(item))
            else:
                out.append(int(item))
        return out
    return [int(value)]


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Output quality metrics require PyTorch. Install with: pip install torch") from exc
    return torch
