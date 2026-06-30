from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .benchmark_store import BenchmarkRecord, collect_run_metadata, write_benchmark_csv, write_benchmark_json
from .hf_adapter import MemoryFirstHFConfig, _get_logits, memory_first_generate_hf, select_memory_adapter
from .hf_cache_profiles import select_quantized_cache_profile
from .quality_metrics import LogitQualityResult, TokenQualityResult, compare_logits, compare_token_sequences
from .report import make_table

HF_CACHE_CERTIFICATION_SCHEMA_VERSION = "llm_memlab.hf_cache_certification.v2"


@dataclass(frozen=True)
class HFCacheCertificationCase:
    prompt: str
    max_new_tokens: int = 1
    cache: str = "paged"
    quant_dtype: str = "int8"
    production: bool = True

    @property
    def name(self) -> str:
        tier = "prod" if self.production else "exp"
        return f"{tier}/prompt{len(self.prompt)}/tok{self.max_new_tokens}/{self.cache}/{self.quant_dtype}"


@dataclass(frozen=True)
class HFCacheCertificationResult:
    case: HFCacheCertificationCase
    prompt_tokens: int
    family: str
    adapter: str
    cache_impl: str
    requested_cache_impl: str
    direct_cache: bool
    production: bool
    token_quality: TokenQualityResult
    prefill_logit_quality: LogitQualityResult | None
    drift_metrics: dict[str, float | None]
    elapsed_ms: float
    peak_cuda_bytes: int | None
    profile_passed: bool | None = None
    profile_reasons: tuple[str, ...] = ()
    fallback_reason: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        logits_ok = self.prefill_logit_quality is None or self.prefill_logit_quality.passed
        profile_ok = self.profile_passed is not False
        base_ok = self.error is None and self.direct_cache and self.token_quality.exact_match and logits_ok and profile_ok
        return base_ok if self.production else True

    @property
    def failure_hint(self) -> str:
        if self.error:
            return self.error
        if not self.direct_cache:
            return self.fallback_reason or "adapter did not use direct cache"
        if not self.token_quality.exact_match:
            return f"generated token drift: agreement={self.token_quality.token_agreement:.3f}"
        if self.prefill_logit_quality is not None and not self.prefill_logit_quality.passed:
            return f"prefill logit drift: mean_abs={self.prefill_logit_quality.mean_abs_error:.6f}"
        if self.profile_passed is False:
            return "cache profile failed: " + "; ".join(self.profile_reasons)
        return "passed"

    def to_record(self, model: str, metadata: dict[str, Any]) -> BenchmarkRecord:
        logit = self.prefill_logit_quality
        return BenchmarkRecord(
            name=f"hf-cache-cert/{self.family}/{self.case.name}",
            kind="hf-cache-certification",
            mean_ms=self.elapsed_ms,
            min_ms=self.elapsed_ms,
            max_ms=self.elapsed_ms,
            peak_cuda_bytes=self.peak_cuda_bytes,
            extra={
                "schema_version": HF_CACHE_CERTIFICATION_SCHEMA_VERSION,
                "model": model,
                "adapter": self.adapter,
                "cache_type": self.case.cache,
                "cache_impl": self.cache_impl,
                "requested_cache_impl": self.requested_cache_impl,
                "direct_cache": self.direct_cache,
                "fallback": self.fallback_reason is not None or self.cache_impl != self.requested_cache_impl,
                "production": self.production,
                "quality_passed": self.passed,
                "fallback_reason": self.fallback_reason,
                "failure_hint": self.failure_hint,
                "prompt_tokens": self.prompt_tokens,
                "prompt_length": len(self.case.prompt),
                "generated_exact_match": self.token_quality.exact_match,
                "generated_token_agreement": self.token_quality.token_agreement,
                "prefill_mean_abs": None if logit is None else logit.mean_abs_error,
                "prefill_max_abs": None if logit is None else logit.max_abs_error,
                "prefill_top1": None if logit is None else logit.top1_agreement,
                "prefill_topk": None if logit is None else logit.topk_agreement,
                "profile_passed": self.profile_passed,
                "profile_reasons": list(self.profile_reasons),
                **self.drift_metrics,
            },
            metadata=metadata,
        )


@dataclass(frozen=True)
class HFCacheCertificationReport:
    model: str
    family: str
    results: tuple[HFCacheCertificationResult, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = HF_CACHE_CERTIFICATION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    def to_records(self) -> list[BenchmarkRecord]:
        common = {"schema_version": self.schema_version, **self.metadata}
        return [result.to_record(self.model, common) for result in self.results]

    def write_json(self, path: str | Path) -> Path:
        return write_benchmark_json(self.to_records(), path)

    def write_csv(self, path: str | Path) -> Path:
        return write_benchmark_csv(self.to_records(), path)

    def write_html(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(hf_cache_certification_to_html(self), encoding="utf-8")
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model,
            "family": self.family,
            "passed": self.passed,
            "metadata": self.metadata,
            "results": [asdict(result) for result in self.results],
        }

    def to_text(self) -> str:
        rows = [
            (
                result.case.name,
                result.adapter,
                result.cache_impl,
                result.requested_cache_impl,
                result.direct_cache,
                result.production,
                f"{result.token_quality.token_agreement:.1%}",
                _fmt_logit(result.prefill_logit_quality),
                _fmt_metric(result.drift_metrics.get("kv_mean_abs")),
                "PASS" if result.passed else "FAIL",
                result.failure_hint,
            )
            for result in self.results
        ]
        header = make_table(
            ("Metric", "Value"),
            [
                ("Schema", self.schema_version),
                ("Model", self.model),
                ("Family", self.family),
                ("Cases", len(self.results)),
                ("Passed", self.passed),
            ],
        )
        return (
            header
            + "\n\n"
            + make_table(
                (
                    "Case",
                    "Adapter",
                    "Cache",
                    "Requested",
                    "Direct",
                    "Production",
                    "Token agreement",
                    "Prefill mean abs",
                    "KV mean abs",
                    "Status",
                    "Hint",
                ),
                rows,
            )
        )


def certify_hf_cache(
    model_id: str,
    *,
    prompts: Iterable[str] = ("hello",),
    token_counts: Iterable[int] = (1,),
    caches: Iterable[str] = ("paged",),
    experimental_caches: Iterable[str] = (),
    quant_dtypes: Iterable[str] = ("int8",),
    device: str | None = None,
    dtype: str = "auto",
    local_files_only: bool = False,
    allow_experimental_direct_cache: bool = True,
    max_logit_mean_abs: float = 0.02,
    min_logit_top1: float = 0.98,
) -> HFCacheCertificationReport:
    torch = _import_torch()
    AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    device = _resolve_device(torch, device)
    torch_dtype = _resolve_dtype(torch, dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    load_kwargs: dict[str, Any] = {"local_files_only": local_files_only, "low_cpu_mem_usage": True}
    if torch_dtype is not None:
        load_kwargs["dtype"] = torch_dtype
    if device == "cuda":
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    try:
        return certify_hf_cache_model(
            model,
            tokenizer,
            model_name=model_id,
            prompts=prompts,
            token_counts=token_counts,
            caches=caches,
            experimental_caches=experimental_caches,
            quant_dtypes=quant_dtypes,
            device=device,
            allow_experimental_direct_cache=allow_experimental_direct_cache,
            max_logit_mean_abs=max_logit_mean_abs,
            min_logit_top1=min_logit_top1,
            dtype=str(torch_dtype or dtype),
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def certify_hf_cache_model(
    model: Any,
    tokenizer: Any,
    *,
    model_name: str = "model",
    prompts: Iterable[str] = ("hello",),
    token_counts: Iterable[int] = (1,),
    caches: Iterable[str] = ("paged",),
    experimental_caches: Iterable[str] = (),
    quant_dtypes: Iterable[str] = ("int8",),
    device: str | None = None,
    allow_experimental_direct_cache: bool = True,
    max_logit_mean_abs: float = 0.02,
    min_logit_top1: float = 0.98,
    dtype: str = "auto",
) -> HFCacheCertificationReport:
    torch = _import_torch()
    family = str(getattr(getattr(model, "config", None), "model_type", model.__class__.__name__)).lower()
    results: list[HFCacheCertificationResult] = []
    metadata = collect_run_metadata(dtype=dtype, backend="hf-cache-certification").__dict__
    metadata.update(
        {
            "allow_experimental_direct_cache": allow_experimental_direct_cache,
            "device": device,
            "transformers_version": _transformers_version(),
        }
    )
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        if device:
            encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        for max_new_tokens in token_counts:
            cache_items = [(cache, True) for cache in caches] + [(cache, False) for cache in experimental_caches]
            for cache, production in cache_items:
                for quant_dtype in quant_dtypes:
                    case = HFCacheCertificationCase(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        cache=cache,
                        quant_dtype=quant_dtype,
                        production=production,
                    )
                    results.append(
                        _run_case(
                            torch,
                            model,
                            encoded,
                            case,
                            prompt_tokens=prompt_tokens,
                            allow_experimental_direct_cache=allow_experimental_direct_cache,
                            max_logit_mean_abs=max_logit_mean_abs,
                            min_logit_top1=min_logit_top1,
                        )
                    )
    return HFCacheCertificationReport(model=model_name, family=family, results=tuple(results), metadata=metadata)


def assert_hf_cache_certified(report: HFCacheCertificationReport) -> None:
    failures = [result for result in report.results if result.production and not result.passed]
    if failures:
        detail = "; ".join(f"{result.case.name}: {result.failure_hint}" for result in failures)
        raise AssertionError("HF cache certification failed: " + detail)


def hf_cache_certification_to_html(report: HFCacheCertificationReport) -> str:
    rows = []
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        fallback = result.fallback_reason or "n/a"
        logit = _fmt_logit(result.prefill_logit_quality)
        kv = _fmt_metric(result.drift_metrics.get("kv_mean_abs"))
        peak = _fmt_bytes(result.peak_cuda_bytes)
        rows.append(
            "<tr>"
            f"<td>{_e(result.case.name)}</td>"
            f"<td>{_e(result.cache_impl)}</td>"
            f"<td>{_e(result.requested_cache_impl)}</td>"
            f"<td>{result.direct_cache}</td>"
            f"<td>{result.production}</td>"
            f"<td>{result.token_quality.token_agreement:.1%}</td>"
            f"<td>{_e(logit)}</td>"
            f"<td>{_e(kv)}</td>"
            f"<td>{_e(peak)}</td>"
            f"<td>{_e(fallback)}</td>"
            f"<td>{status}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>llm-memlab HF cache certification</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #17202a; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px; text-align: left; vertical-align: top; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .metric {{ border: 1px solid #d8dee9; padding: 10px; border-radius: 6px; }}
    .label {{ color: #52606d; font-size: 12px; }}
    .value {{ font-size: 18px; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>HF cache certification</h1>
  <div class="summary">
    <div class="metric"><div class="label">Model</div><div class="value">{_e(report.model)}</div></div>
    <div class="metric"><div class="label">Family</div><div class="value">{_e(report.family)}</div></div>
    <div class="metric"><div class="label">Cases</div><div class="value">{len(report.results)}</div></div>
    <div class="metric"><div class="label">Passed</div><div class="value">{report.passed}</div></div>
  </div>
  <table>
    <thead>
      <tr><th>Case</th><th>Effective cache</th><th>Requested</th><th>Direct</th><th>Production</th><th>Token match</th><th>Prefill drift</th><th>KV drift</th><th>Peak memory</th><th>Fallback</th><th>Status</th></tr>
    </thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</body>
</html>"""


def _run_case(
    torch,
    model,
    encoded: dict[str, Any],
    case: HFCacheCertificationCase,
    *,
    prompt_tokens: int,
    allow_experimental_direct_cache: bool,
    max_logit_mean_abs: float,
    min_logit_top1: float,
) -> HFCacheCertificationResult:
    adapter = select_memory_adapter(
        model, MemoryFirstHFConfig(cache=case.cache, quant_dtype=case.quant_dtype, max_new_tokens=case.max_new_tokens)
    )
    started = time.perf_counter()
    _reset_peak(torch)
    try:
        with torch.inference_mode():
            baseline = model.generate(**encoded, max_new_tokens=case.max_new_tokens, do_sample=False, use_cache=True)
            config = MemoryFirstHFConfig(
                cache=case.cache,
                quant_dtype=case.quant_dtype,
                model_name=str(getattr(getattr(model, "config", None), "_name_or_path", "") or ""),
                max_new_tokens=case.max_new_tokens,
                allow_experimental_direct_cache=allow_experimental_direct_cache,
            )
            model_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}
            candidate = memory_first_generate_hf(
                model,
                encoded["input_ids"],
                config,
                max_new_tokens=case.max_new_tokens,
                do_sample=False,
                **model_kwargs,
            )
            effective_config = _effective_config_from_result(config, candidate)
            logits = _compare_prefill_logits(
                model,
                encoded,
                config=effective_config,
                max_mean_abs=max_logit_mean_abs,
                min_top1=min_logit_top1,
            )
            drift_metrics = _compare_paged_vs_quantized_cache_drift(model, encoded, config)
            profile_passed, profile_reasons = _evaluate_cache_profile(
                model,
                prompt_tokens=prompt_tokens,
                cache=case.cache,
                quant_dtype=case.quant_dtype,
                drift_metrics=drift_metrics,
            )
        _sync(torch)
        elapsed_ms = (time.perf_counter() - started) * 1000
        token_quality = compare_token_sequences(baseline[:, : candidate.sequences.shape[-1]], candidate.sequences)
        return HFCacheCertificationResult(
            case=case,
            prompt_tokens=prompt_tokens,
            family=str(getattr(getattr(model, "config", None), "model_type", model.__class__.__name__)).lower(),
            adapter=adapter.__class__.__name__,
            cache_impl=candidate.cache_impl,
            requested_cache_impl=candidate.requested_cache_impl or f"{adapter.family}:{case.cache}",
            direct_cache=candidate.direct_cache,
            production=case.production,
            fallback_reason=candidate.fallback_reason,
            token_quality=token_quality,
            prefill_logit_quality=logits,
            drift_metrics=drift_metrics,
            elapsed_ms=elapsed_ms,
            peak_cuda_bytes=_peak(torch),
            profile_passed=profile_passed,
            profile_reasons=profile_reasons,
        )
    except Exception as exc:
        _sync(torch)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return HFCacheCertificationResult(
            case=case,
            prompt_tokens=prompt_tokens,
            family=str(getattr(getattr(model, "config", None), "model_type", model.__class__.__name__)).lower(),
            adapter=adapter.__class__.__name__,
            cache_impl=f"{case.cache}:{case.quant_dtype}",
            requested_cache_impl=f"{case.cache}:{case.quant_dtype}",
            direct_cache=False,
            production=case.production,
            token_quality=TokenQualityResult(False, 0.0, 0, 0),
            prefill_logit_quality=None,
            drift_metrics={},
            elapsed_ms=elapsed_ms,
            peak_cuda_bytes=_peak(torch),
            error=str(exc)[:500],
        )


def _evaluate_cache_profile(
    model: Any,
    *,
    prompt_tokens: int,
    cache: str,
    quant_dtype: str,
    drift_metrics: dict[str, float | None],
) -> tuple[bool | None, tuple[str, ...]]:
    if cache != "quantized":
        return None, ()
    family = str(getattr(getattr(model, "config", None), "model_type", model.__class__.__name__)).lower()
    model_name = str(getattr(getattr(model, "config", None), "_name_or_path", "") or "")
    profile = select_quantized_cache_profile(family=family, model=model_name, quant_dtype=quant_dtype)
    return profile.evaluate(prompt_tokens=prompt_tokens, drift_metrics=drift_metrics)


def _compare_prefill_logits(
    model: Any,
    encoded: dict[str, Any],
    *,
    config: MemoryFirstHFConfig,
    max_mean_abs: float,
    min_top1: float,
) -> LogitQualityResult:
    adapter = select_memory_adapter(model, config)
    input_ids = encoded["input_ids"]
    model_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}
    baseline = model(input_ids=input_ids, use_cache=True, **model_kwargs)
    cache = adapter.make_cache(input_ids, max_new_tokens=config.max_new_tokens)
    candidate_kwargs = adapter._prepare_family_kwargs(input_ids, {**model_kwargs, "use_cache": True, "past_key_values": cache})
    candidate = model(input_ids=input_ids, **candidate_kwargs)
    return compare_logits(_get_logits(baseline)[:, -1:, :], _get_logits(candidate)[:, -1:, :], max_mean_abs=max_mean_abs, min_top1=min_top1)


def _effective_config_from_result(config: MemoryFirstHFConfig, result) -> MemoryFirstHFConfig:
    cache_impl = str(result.cache_impl)
    if "paged" in cache_impl and config.cache != "paged":
        return MemoryFirstHFConfig(
            cache="paged",
            quant_dtype=config.quant_dtype,
            page_size=config.page_size,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            allow_experimental_direct_cache=config.allow_experimental_direct_cache,
            allow_experimental_quantized_cache=config.allow_experimental_quantized_cache,
            quantized_prefill_token_limit=config.quantized_prefill_token_limit,
        )
    return config


def _compare_paged_vs_quantized_cache_drift(model: Any, encoded: dict[str, Any], config: MemoryFirstHFConfig) -> dict[str, float | None]:
    if config.cache != "quantized":
        return {"kv_mean_abs": None, "kv_max_abs": None, "attention_proxy_mean_abs": None, "logit_mean_abs_paged_vs_quantized": None}
    torch = _import_torch()
    input_ids = encoded["input_ids"]
    model_kwargs = {key: value for key, value in encoded.items() if key != "input_ids"}
    paged_cfg = MemoryFirstHFConfig(
        cache="paged",
        quant_dtype=config.quant_dtype,
        page_size=config.page_size,
        max_new_tokens=config.max_new_tokens,
        allow_experimental_direct_cache=True,
        allow_experimental_quantized_cache=True,
    )
    quant_cfg = MemoryFirstHFConfig(
        cache="quantized",
        quant_dtype=config.quant_dtype,
        page_size=config.page_size,
        max_new_tokens=config.max_new_tokens,
        allow_experimental_direct_cache=True,
        allow_experimental_quantized_cache=True,
    )
    paged_adapter = select_memory_adapter(model, paged_cfg)
    quant_adapter = select_memory_adapter(model, quant_cfg)
    paged_cache = paged_adapter.make_cache(input_ids, max_new_tokens=config.max_new_tokens)
    quant_cache = quant_adapter.make_cache(input_ids, max_new_tokens=config.max_new_tokens)
    paged_kwargs = paged_adapter._prepare_family_kwargs(input_ids, {**model_kwargs, "use_cache": True, "past_key_values": paged_cache})
    quant_kwargs = quant_adapter._prepare_family_kwargs(input_ids, {**model_kwargs, "use_cache": True, "past_key_values": quant_cache})
    paged_out = model(input_ids=input_ids, **paged_kwargs)
    quant_out = model(input_ids=input_ids, **quant_kwargs)
    kv_diffs = []
    attention_diffs = []
    for layer_idx in range(paged_cache.config.num_layers):
        pk, pv = paged_cache.storage.get_layer(layer_idx)
        qk, qv = quant_cache.storage.get_layer(layer_idx)
        kv_diffs.append((pk.float() - qk.float()).abs().flatten())
        kv_diffs.append((pv.float() - qv.float()).abs().flatten())
        if pk.shape[-2] > 0:
            query = pk[:, :, -1:, :].float()
            p_attn = torch.nn.functional.scaled_dot_product_attention(query, pk.float(), pv.float())
            q_attn = torch.nn.functional.scaled_dot_product_attention(query, qk.float(), qv.float())
            attention_diffs.append((p_attn - q_attn).abs().flatten())
    kv_all = torch.cat(kv_diffs) if kv_diffs else torch.zeros(1, device=input_ids.device)
    attn_all = torch.cat(attention_diffs) if attention_diffs else torch.zeros(1, device=input_ids.device)
    logit_quality = compare_logits(_get_logits(paged_out)[:, -1:, :], _get_logits(quant_out)[:, -1:, :])
    return {
        "kv_mean_abs": float(kv_all.mean().item()),
        "kv_max_abs": float(kv_all.max().item()),
        "attention_proxy_mean_abs": float(attn_all.mean().item()),
        "logit_mean_abs_paged_vs_quantized": logit_quality.mean_abs_error,
    }


def _fmt_logit(value: LogitQualityResult | None) -> str:
    if value is None:
        return "n/a"
    return f"{value.mean_abs_error:.6f}"


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1024**3:.3f} GiB"


def _e(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _transformers_version() -> str | None:
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:
        return None


def _resolve_device(torch, device: str | None) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(torch, dtype: str, device: str):
    value = dtype.lower()
    if value == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be auto, fp16, bf16, or fp32")


def _reset_peak(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _sync(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak(torch) -> int | None:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("HF cache certification requires PyTorch") from exc
    return torch


def _import_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("HF cache certification requires transformers") from exc
    return AutoModelForCausalLM, AutoTokenizer
