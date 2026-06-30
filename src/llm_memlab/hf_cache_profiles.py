from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .report import make_table

HF_CACHE_PROFILE_SCHEMA_VERSION = "llm_memlab.hf_cache_profile.v1"


@dataclass(frozen=True)
class QuantizedCacheCertificationProfile:
    family: str
    model: str = "*"
    model_architecture: str | None = None
    transformers_version: str | None = None
    torch_version: str | None = None
    gpu_arch: str | None = None
    certified_backend: str = "paged"
    quant_dtype: str = "int8"
    safe_prompt_tokens: int = 1
    kv_mean_abs_threshold: float = 0.08
    kv_max_abs_threshold: float = 2.0
    attention_mean_abs_threshold: float = 0.05
    logit_mean_abs_threshold: float = 0.02
    production: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = HF_CACHE_PROFILE_SCHEMA_VERSION

    def matches(self, *, family: str, model: str | None = None, quant_dtype: str = "int8") -> bool:
        family_ok = family.lower().startswith(self.family.lower())
        model_ok = self.model == "*" or (model is not None and self.model.lower() in model.lower())
        dtype_ok = self.quant_dtype.lower() == quant_dtype.lower()
        return family_ok and model_ok and dtype_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family": self.family,
            "model": self.model,
            "model_architecture": self.model_architecture,
            "transformers_version": self.transformers_version,
            "torch_version": self.torch_version,
            "gpu_arch": self.gpu_arch,
            "certified_backend": self.certified_backend,
            "quant_dtype": self.quant_dtype,
            "safe_prompt_tokens": self.safe_prompt_tokens,
            "kv_mean_abs_threshold": self.kv_mean_abs_threshold,
            "kv_max_abs_threshold": self.kv_max_abs_threshold,
            "attention_mean_abs_threshold": self.attention_mean_abs_threshold,
            "logit_mean_abs_threshold": self.logit_mean_abs_threshold,
            "production": self.production,
            "notes": list(self.notes),
        }

    def evaluate(self, *, prompt_tokens: int, drift_metrics: dict[str, Any] | None = None) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if prompt_tokens > self.safe_prompt_tokens:
            reasons.append(f"prompt tokens {prompt_tokens} > certified limit {self.safe_prompt_tokens}")
        metrics = drift_metrics or {}
        _check_metric(metrics, "kv_mean_abs", self.kv_mean_abs_threshold, reasons)
        _check_metric(metrics, "kv_max_abs", self.kv_max_abs_threshold, reasons)
        _check_metric(metrics, "attention_proxy_mean_abs", self.attention_mean_abs_threshold, reasons)
        _check_metric(metrics, "logit_mean_abs_paged_vs_quantized", self.logit_mean_abs_threshold, reasons)
        return not reasons, tuple(reasons)

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Family", self.family),
                ("Model", self.model),
                ("Architecture", self.model_architecture or "n/a"),
                ("GPU arch", self.gpu_arch or "n/a"),
                ("Certified backend", self.certified_backend),
                ("Quant dtype", self.quant_dtype),
                ("Safe prompt tokens", self.safe_prompt_tokens),
                ("KV mean threshold", self.kv_mean_abs_threshold),
                ("KV max threshold", self.kv_max_abs_threshold),
                ("Attention threshold", self.attention_mean_abs_threshold),
                ("Logit threshold", self.logit_mean_abs_threshold),
                ("Production", self.production),
                ("Notes", "; ".join(self.notes)),
            ],
        )


DEFAULT_QUANTIZED_CACHE_PROFILES: tuple[QuantizedCacheCertificationProfile, ...] = tuple(
    QuantizedCacheCertificationProfile(
        family=family,
        quant_dtype=dtype,
        safe_prompt_tokens=1,
        kv_mean_abs_threshold=0.08 if family == "qwen3" else 0.06,
        kv_max_abs_threshold=2.0,
        attention_mean_abs_threshold=0.05,
        logit_mean_abs_threshold=0.02,
        production=False,
        notes=(
            "single-token direct quantized cache only; multi-token prefill falls back to paged until certified"
            if family == "qwen3"
            else "conservative default pending per-model certification",
        ),
    )
    for family in ("llama", "qwen", "qwen3", "mistral", "mixtral", "gemma", "phi", "deepseek", "gpt_neox", "falcon")
    for dtype in ("int8", "uint8")
)


@dataclass(frozen=True)
class CacheProfileRegistry:
    profiles: tuple[QuantizedCacheCertificationProfile, ...] = DEFAULT_QUANTIZED_CACHE_PROFILES
    source: str = "builtin"

    def select(self, *, family: str, model: str | None = None, quant_dtype: str = "int8") -> QuantizedCacheCertificationProfile:
        return select_quantized_cache_profile(family=family, model=model, quant_dtype=quant_dtype, profiles=self.profiles)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HF_CACHE_PROFILE_SCHEMA_VERSION,
            "source": self.source,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }


def select_quantized_cache_profile(
    *,
    family: str,
    model: str | None = None,
    quant_dtype: str = "int8",
    profiles: tuple[QuantizedCacheCertificationProfile, ...] | list[QuantizedCacheCertificationProfile] | None = None,
) -> QuantizedCacheCertificationProfile:
    selected = tuple(profiles or DEFAULT_QUANTIZED_CACHE_PROFILES)
    for profile in selected:
        if profile.matches(family=family, model=model, quant_dtype=quant_dtype):
            return profile
    return QuantizedCacheCertificationProfile(
        family=family,
        model=model or "*",
        quant_dtype=quant_dtype,
        safe_prompt_tokens=0,
        production=False,
        notes=("no profile found; production policy falls back to paged",),
    )


def load_quantized_cache_profiles(paths: list[str | Path] | tuple[str | Path, ...]) -> CacheProfileRegistry:
    profiles: list[QuantizedCacheCertificationProfile] = []
    sources: list[str] = []
    for path in paths:
        loaded = _load_profile_file(path)
        profiles.extend(loaded)
        sources.append(str(path))
    profiles.extend(DEFAULT_QUANTIZED_CACHE_PROFILES)
    return CacheProfileRegistry(tuple(profiles), source=",".join(sources) if sources else "builtin")


def write_quantized_cache_profiles(
    profiles: list[QuantizedCacheCertificationProfile] | tuple[QuantizedCacheCertificationProfile, ...], path: str | Path
) -> Path:
    output = Path(path)
    payload = {
        "schema_version": HF_CACHE_PROFILE_SCHEMA_VERSION,
        "profiles": [profile.to_dict() for profile in profiles],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _load_profile_file(path: str | Path) -> list[QuantizedCacheCertificationProfile]:
    profile_path = Path(path)
    text = profile_path.read_text(encoding="utf-8")
    if profile_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML profile files require PyYAML; use JSON or install pyyaml") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    items = data.get("profiles", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("profile registry must contain a list or a {'profiles': [...]} object")
    return [_profile_from_dict(item) for item in items]


def _profile_from_dict(item: dict[str, Any]) -> QuantizedCacheCertificationProfile:
    payload = dict(item)
    payload.pop("schema_version", None)
    if "notes" in payload and isinstance(payload["notes"], list):
        payload["notes"] = tuple(str(note) for note in payload["notes"])
    return QuantizedCacheCertificationProfile(**payload)


def profile_from_certification_result(result: Any) -> QuantizedCacheCertificationProfile:
    extra = getattr(result, "to_record", lambda: None)()
    payload = getattr(extra, "extra", None) or {}
    model = str(payload.get("model") or "*")
    family = str(getattr(getattr(result, "case", None), "family", "") or payload.get("adapter", "generic")).lower()
    family = family.replace("memoryadapter", "").replace("adapter", "") or "generic"
    quant_dtype = str(getattr(getattr(result, "case", None), "quant_dtype", None) or payload.get("quant_dtype") or "int8")
    prompt_tokens = int(payload.get("prompt_tokens") or payload.get("prompt_length") or 0)
    production = bool(payload.get("quality_passed")) and not bool(payload.get("fallback"))
    return QuantizedCacheCertificationProfile(
        family=family,
        model=model,
        model_architecture=str(payload.get("model_architecture") or "") or None,
        transformers_version=str(payload.get("transformers_version") or "") or None,
        torch_version=str(payload.get("torch_version") or "") or None,
        gpu_arch=str(payload.get("gpu_arch") or "") or None,
        certified_backend=str(payload.get("certified_backend") or "paged"),
        quant_dtype=quant_dtype,
        safe_prompt_tokens=prompt_tokens if production else 0,
        kv_mean_abs_threshold=float(payload.get("kv_mean_abs") or 0.08),
        kv_max_abs_threshold=float(payload.get("kv_max_abs") or 2.0),
        attention_mean_abs_threshold=float(payload.get("attention_proxy_mean_abs") or 0.05),
        logit_mean_abs_threshold=float(payload.get("logit_mean_abs_paged_vs_quantized") or payload.get("prefill_mean_abs") or 0.02),
        production=production,
        notes=("derived from certification result",),
    )


def promote_profile(
    profile: QuantizedCacheCertificationProfile, *, safe_prompt_tokens: int | None = None
) -> QuantizedCacheCertificationProfile:
    return replace(profile, production=True, safe_prompt_tokens=safe_prompt_tokens or profile.safe_prompt_tokens)


def _check_metric(metrics: dict[str, Any], name: str, threshold: float, reasons: list[str]) -> None:
    value = metrics.get(name)
    if value is None:
        return
    try:
        if float(value) > threshold:
            reasons.append(f"{name} {float(value):.6f} > {threshold:.6f}")
    except (TypeError, ValueError):
        reasons.append(f"{name} is not numeric")
