from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hardware import HardwareProfile, detect_hardware_profile
from .hf_cache_profiles import QuantizedCacheCertificationProfile, load_quantized_cache_profiles, select_quantized_cache_profile
from .report import make_table


@dataclass(frozen=True)
class HFCachePolicy:
    requested_cache: str = "auto"
    quant_dtype: str = "int8"
    model: str | None = None
    production: bool = True
    qwen3_quantized_prefill_limit: int = 1
    allow_experimental_quantized: bool = False
    allow_experimental_cutile: bool = False
    quantized_profiles: tuple[QuantizedCacheCertificationProfile, ...] | None = None
    quantized_profile_paths: tuple[str, ...] = ()
    hardware_profile: HardwareProfile | None = None


@dataclass(frozen=True)
class HFCachePolicyDecision:
    cache: str
    quant_dtype: str
    direct_cache_allowed: bool
    quantized_allowed: bool
    cutile_backend: str
    production: bool
    profile: QuantizedCacheCertificationProfile | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Cache", self.cache),
                ("Quant dtype", self.quant_dtype),
                ("Direct cache", self.direct_cache_allowed),
                ("Quantized", self.quantized_allowed),
                ("CuTile backend", self.cutile_backend),
                ("Production", self.production),
                ("Profile", self.profile.family if self.profile is not None else "n/a"),
                ("Reasons", "; ".join(self.reasons)),
            ],
        )


def select_hf_cache_policy(
    *,
    family: str,
    prompt_tokens: int,
    device: Any = None,
    policy: HFCachePolicy | None = None,
) -> HFCachePolicyDecision:
    cfg = policy or HFCachePolicy()
    family = family.lower()
    requested = cfg.requested_cache.lower()
    reasons: list[str] = []

    cache = "paged" if requested == "auto" else requested
    quantized_allowed = cache == "quantized"
    direct_cache_allowed = True
    production = cfg.production
    hardware = cfg.hardware_profile or detect_hardware_profile(device)

    profile = None
    if requested == "quantized" or cache == "quantized":
        profiles = cfg.quantized_profiles
        if cfg.quantized_profile_paths:
            registry = load_quantized_cache_profiles(list(cfg.quantized_profile_paths))
            profiles = registry.profiles if profiles is None else tuple(profiles) + tuple(registry.profiles)
        profile = select_quantized_cache_profile(
            family=family,
            model=cfg.model,
            quant_dtype=cfg.quant_dtype,
            profiles=profiles,
        )
        profile_ok, profile_reasons = profile.evaluate(prompt_tokens=prompt_tokens)
        if profile.gpu_arch and profile.gpu_arch != hardware.architecture:
            profile_ok = False
            profile_reasons = (*profile_reasons, f"gpu arch {hardware.architecture} != certified {profile.gpu_arch}")
        if not cfg.allow_experimental_quantized and (not profile.production or not profile_ok):
            cache = "paged"
            quantized_allowed = False
            reasons.append(f"quantized cache profile is not production-certified for {family}/{cfg.quant_dtype}; falling back to paged")
            reasons.extend(profile_reasons)
        elif requested == "quantized":
            reasons.append(f"quantized cache allowed by certification profile for {family}/{cfg.quant_dtype}")

    if family.startswith("qwen3"):
        if requested == "auto":
            cache = "paged"
            quantized_allowed = False
            reasons.append("Qwen3 production default is paged direct cache")
        if (
            requested == "quantized"
            and prompt_tokens > cfg.qwen3_quantized_prefill_limit
            and not cfg.allow_experimental_quantized
            and quantized_allowed
        ):
            cache = "paged"
            quantized_allowed = False
            reasons.append("Qwen3 quantized direct cache is not certified for this prefill length; falling back to paged")
        elif requested == "quantized" and quantized_allowed:
            reasons.append("Qwen3 quantized direct cache explicitly allowed by policy")

    cutile_backend = "disabled"
    if hardware.cutile_supported and hardware.architecture in {"hopper", "blackwell"} and cfg.allow_experimental_cutile:
        cutile_backend = "cutile-experimental"
        reasons.append("CuTile experimental backend allowed on Hopper/Blackwell-class runtime")
    elif hardware.architecture in {"ampere", "ada", "pre-ampere", "volta-turing", "cpu", "unknown"}:
        cutile_backend = "triton-or-torch"
        reasons.append(f"CuTile not production-selected on {hardware.architecture}; use Triton/Torch fallback")
    else:
        cutile_backend = "triton-or-torch"
        reasons.append("CuTile runtime is unavailable or not production-certified")

    return HFCachePolicyDecision(
        cache=cache,
        quant_dtype=cfg.quant_dtype,
        direct_cache_allowed=direct_cache_allowed,
        quantized_allowed=quantized_allowed,
        cutile_backend=cutile_backend,
        production=production,
        profile=profile,
        reasons=tuple(reasons),
    )
