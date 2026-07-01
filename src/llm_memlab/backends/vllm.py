from __future__ import annotations

import importlib.util
import platform
import time
from dataclasses import dataclass, field
from typing import Any

from ..backend_registry import BackendPlugin
from ..report import make_table

VLLM_BACKEND_SCHEMA_VERSION = "llm_memlab.vllm_backend.v1"


@dataclass(frozen=True)
class VLLMRuntimeInfo:
    installed: bool
    available: bool
    version: str | None = None
    platform: str = field(default_factory=platform.system)
    cuda_available: bool = False
    gpu_name: str | None = None
    compatible: bool = False
    reasons: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = VLLM_BACKEND_SCHEMA_VERSION

    @property
    def fallback_reason(self) -> str:
        return "; ".join(self.reasons) or "vLLM serving backend is available"

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Schema", self.schema_version),
                ("Installed", self.installed),
                ("Available", self.available),
                ("Version", self.version or "n/a"),
                ("Platform", self.platform),
                ("CUDA available", self.cuda_available),
                ("GPU", self.gpu_name or "n/a"),
                ("Compatible", self.compatible),
                ("Reasons", self.fallback_reason),
            ],
        )


@dataclass(frozen=True)
class VLLMGenerateResult:
    text: str
    elapsed_ms: float
    first_token_ms: float | None
    new_tokens: int
    tokens_per_second: float
    peak_cuda_bytes: int | None
    info: VLLMRuntimeInfo
    prefix_cache_enabled: bool | None = None


def detect_vllm_runtime(*, require_cuda: bool = True) -> VLLMRuntimeInfo:
    reasons: list[str] = []
    installed = _module_exists("vllm")
    version = _module_version("vllm") if installed else None
    if not installed:
        reasons.append("vLLM is not installed")
    system = platform.system()
    if system.lower().startswith("windows"):
        reasons.append("native Windows vLLM serving is treated as unsupported; use Linux/WSL/Docker or fallback to HF")
    cuda_available, gpu_name = _cuda_status()
    if require_cuda and not cuda_available:
        reasons.append("CUDA is not available for vLLM serving")
    compatible = installed and (cuda_available or not require_cuda) and not system.lower().startswith("windows")
    if compatible:
        reasons.append(f"vLLM {version or 'unknown'} is installed and CUDA serving prerequisites look usable")
    else:
        reasons.append("policy fallback: use llm-memlab memory-first HF or plain HF generate")
    return VLLMRuntimeInfo(
        installed=installed,
        available=compatible,
        version=version,
        platform=system,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        compatible=compatible,
        reasons=tuple(reasons),
    )


def vllm_backend_plugin() -> BackendPlugin:
    def check() -> tuple[bool, str]:
        info = detect_vllm_runtime()
        return info.available, info.fallback_reason

    return BackendPlugin(
        name="vllm-serving",
        check=check,
        priority=34,
        kind="serving",
        metadata={"schema_version": VLLM_BACKEND_SCHEMA_VERSION, "role": "serving-engine"},
    )


def run_vllm_generate(
    model: str,
    prompt: str,
    *,
    max_new_tokens: int = 8,
    dtype: str = "auto",
    trust_remote_code: bool = True,
    enable_prefix_caching: bool | None = None,
    max_model_len: int | None = None,
) -> VLLMGenerateResult:
    info = detect_vllm_runtime()
    if not info.available:
        raise RuntimeError(info.fallback_reason)
    try:
        import torch
        from vllm import LLM, SamplingParams
    except Exception as exc:  # pragma: no cover - depends on optional vLLM install.
        raise RuntimeError(f"Could not import vLLM runtime: {exc}") from exc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    llm_kwargs: dict[str, Any] = {"model": model, "dtype": dtype, "trust_remote_code": trust_remote_code}
    if enable_prefix_caching is not None:
        llm_kwargs["enable_prefix_caching"] = enable_prefix_caching
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    started = time.perf_counter()
    engine = LLM(**llm_kwargs)
    outputs = engine.generate([prompt], sampling)
    elapsed_ms = (time.perf_counter() - started) * 1000
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    token_ids = getattr(outputs[0].outputs[0], "token_ids", ()) if outputs and outputs[0].outputs else ()
    new_tokens = len(token_ids) if token_ids is not None else max_new_tokens
    seconds = max(elapsed_ms / 1000.0, 1e-9)
    return VLLMGenerateResult(
        text=text,
        elapsed_ms=elapsed_ms,
        first_token_ms=elapsed_ms / max(1, new_tokens),
        new_tokens=new_tokens,
        tokens_per_second=new_tokens / seconds,
        peak_cuda_bytes=peak,
        info=info,
        prefix_cache_enabled=enable_prefix_caching,
    )


def _module_exists(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name, fromlist=["__version__"])
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _cuda_status() -> tuple[bool, str | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return False, None
        return True, torch.cuda.get_device_name(0)
    except Exception:
        return False, None
