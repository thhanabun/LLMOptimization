from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .backend_registry import BackendInfo, default_backend_registry
from .hardware import HARDWARE_PROFILE_SCHEMA_VERSION, HardwareProfile, detect_hardware_profile
from .hf_adapter_matrix import HFAdapterMatrixEntry, production_hf_adapter_matrix
from .hf_cache_certification import HFCacheCertificationReport, certify_hf_cache
from .kernel_certification import KernelCertificationReport, certify_quantized_attention
from .report import make_table

ENV_CERTIFICATION_SCHEMA_VERSION = "llm_memlab.env_certification.v1"


@dataclass(frozen=True)
class EnvironmentCertificationReport:
    hardware: HardwareProfile
    backends: tuple[BackendInfo, ...]
    adapter_matrix: tuple[HFAdapterMatrixEntry, ...]
    hf_cache: HFCacheCertificationReport | None = None
    kernel: KernelCertificationReport | None = None
    cutile: dict[str, Any] | None = None
    fallback_rules: tuple[str, ...] = ()
    schema_version: str = ENV_CERTIFICATION_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        hf_ok = self.hf_cache is None or self.hf_cache.passed
        kernel_ok = self.kernel is None or self.kernel.passed
        return hf_ok and kernel_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "hardware_schema_version": HARDWARE_PROFILE_SCHEMA_VERSION,
            "hardware": self.hardware.to_dict(),
            "backends": [asdict(item) for item in self.backends],
            "adapter_matrix": [asdict(item) for item in self.adapter_matrix],
            "hf_cache": None if self.hf_cache is None else self.hf_cache.to_dict(),
            "kernel": None if self.kernel is None else [record.__dict__ for record in self.kernel.to_records()],
            "cutile": self.cutile,
            "fallback_rules": list(self.fallback_rules),
            "metadata": self.metadata,
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    def write_html(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(env_certification_to_html(self), encoding="utf-8")
        return output

    def to_text(self) -> str:
        return make_table(
            ("Metric", "Value"),
            [
                ("Schema", self.schema_version),
                ("Passed", self.passed),
                ("GPU", self.hardware.gpu_name or self.hardware.device),
                ("Architecture", self.hardware.architecture),
                ("Triton", self.hardware.triton_supported),
                ("CuTile", self.hardware.cutile_supported),
                ("HF cache cases", 0 if self.hf_cache is None else len(self.hf_cache.results)),
                ("Kernel cases", 0 if self.kernel is None else len(self.kernel.results)),
                ("Fallback rules", "; ".join(self.fallback_rules)),
            ],
        )


def certify_environment(
    *,
    model: str | None = None,
    prompts: tuple[str, ...] = ("hello",),
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    quick: bool = True,
    run_hf: bool = True,
    run_kernel: bool = True,
) -> EnvironmentCertificationReport:
    hardware = detect_hardware_profile(None if device == "auto" else device)
    backends = default_backend_registry().list()
    adapter_matrix = production_hf_adapter_matrix()
    hf_report = None
    if model and run_hf:
        hf_report = certify_hf_cache(
            model,
            prompts=list(prompts),
            token_counts=[1],
            caches=["paged"],
            experimental_caches=["quantized"],
            quant_dtypes=["int8", "uint8"],
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            allow_experimental_direct_cache=True,
        )
    kernel_report = certify_quantized_attention(quick=quick, repeats=1, warmup=0) if run_kernel else None
    cutile_info = _cutile_info()
    return EnvironmentCertificationReport(
        hardware=hardware,
        backends=backends,
        adapter_matrix=adapter_matrix,
        hf_cache=hf_report,
        kernel=kernel_report,
        cutile=cutile_info,
        fallback_rules=_fallback_rules(hardware),
        metadata={"model": model, "device": device, "dtype": dtype, "quick": quick},
    )


def env_certification_to_html(report: EnvironmentCertificationReport) -> str:
    backend_rows = "".join(
        f"<tr><td>{_e(item.name)}</td><td>{item.available}</td><td>{item.priority}</td><td>{_e(item.kind)}</td><td>{_e(item.reason)}</td></tr>"
        for item in report.backends
    )
    adapter_rows = "".join(
        f"<tr><td>{_e(item.family)}</td><td>{_e(item.adapter)}</td><td>{item.cache_api}</td><td>{_e(item.production_default_cache)}</td><td>{_e(item.quantized_direct)}</td></tr>"
        for item in report.adapter_matrix
    )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>llm-memlab environment certification</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}}pre{{background:#f6f8fb;padding:12px;border-radius:8px;white-space:pre-wrap}}</style>
</head><body><h1>Environment certification</h1><pre>{_e(report.to_text())}</pre>
<h2>Hardware</h2><pre>{_e(json.dumps(report.hardware.to_dict(), indent=2))}</pre>
<h2>Backends</h2><table><thead><tr><th>Name</th><th>Available</th><th>Priority</th><th>Kind</th><th>Reason</th></tr></thead><tbody>{backend_rows}</tbody></table>
<h2>HF adapter matrix</h2><table><thead><tr><th>Family</th><th>Adapter</th><th>Cache API</th><th>Default</th><th>Quantized</th></tr></thead><tbody>{adapter_rows}</tbody></table>
<h2>Fallback rules</h2><ul>{"".join(f"<li>{_e(rule)}</li>" for rule in report.fallback_rules)}</ul>
</body></html>"""


def _fallback_rules(hardware: HardwareProfile) -> tuple[str, ...]:
    rules = ["production quantized cache requires a matching certification profile"]
    if hardware.architecture not in {"hopper", "blackwell"}:
        rules.append(f"CuTile stays experimental/fallback on {hardware.architecture}")
    if not hardware.triton_supported:
        rules.append("Triton kernels fall back to Torch when Triton is unavailable")
    return tuple(rules)


def _cutile_info() -> dict[str, Any]:
    from .backends.cutile import detect_cutile_runtime

    info = detect_cutile_runtime()
    return asdict(info)


def _e(value: Any) -> str:
    import html

    return html.escape(str(value))
