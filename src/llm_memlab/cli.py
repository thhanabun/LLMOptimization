from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .estimates import TransformerConfig, estimate_transformer_memory, preset_config
from .ir import GraphSpec, OperationSpec, TensorSpec
from .planner import MemoryPlanner


def _default_local_model_root() -> str:
    configured = os.environ.get("LLM_MEMLAB_MODEL_ROOT")
    if configured:
        return configured
    if os.name == "nt":
        return r"D:\hf_models"
    return "./models"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-memlab", description="Memory-first LLM analysis toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate", help="Estimate LLM memory use.")
    estimate_parser.add_argument("--preset", default="7b-like", help="tiny, 1b-like, 3b-like, 7b-like, 13b-like")
    estimate_parser.add_argument("--layers", type=int)
    estimate_parser.add_argument("--hidden", type=int)
    estimate_parser.add_argument("--intermediate", type=int)
    estimate_parser.add_argument("--heads", type=int)
    estimate_parser.add_argument("--kv-heads", type=int)
    estimate_parser.add_argument("--vocab", type=int)
    estimate_parser.add_argument("--seq", type=int, default=2048)
    estimate_parser.add_argument("--batch", type=int, default=1)
    estimate_parser.add_argument("--dtype", default="bf16")
    estimate_parser.add_argument("--training", choices=["inference", "lora", "full"], default="inference")
    estimate_parser.add_argument("--optimizer", default="adamw")
    estimate_parser.add_argument("--checkpointing", choices=["none", "selective", "full"], default="none")
    estimate_parser.add_argument("--lora-rank", type=int, default=0)
    estimate_parser.add_argument("--no-flash-attention", action="store_true")
    estimate_parser.add_argument("--untied-embeddings", action="store_true")
    estimate_parser.set_defaults(func=_estimate)

    plan_parser = subparsers.add_parser("plan-demo", help="Show tensor-lifetime planning on a toy transformer block.")
    plan_parser.add_argument("--seq", type=int, default=1024)
    plan_parser.add_argument("--hidden", type=int, default=4096)
    plan_parser.add_argument("--intermediate", type=int, default=11008)
    plan_parser.add_argument("--dtype", default="bf16")
    plan_parser.set_defaults(func=_plan_demo)

    trace_parser = subparsers.add_parser("trace-demo", help="Trace a tiny PyTorch model if torch is installed.")
    trace_parser.add_argument("--all-modules", action="store_true", help="Record container modules as well as leaf modules.")
    trace_parser.add_argument("--html-out", help="Write an interactive-ish HTML layer report to this path.")
    trace_parser.add_argument("--timeline-out", help="Write a timeline-style HTML report to this path.")
    trace_parser.add_argument("--interactive-out", help="Write an interactive sortable/filterable HTML report.")
    trace_parser.set_defaults(func=_trace_demo)

    kernel_parser = subparsers.add_parser("kernel-demo", help="Run correctness checks and microbenchmarks for optimized kernels.")
    kernel_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    kernel_parser.add_argument("--compile", action="store_true", help="Use torch.compile where available.")
    kernel_parser.add_argument("--repeats", type=int, default=20)
    kernel_parser.set_defaults(func=_kernel_demo)

    decode_parser = subparsers.add_parser("decode-demo", help="Run a tiny KV-cache decode-loop demo.")
    decode_parser.add_argument("--steps", type=int, default=6)
    decode_parser.set_defaults(func=_decode_demo)

    cache_parser = subparsers.add_parser("cache-demo", help="Compare fp and quantized KV cache memory on random K/V tensors.")
    cache_parser.add_argument("--quantized", action="store_true", help="Use QuantizedStaticKVCache instead of fp cache.")
    cache_parser.add_argument("--dtype", default="int8", help="Quantized cache storage dtype: int8, uint8, fp16, bf16, fp32, fp8_e4m3fn")
    cache_parser.add_argument("--tokens", type=int, default=8)
    cache_parser.set_defaults(func=_cache_demo)

    patch_parser = subparsers.add_parser("patch-demo", help="Patch a tiny Hugging Face-style model and print the patch report.")
    patch_parser.add_argument("--attention", action="store_true", help="Also run the experimental packed-QKV attention patcher.")
    patch_parser.set_defaults(func=_patch_demo)

    bench_parser = subparsers.add_parser("benchmark-demo", help="Run a tiny forward benchmark and patcher comparison.")
    bench_parser.add_argument("--repeats", type=int, default=10)
    bench_parser.set_defaults(func=_benchmark_demo)

    compare_parser = subparsers.add_parser("compare-demo", help="Write an HTML baseline-vs-optimized report for a tiny model.")
    compare_parser.add_argument("--out", default="compare_demo.html")
    compare_parser.add_argument("--repeats", type=int, default=5)
    compare_parser.add_argument("--kv-dtype", default="int8", help="KV dtype used in the quality section.")
    compare_parser.set_defaults(func=_compare_demo)

    policy_parser = subparsers.add_parser("policy-demo", help="Choose a memory-first runtime policy from a VRAM budget.")
    policy_parser.add_argument("--max-vram", default="8GB")
    policy_parser.add_argument("--preset", default="7b-like")
    policy_parser.add_argument("--seq", type=int, default=4096)
    policy_parser.set_defaults(func=_policy_demo)

    debug_hf_parser = subparsers.add_parser("debug-hf", help="Trace and attention-debug a Hugging Face causal LM forward pass.")
    debug_hf_parser.add_argument("--model", required=True)
    debug_hf_parser.add_argument("--prompt", default="Hello")
    debug_hf_parser.add_argument("--device")
    debug_hf_parser.add_argument("--dtype", default="auto")
    debug_hf_parser.add_argument("--html-out")
    debug_hf_parser.add_argument("--timeline-out")
    debug_hf_parser.add_argument("--interactive-out")
    debug_hf_parser.add_argument("--local-files-only", action="store_true")
    debug_hf_parser.set_defaults(func=_debug_hf)

    compare_hf_parser = subparsers.add_parser("compare-hf", help="Write an HTML baseline-vs-optimized HF report.")
    compare_hf_parser.add_argument("--model", required=True)
    compare_hf_parser.add_argument("--prompt", default="Hello")
    compare_hf_parser.add_argument("--out", default="compare_hf.html")
    compare_hf_parser.add_argument("--repeats", type=int, default=2)
    compare_hf_parser.add_argument("--device")
    compare_hf_parser.add_argument("--dtype", default="auto")
    compare_hf_parser.add_argument("--max-vram", default="8GB")
    compare_hf_parser.add_argument("--kv-dtype", default="int8")
    compare_hf_parser.add_argument("--local-files-only", action="store_true")
    compare_hf_parser.set_defaults(func=_compare_hf)

    suite_hf_parser = subparsers.add_parser("suite-hf", help="Run prefill/generate/VRAM benchmark suite on a HF model.")
    suite_hf_parser.add_argument("--model", required=True)
    suite_hf_parser.add_argument("--prompt", default="Hello")
    suite_hf_parser.add_argument("--tokens", type=int, default=16)
    suite_hf_parser.add_argument("--repeats", type=int, default=2)
    suite_hf_parser.add_argument("--device")
    suite_hf_parser.add_argument("--dtype", default="auto")
    suite_hf_parser.add_argument("--local-files-only", action="store_true")
    suite_hf_parser.add_argument("--json-out")
    suite_hf_parser.add_argument("--csv-out")
    suite_hf_parser.set_defaults(func=_suite_hf)

    drift_demo_parser = subparsers.add_parser("drift-demo", help="Compare layer-by-layer drift on two tiny local models.")
    drift_demo_parser.set_defaults(func=_drift_demo)

    drift_hf_parser = subparsers.add_parser("drift-hf", help="Compare layer drift before/after conservative HF patching.")
    drift_hf_parser.add_argument("--model", required=True)
    drift_hf_parser.add_argument("--prompt", default="Hello")
    drift_hf_parser.add_argument("--device")
    drift_hf_parser.add_argument("--dtype", default="auto")
    drift_hf_parser.add_argument("--local-files-only", action="store_true")
    drift_hf_parser.set_defaults(func=_drift_hf)
    scoreboard_hf_parser = subparsers.add_parser(
        "scoreboard-hf", help="Benchmark one or more HF models and write an optimization scoreboard."
    )
    scoreboard_hf_parser.add_argument("--models", nargs="+", required=True)
    scoreboard_hf_parser.add_argument("--prompt", default="Hello")
    scoreboard_hf_parser.add_argument("--out", default="scoreboard_hf.html")
    scoreboard_hf_parser.add_argument("--repeats", type=int, default=1)
    scoreboard_hf_parser.add_argument("--device")
    scoreboard_hf_parser.add_argument("--dtype", default="auto")
    scoreboard_hf_parser.add_argument("--local-files-only", action="store_true")
    scoreboard_hf_parser.add_argument("--json-out")
    scoreboard_hf_parser.add_argument("--csv-out")
    scoreboard_hf_parser.set_defaults(func=_scoreboard_hf)

    run_hf_parser = subparsers.add_parser("run-hf", help="Run HF generation with a memory-first policy report.")
    run_hf_parser.add_argument("--model", required=True)
    run_hf_parser.add_argument("--prompt", default="Hello")
    run_hf_parser.add_argument("--tokens", type=int, default=16)
    run_hf_parser.add_argument("--max-vram", default="8GB")
    run_hf_parser.add_argument("--device")
    run_hf_parser.add_argument("--dtype", default="auto")
    run_hf_parser.add_argument("--local-files-only", action="store_true")
    run_hf_parser.set_defaults(func=_run_hf)
    inspect_demo_parser = subparsers.add_parser("inspect-demo", help="Inspect a tiny local HF-style model.")
    inspect_demo_parser.set_defaults(func=_inspect_demo)

    inspect_hf_parser = subparsers.add_parser("inspect-hf", help="Inspect a Hugging Face causal LM architecture.")
    inspect_hf_parser.add_argument("--model", required=True, help="Model name or local path.")
    inspect_hf_parser.add_argument("--device")
    inspect_hf_parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, or fp32")
    inspect_hf_parser.add_argument("--seq", type=int, help="Override sequence length for KV cache estimate.")
    inspect_hf_parser.add_argument("--local-files-only", action="store_true")
    inspect_hf_parser.set_defaults(func=_inspect_hf)

    benchmark_hf_parser = subparsers.add_parser("benchmark-hf", help="Benchmark Hugging Face generate before/after conservative patching.")
    benchmark_hf_parser.add_argument("--model", required=True, help="Model name or local path.")
    benchmark_hf_parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    benchmark_hf_parser.add_argument("--tokens", type=int, default=16)
    benchmark_hf_parser.add_argument("--repeats", type=int, default=3)
    benchmark_hf_parser.add_argument("--device")
    benchmark_hf_parser.add_argument("--dtype", default="auto")
    benchmark_hf_parser.add_argument("--local-files-only", action="store_true")
    benchmark_hf_parser.set_defaults(func=_benchmark_hf)

    memory_first_hf_parser = subparsers.add_parser(
        "memory-first-hf-bench", help="Benchmark baseline HF generate vs llm-memlab memory-first adapter."
    )
    memory_first_hf_parser.add_argument("--model", required=True, help="Model name or local path.")
    memory_first_hf_parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    memory_first_hf_parser.add_argument("--tokens", type=int, default=8)
    memory_first_hf_parser.add_argument(
        "--adapter-tokens", type=int, help="Override memory-first adapter token count; defaults to --tokens."
    )
    memory_first_hf_parser.add_argument("--device", default="auto")
    memory_first_hf_parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, or fp32")
    memory_first_hf_parser.add_argument("--cache", choices=["quantized", "paged"], default="quantized")
    memory_first_hf_parser.add_argument("--quant-dtype", default="int8")
    memory_first_hf_parser.add_argument(
        "--allow-experimental-direct-cache",
        action="store_true",
        help="Allow non-certified direct Transformers Cache injection for model families that default to safe fallback.",
    )
    memory_first_hf_parser.add_argument("--local-files-only", action="store_true")
    memory_first_hf_parser.add_argument("--json-out")
    memory_first_hf_parser.add_argument("--csv-out")
    memory_first_hf_parser.add_argument("--min-token-agreement", type=float, default=1.0)
    memory_first_hf_parser.add_argument("--max-slowdown-pct", type=float)
    memory_first_hf_parser.add_argument("--fail-on-regression", action="store_true")
    memory_first_hf_parser.set_defaults(func=_memory_first_hf_bench)

    serving_parser = subparsers.add_parser(
        "serving-bench", help="Compare HF generate, llm-memlab memory-first HF, and optional vLLM serving."
    )
    serving_parser.add_argument("--model", required=True, help="Model name or local path.")
    serving_parser.add_argument("--prompt", default="Hello", help="Prompt text.")
    serving_parser.add_argument("--tokens", type=int, default=8)
    serving_parser.add_argument("--adapter-tokens", type=int, help="Override memory-first adapter token count.")
    serving_parser.add_argument("--device", default="auto")
    serving_parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, or fp32")
    serving_parser.add_argument("--cache", choices=["quantized", "paged"], default="paged")
    serving_parser.add_argument("--quant-dtype", default="int8")
    serving_parser.add_argument("--no-vllm", action="store_true", help="Skip the vLLM serving path.")
    serving_parser.add_argument("--allow-experimental-direct-cache", action="store_true")
    serving_parser.add_argument("--local-files-only", action="store_true")
    serving_parser.add_argument("--json-out")
    serving_parser.add_argument("--csv-out")
    serving_parser.add_argument("--html-out")
    serving_parser.add_argument("--fail-on-regression", action="store_true")
    serving_parser.set_defaults(func=_serving_bench)

    hf_cache_cert_parser = subparsers.add_parser(
        "hf-cache-certify", help="Certify direct HF Cache adapter correctness with generated-token and prefill-logit gates."
    )
    hf_cache_cert_parser.add_argument("--model", required=True, help="Model name or local path.")
    hf_cache_cert_parser.add_argument("--prompts", default="hello", help="Pipe-separated prompts, e.g. 'hello|Explain KV cache'.")
    hf_cache_cert_parser.add_argument("--tokens", default="1", help="Comma-separated max_new_tokens values.")
    hf_cache_cert_parser.add_argument("--caches", default="paged", help="Comma-separated cache modes: paged,quantized.")
    hf_cache_cert_parser.add_argument(
        "--experimental-caches",
        default="",
        help="Comma-separated cache modes to report as experimental without failing the production gate.",
    )
    hf_cache_cert_parser.add_argument("--quant-dtypes", default="int8", help="Comma-separated quant dtypes for quantized cache cases.")
    hf_cache_cert_parser.add_argument("--device", default="auto")
    hf_cache_cert_parser.add_argument("--dtype", default="auto", help="auto, fp16, bf16, or fp32")
    hf_cache_cert_parser.add_argument("--local-files-only", action="store_true")
    hf_cache_cert_parser.add_argument("--no-experimental-direct-cache", action="store_true")
    hf_cache_cert_parser.add_argument("--max-logit-mean-abs", type=float, default=0.02)
    hf_cache_cert_parser.add_argument("--min-logit-top1", type=float, default=0.98)
    hf_cache_cert_parser.add_argument("--json-out")
    hf_cache_cert_parser.add_argument("--csv-out")
    hf_cache_cert_parser.add_argument("--html-out")
    hf_cache_cert_parser.add_argument("--fail-on-regression", action="store_true")
    hf_cache_cert_parser.set_defaults(func=_hf_cache_certify)

    certify_env_parser = subparsers.add_parser(
        "certify-env", help="Certify local hardware, backends, HF cache policy, and kernel readiness in one run."
    )
    certify_env_parser.add_argument("--model", help="Optional HF model path/name for cache certification.")
    certify_env_parser.add_argument("--prompt", default="hello")
    certify_env_parser.add_argument("--device", default="auto")
    certify_env_parser.add_argument("--dtype", default="auto")
    certify_env_parser.add_argument("--local-files-only", action="store_true")
    certify_env_parser.add_argument("--skip-hf", action="store_true")
    certify_env_parser.add_argument("--skip-kernel", action="store_true")
    certify_env_parser.add_argument("--json-out")
    certify_env_parser.add_argument("--html-out")
    certify_env_parser.add_argument("--fail-on-regression", action="store_true")
    certify_env_parser.set_defaults(func=_certify_env)

    matrix_parser = subparsers.add_parser("certify-model-matrix", help="Certify a real-model HF cache matrix and emit profiles.json.")
    matrix_parser.add_argument("--models", nargs="*", help="Entries like family=path_or_model. Defaults to known small model families.")
    matrix_parser.add_argument("--local-root", help="Optional directory containing locally cached model folders.")
    matrix_parser.add_argument("--allow-remote", action="store_true", help="Allow Transformers to fetch remote models.")
    matrix_parser.add_argument("--prompt", action="append", help="Prompt to certify. Can be repeated.")
    matrix_parser.add_argument("--device", default="auto")
    matrix_parser.add_argument("--dtype", default="auto")
    matrix_parser.add_argument("--json-out")
    matrix_parser.add_argument("--profiles-out", default="profiles.json")
    matrix_parser.add_argument("--require-real-models", action="store_true", help="Fail if local production targets are skipped.")
    matrix_parser.add_argument("--min-certified-models", type=int, default=0, help="Minimum real model certifications required to pass.")
    matrix_parser.add_argument("--strict", action="store_true", help="Require every production target to be certified by a real model.")
    matrix_parser.add_argument("--fail-on-regression", action="store_true")
    matrix_parser.set_defaults(func=_certify_model_matrix)

    local_harness_parser = subparsers.add_parser(
        "local-model-harness", help="Scan local model fixtures and optionally certify available cached models."
    )
    local_harness_parser.add_argument(
        "--root",
        default=_default_local_model_root(),
        help="Directory containing local cached model folders. Defaults to LLM_MEMLAB_MODEL_ROOT or a platform-local model root.",
    )
    local_harness_parser.add_argument("--json-out", default="local_model_fixtures.json")
    local_harness_parser.add_argument("--certify", action="store_true", help="Run certify-model-matrix against available fixtures.")
    local_harness_parser.add_argument("--matrix-out", default="local_model_matrix.json")
    local_harness_parser.add_argument("--profiles-out", default="profiles.json")
    local_harness_parser.add_argument("--prompt", action="append")
    local_harness_parser.add_argument("--device", default="auto")
    local_harness_parser.add_argument("--dtype", default="auto")
    local_harness_parser.add_argument("--require-real-models", action="store_true")
    local_harness_parser.add_argument("--min-certified-models", type=int, default=0)
    local_harness_parser.add_argument("--strict", action="store_true")
    local_harness_parser.set_defaults(func=_local_model_harness)

    profile_parser = subparsers.add_parser("profile", help="Inspect and manage quantized cache certification profiles.")
    profile_subparsers = profile_parser.add_subparsers(dest="profile_command", required=True)
    profile_export = profile_subparsers.add_parser("export", help="Export built-in/default profiles to JSON.")
    profile_export.add_argument("--out", default="profiles.json")
    profile_export.set_defaults(func=_profile_export)
    profile_merge = profile_subparsers.add_parser("merge", help="Merge profile JSON/YAML files, with earlier files taking priority.")
    profile_merge.add_argument("--inputs", nargs="+", required=True)
    profile_merge.add_argument("--out", default="profiles.merged.json")
    profile_merge.set_defaults(func=_profile_merge)
    profile_explain = profile_subparsers.add_parser("explain", help="Explain the cache policy decision for a family/model/profile.")
    profile_explain.add_argument("--family", required=True)
    profile_explain.add_argument("--model")
    profile_explain.add_argument("--quant-dtype", default="int8")
    profile_explain.add_argument("--prompt-tokens", type=int, default=1)
    profile_explain.add_argument("--profile", action="append", default=[])
    profile_explain.add_argument("--allow-experimental-quantized", action="store_true")
    profile_explain.set_defaults(func=_profile_explain)

    kv_quality_parser = subparsers.add_parser("kv-quality-demo", help="Measure KV quantization error on random K/V-like tensors.")
    kv_quality_parser.add_argument("--tokens", type=int, default=16)
    kv_quality_parser.add_argument("--heads", type=int, default=8)
    kv_quality_parser.add_argument("--head-dim", type=int, default=64)
    kv_quality_parser.add_argument("--dtype", default="int8", help="int8, uint8, fp16, bf16, fp32, fp8_e4m3fn")
    kv_quality_parser.add_argument("--attention", action="store_true", help="Compare SDPA output with quantized/dequantized K/V.")
    kv_quality_parser.set_defaults(func=_kv_quality_demo)
    quality_parser = subparsers.add_parser("quality-demo", help="Compare logits and token outputs with quality metrics.")
    quality_parser.add_argument("--top-k", type=int, default=5)
    quality_parser.set_defaults(func=_quality_demo)

    backend_parser = subparsers.add_parser("backend-demo", help="Show available runtime backends and priorities.")
    backend_parser.set_defaults(func=_backend_demo)

    benchmark_compare_parser = subparsers.add_parser("benchmark-compare", help="Compare benchmark JSON/CSV files and fail on regressions.")
    benchmark_compare_parser.add_argument(
        "--baseline", nargs="+", required=True, help="Baseline JSON or CSV benchmark records. Multiple files form a history median."
    )
    benchmark_compare_parser.add_argument(
        "--candidate", nargs="+", required=True, help="Candidate JSON/CSV benchmark records. Multiple files are compared independently."
    )
    benchmark_compare_parser.add_argument(
        "--max-slowdown-pct", type=float, default=10.0, help="Fail if candidate mean latency is slower by more than this percentage."
    )
    benchmark_compare_parser.add_argument(
        "--fail-on-regression", action="store_true", help="Return exit code 1 if any comparison exceeds the threshold."
    )
    benchmark_compare_parser.set_defaults(func=_benchmark_compare)

    dashboard_parser = subparsers.add_parser("benchmark-dashboard", help="Build an HTML dashboard from benchmark JSON/CSV history.")
    dashboard_parser.add_argument("--inputs", nargs="+", required=True)
    dashboard_parser.add_argument("--out", default="benchmark_dashboard.html")
    dashboard_parser.add_argument("--title", default="llm-memlab benchmark dashboard")
    dashboard_parser.set_defaults(func=_benchmark_dashboard)

    fused_bench_parser = subparsers.add_parser("fused-decode-bench", help="Run a CUDA fused decode benchmark matrix with quality metrics.")
    fused_bench_parser.add_argument("--q-heads", default="8", help="Comma-separated Q head counts.")
    fused_bench_parser.add_argument("--kv-heads", default="8", help="Comma-separated KV head counts.")
    fused_bench_parser.add_argument("--tokens", default="128", help="Comma-separated sequence lengths.")
    fused_bench_parser.add_argument("--head-dim", type=int, default=64)
    fused_bench_parser.add_argument("--dtype", default="fp16")
    fused_bench_parser.add_argument("--quant-dtype", choices=["int8", "uint8"], default="int8")
    fused_bench_parser.add_argument("--page-size", default="16", help="Comma-separated page sizes recorded in metadata.")
    fused_bench_parser.add_argument("--warmup", type=int, default=3)
    fused_bench_parser.add_argument("--repeats", type=int, default=10)
    fused_bench_parser.add_argument("--seed", type=int, default=0)
    fused_bench_parser.add_argument("--max-mean-abs", type=float, default=0.03)
    fused_bench_parser.add_argument("--json-out")
    fused_bench_parser.add_argument("--csv-out")
    fused_bench_parser.add_argument("--html-out")
    fused_bench_parser.add_argument("--min-speedup", type=float, default=0.0)
    fused_bench_parser.add_argument("--fail-on-regression", action="store_true")
    fused_bench_parser.set_defaults(func=_fused_decode_bench)

    cutile_bench_parser = subparsers.add_parser("cutile-bench", help="Run torch/Triton/CuTile decode backend matrix.")
    cutile_bench_parser.add_argument("--q-heads", type=int, default=8)
    cutile_bench_parser.add_argument("--kv-heads", type=int, default=2)
    cutile_bench_parser.add_argument("--tokens", type=int, default=128)
    cutile_bench_parser.add_argument("--head-dim", type=int, default=64)
    cutile_bench_parser.add_argument("--page-size", type=int, default=16)
    cutile_bench_parser.add_argument("--dtype", default="fp16")
    cutile_bench_parser.add_argument("--quant-dtype", default="int8")
    cutile_bench_parser.add_argument("--warmup", type=int, default=3)
    cutile_bench_parser.add_argument("--repeats", type=int, default=10)
    cutile_bench_parser.add_argument("--seed", type=int, default=0)
    cutile_bench_parser.add_argument("--json-out")
    cutile_bench_parser.add_argument("--csv-out")
    cutile_bench_parser.add_argument("--fail-on-regression", action="store_true")
    cutile_bench_parser.set_defaults(func=_cutile_bench)

    cutile_cert_parser = subparsers.add_parser("cutile-certify", help="Certify CuTile paged decode correctness and policy readiness.")
    cutile_cert_parser.add_argument("--q-heads", type=int, default=8)
    cutile_cert_parser.add_argument("--kv-heads", type=int, default=2)
    cutile_cert_parser.add_argument("--tokens", type=int, default=128)
    cutile_cert_parser.add_argument("--head-dim", type=int, default=64)
    cutile_cert_parser.add_argument("--page-size", type=int, default=16)
    cutile_cert_parser.add_argument("--dtype", default="fp16")
    cutile_cert_parser.add_argument("--seed", type=int, default=0)
    cutile_cert_parser.add_argument("--fail-on-regression", action="store_true")
    cutile_cert_parser.set_defaults(func=_cutile_certify)

    certify_parser = subparsers.add_parser(
        "kernel-certify", help="Run the production kernel certification suite and write benchmark DB records."
    )
    certify_parser.add_argument("--quick", action="store_true", help="Run a small smoke matrix instead of the full shape matrix.")
    certify_parser.add_argument("--repeats", type=int, default=5)
    certify_parser.add_argument("--warmup", type=int, default=2)
    certify_parser.add_argument("--seed", type=int, default=0)
    certify_parser.add_argument("--max-mean-abs", type=float, default=0.03)
    certify_parser.add_argument("--min-top1", type=float, default=0.90)
    certify_parser.add_argument("--json-out")
    certify_parser.add_argument("--csv-out")
    certify_parser.add_argument("--fail-on-regression", action="store_true", help="Return exit code 1 if any certified case fails.")
    certify_parser.set_defaults(func=_kernel_certify)

    promote_parser = subparsers.add_parser("kernel-promote", help="Run kernel certification and explain whether a backend can be promoted.")
    promote_parser.add_argument("--backend", choices=["triton", "cutile"], default="triton")
    promote_parser.add_argument("--quick", action="store_true")
    promote_parser.add_argument("--repeats", type=int, default=1)
    promote_parser.add_argument("--warmup", type=int, default=0)
    promote_parser.add_argument("--require-long-context", action="store_true")
    promote_parser.add_argument(
        "--allow-missing-long-context",
        action="store_true",
        help="Do not require seq >= 4096 coverage for experimental local smoke runs.",
    )
    promote_parser.add_argument("--fail-on-regression", action="store_true")
    promote_parser.set_defaults(func=_kernel_promote)

    args = parser.parse_args(argv)
    return args.func(args)


def _quality_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run quality-demo: pip install torch")
        return 2

    from .quality_metrics import compare_logits, compare_token_sequences

    baseline = torch.randn(2, 4, 16)
    candidate = baseline + torch.randn_like(baseline) * 0.001
    print(compare_logits(baseline, candidate, top_k=args.top_k).to_text())
    print("")
    print(compare_token_sequences(torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 5, 4])).to_text())
    return 0


def _backend_demo(args: argparse.Namespace) -> int:
    from .backend_registry import default_backend_registry

    print(default_backend_registry().to_text())
    return 0


def _certify_model_matrix(args: argparse.Namespace) -> int:
    from .certification_matrix import ModelCertificationTarget, certify_model_matrix, default_model_certification_targets

    if args.models:
        targets = []
        for item in args.models:
            if "=" not in item:
                print(f"Invalid --models entry {item!r}; expected family=path_or_model")
                return 2
            family, model = item.split("=", 1)
            targets.append(ModelCertificationTarget(name=family, family=family, model=model, local_files_only=not args.allow_remote))
    else:
        targets = list(default_model_certification_targets(local_root=args.local_root))
    report = certify_model_matrix(
        targets,
        prompts=tuple(args.prompt or ["hello", "Explain KV cache briefly."]),
        device=args.device,
        dtype=args.dtype,
        allow_remote=args.allow_remote,
    )
    print(report.to_text())
    if args.json_out:
        print(f"Certification matrix JSON written to {report.write_json(args.json_out)}")
    if args.profiles_out:
        print(f"Profiles JSON written to {report.write_profiles(args.profiles_out)}")
    gate = report.evaluate_gate(
        require_real_models=args.require_real_models,
        min_certified_models=args.min_certified_models,
        strict=args.strict,
    )
    if args.require_real_models or args.min_certified_models > 0 or args.strict:
        print("")
        print(gate.to_text())
    if (args.fail_on_regression and not report.passed) or not gate.passed:
        return 1
    return 0


def _local_model_harness(args: argparse.Namespace) -> int:
    from .certification_matrix import certify_model_matrix
    from .local_model_harness import scan_local_model_fixtures

    report = scan_local_model_fixtures(args.root)
    print(report.to_text())
    if args.json_out:
        print(f"Local model fixture JSON written to {report.write_json(args.json_out)}")
    if not args.certify:
        if (args.require_real_models or args.strict or args.min_certified_models > 0) and report.available_count < max(
            1 if args.require_real_models or args.strict else 0, args.min_certified_models
        ):
            print("Local model harness gate failed: not enough cached model fixtures are available.")
            return 1
        return 0

    targets = report.targets(available_only=True)
    if not targets and (args.require_real_models or args.strict or args.min_certified_models > 0):
        print("Local model harness gate failed: no available local model fixtures to certify.")
        return 1
    matrix = certify_model_matrix(
        targets,
        prompts=tuple(args.prompt or ["hello", "Explain KV cache briefly."]),
        device=args.device,
        dtype=args.dtype,
        allow_remote=False,
    )
    print("")
    print(matrix.to_text())
    if args.matrix_out:
        print(f"Certification matrix JSON written to {matrix.write_json(args.matrix_out)}")
    if args.profiles_out:
        print(f"Profiles JSON written to {matrix.write_profiles(args.profiles_out)}")
    gate = matrix.evaluate_gate(
        require_real_models=args.require_real_models,
        min_certified_models=args.min_certified_models,
        strict=args.strict,
    )
    print("")
    print(gate.to_text())
    return 0 if gate.passed else 1


def _profile_export(args: argparse.Namespace) -> int:
    from .hf_cache_profiles import DEFAULT_QUANTIZED_CACHE_PROFILES, write_quantized_cache_profiles

    print(f"Profiles JSON written to {write_quantized_cache_profiles(DEFAULT_QUANTIZED_CACHE_PROFILES, args.out)}")
    return 0


def _profile_merge(args: argparse.Namespace) -> int:
    from .hf_cache_profiles import load_quantized_cache_profiles, write_quantized_cache_profiles

    registry = load_quantized_cache_profiles(args.inputs)
    print(f"Profiles JSON written to {write_quantized_cache_profiles(registry.profiles, args.out)}")
    return 0


def _profile_explain(args: argparse.Namespace) -> int:
    from .hf_cache_policy import HFCachePolicy, select_hf_cache_policy

    policy = HFCachePolicy(
        requested_cache="quantized",
        model=args.model,
        quant_dtype=args.quant_dtype,
        quantized_profile_paths=tuple(args.profile),
        allow_experimental_quantized=args.allow_experimental_quantized,
    )
    decision = select_hf_cache_policy(family=args.family, prompt_tokens=args.prompt_tokens, policy=policy)
    print(decision.to_text())
    if decision.profile is not None:
        print("")
        print(decision.profile.to_text())
    return 0


def _benchmark_compare(args: argparse.Namespace) -> int:
    from .benchmark_store import assert_no_regressions, benchmark_history, compare_record_sets, read_benchmark_files

    baseline = benchmark_history(args.baseline).baseline_records()
    all_comparisons = []
    for candidate_path in args.candidate:
        candidate = read_benchmark_files([candidate_path])
        comparisons = compare_record_sets(baseline, candidate, max_slowdown_pct=args.max_slowdown_pct)
        all_comparisons.extend(comparisons)
        print(f"candidate={candidate_path}")
        if not comparisons:
            print("No matching (name, kind) benchmark records found.")
        for comparison in comparisons:
            print(comparison.to_text())
        print("")
    if args.fail_on_regression:
        try:
            assert_no_regressions(all_comparisons)
        except AssertionError as exc:
            print(str(exc))
            return 1
    return 0


def _benchmark_dashboard(args: argparse.Namespace) -> int:
    from .benchmark_dashboard import write_benchmark_dashboard_html

    path = write_benchmark_dashboard_html(args.inputs, args.out, title=args.title)
    print(f"Benchmark dashboard HTML written to {path}")
    return 0


def _read_benchmark_records(path: str, read_json, read_csv):
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return read_json(path)
    if suffix == ".csv":
        return read_csv(path)
    raise ValueError("benchmark files must end with .json or .csv")


def _fused_decode_bench(args: argparse.Namespace) -> int:
    from .benchmark_store import (
        BENCHMARK_SCHEMA_VERSION,
        BenchmarkGateConfig,
        BenchmarkRecord,
        collect_run_metadata,
        write_benchmark_csv,
        write_benchmark_json,
    )
    from .decode_benchmarks import benchmark_fused_decode_attention
    from .report import make_table

    records = []
    rows = []
    failures = []
    for q_heads in _csv_ints(args.q_heads):
        for kv_heads in _csv_ints(args.kv_heads):
            if q_heads % kv_heads != 0:
                continue
            for tokens in _csv_ints(args.tokens):
                for page_size in _csv_ints(args.page_size):
                    result = benchmark_fused_decode_attention(
                        q_heads=q_heads,
                        kv_heads=kv_heads,
                        tokens=tokens,
                        head_dim=args.head_dim,
                        dtype=args.dtype,
                        quant_dtype=args.quant_dtype,
                        repeats=args.repeats,
                        warmup=args.warmup,
                        seed=args.seed,
                    )
                    name = f"fused_decode/q{q_heads}/kv{kv_heads}/t{tokens}/d{args.head_dim}/{args.quant_dtype}/p{page_size}"
                    metadata = collect_run_metadata(
                        dtype=args.dtype,
                        sequence_length=tokens,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        seed=args.seed,
                        backend="triton-experimental",
                    )
                    quality_passed = bool(result.quality.passed and result.quality.mean_abs_error <= args.max_mean_abs)
                    fused_extra = {
                        "schema_version": BENCHMARK_SCHEMA_VERSION,
                        "speedup": result.speedup,
                        "quality_passed": quality_passed,
                        "mean_abs": result.quality.mean_abs_error,
                        "page_size": page_size,
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "seed": args.seed,
                        "backend": "triton-experimental",
                    }
                    ref_extra = {"page_size": page_size}
                    records.append(
                        BenchmarkRecord(
                            name=name + ":fused",
                            kind="decode",
                            mean_ms=result.fused.mean_ms,
                            min_ms=result.fused.min_ms,
                            max_ms=result.fused.max_ms,
                            peak_cuda_bytes=result.fused.peak_cuda_bytes,
                            extra=fused_extra,
                            metadata=metadata.__dict__,
                        )
                    )
                    records.append(
                        BenchmarkRecord(
                            name=name + ":dequant_sdpa",
                            kind="decode",
                            mean_ms=result.dequant_sdpa.mean_ms,
                            min_ms=result.dequant_sdpa.min_ms,
                            max_ms=result.dequant_sdpa.max_ms,
                            peak_cuda_bytes=result.dequant_sdpa.peak_cuda_bytes,
                            extra=ref_extra,
                            metadata=metadata.__dict__,
                        )
                    )
                    gate = BenchmarkGateConfig(min_speedup=args.min_speedup, max_quality_mean_abs=args.max_mean_abs)
                    passed = quality_passed and result.speedup >= (gate.min_speedup or 0.0)
                    if not passed:
                        failures.append(name)
                    rows.append(
                        (
                            name,
                            f"{result.fused.mean_ms:.3f}",
                            f"{result.dequant_sdpa.mean_ms:.3f}",
                            f"{result.speedup:.3f}x",
                            quality_passed,
                            f"{result.quality.mean_abs_error:.6f}",
                            "PASS" if passed else "FAIL",
                        )
                    )
    print(make_table(("Case", "Fused ms", "Ref ms", "Speedup", "Quality", "Mean abs", "Status"), rows))
    if args.json_out:
        print(f"Benchmark JSON written to {write_benchmark_json(records, args.json_out)}")
    if args.csv_out:
        print(f"Benchmark CSV written to {write_benchmark_csv(records, args.csv_out)}")
    if args.html_out:
        path = _write_fused_bench_html(rows, args.html_out)
        print(f"Benchmark HTML written to {path}")
    if failures and args.fail_on_regression:
        print("Fused decode regression threshold failed: " + ", ".join(failures))
        return 1
    return 0


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _write_fused_bench_html(rows, path: str):
    output = Path(path)
    html_rows = "".join(
        f"<tr><td>{case}</td><td>{fused}</td><td>{ref}</td><td>{speed}</td><td>{quality}</td><td>{mean_abs}</td><td>{status}</td></tr>"
        for case, fused, ref, speed, quality, mean_abs, status in rows
    )
    output.write_text(
        f"""<!doctype html><html><head><meta charset='utf-8'><title>llm-memlab fused decode benchmark</title><style>body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left}}</style></head><body><h1>Fused decode benchmark</h1><table><thead><tr><th>Case</th><th>Fused ms</th><th>Ref ms</th><th>Speedup</th><th>Quality</th><th>Mean abs</th><th>Status</th></tr></thead><tbody>{html_rows}</tbody></table></body></html>""",
        encoding="utf-8",
    )
    return output


def _kernel_certify(args: argparse.Namespace) -> int:
    from .kernel_certification import certify_quantized_attention

    report = certify_quantized_attention(
        quick=args.quick,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        max_mean_abs=args.max_mean_abs,
        min_top1=args.min_top1,
    )
    print(report.to_text())
    if args.json_out:
        print(f"Certification JSON written to {report.write_json(args.json_out)}")
    if args.csv_out:
        print(f"Certification CSV written to {report.write_csv(args.csv_out)}")
    if args.fail_on_regression and not report.passed:
        return 1
    return 0


def _kernel_promote(args: argparse.Namespace) -> int:
    from .kernel_certification import certify_quantized_attention
    from .kernel_promotion import decide_kernel_promotion

    report = certify_quantized_attention(quick=args.quick, repeats=args.repeats, warmup=args.warmup)
    decision = decide_kernel_promotion(
        report,
        backend=args.backend,
        require_long_context=args.require_long_context or not args.allow_missing_long_context,
    )
    print(report.to_text())
    print("")
    print(decision.to_text())
    if args.fail_on_regression and not decision.promoted:
        return 1
    return 0


def _cutile_bench(args: argparse.Namespace) -> int:
    try:
        from .benchmark_store import write_benchmark_csv, write_benchmark_json
        from .decode_benchmarks import benchmark_decode_backend_matrix
    except RuntimeError as exc:
        print(str(exc))
        return 2
    try:
        result = benchmark_decode_backend_matrix(
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            tokens=args.tokens,
            head_dim=args.head_dim,
            page_size=args.page_size,
            dtype=args.dtype,
            quant_dtype=args.quant_dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Could not run CuTile benchmark matrix: {exc}")
        return 2
    print(result.to_text())
    if args.json_out:
        print(f"Benchmark JSON written to {write_benchmark_json(result.records, args.json_out)}")
    if args.csv_out:
        print(f"Benchmark CSV written to {write_benchmark_csv(result.records, args.csv_out)}")
    if args.fail_on_regression and not result.passed:
        return 1
    return 0


def _cutile_certify(args: argparse.Namespace) -> int:
    try:
        import torch

        from .backends.cutile import certify_cutile_decode_attention
    except Exception as exc:
        print(f"Could not import CuTile certification dependencies: {exc}")
        return 2
    if not torch.cuda.is_available():
        print("CuTile certification requires CUDA")
        return 2
    if args.q_heads % args.kv_heads != 0:
        print("q_heads must be divisible by kv_heads")
        return 2
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch_dtype = (
        torch.float16 if args.dtype in {"fp16", "float16"} else torch.bfloat16 if args.dtype in {"bf16", "bfloat16"} else torch.float32
    )
    q = torch.randn(1, args.q_heads, 1, args.head_dim, device="cuda", dtype=torch_dtype)
    num_pages = (args.tokens + args.page_size - 1) // args.page_size
    k_pages = torch.randn(1, args.kv_heads, num_pages, args.page_size, args.head_dim, device="cuda", dtype=torch_dtype)
    v_pages = torch.randn_like(k_pages)
    page_table = torch.arange(num_pages, device="cuda", dtype=torch.long).view(1, num_pages)
    lengths = torch.tensor([args.tokens], device="cuda", dtype=torch.long)
    result = certify_cutile_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=args.page_size)
    print(result.info.to_text())
    print("")
    print(result.to_text())
    if args.fail_on_regression and not result.passed:
        return 1
    return 0


def _estimate(args: argparse.Namespace) -> int:
    if all(value is not None for value in (args.layers, args.hidden, args.intermediate, args.heads, args.vocab)):
        cfg = TransformerConfig(
            num_layers=args.layers,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            num_attention_heads=args.heads,
            vocab_size=args.vocab,
            sequence_length=args.seq,
            batch_size=args.batch,
            dtype=args.dtype,
        )
    else:
        cfg = preset_config(args.preset, sequence_length=args.seq, batch_size=args.batch, dtype=args.dtype)
    cfg = TransformerConfig(
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        dtype=cfg.dtype,
        training=args.training,
        optimizer=args.optimizer,
        activation_checkpointing=args.checkpointing,
        use_flash_attention=not args.no_flash_attention,
        tie_embeddings=not args.untied_embeddings,
        num_key_value_heads=args.kv_heads or cfg.num_key_value_heads,
        lora_rank=args.lora_rank,
    )
    print(estimate_transformer_memory(cfg).to_text())
    return 0


def _plan_demo(args: argparse.Namespace) -> int:
    graph = _toy_block_graph(args.seq, args.hidden, args.intermediate, args.dtype)
    plan = MemoryPlanner(graph.tensor_lifetimes()).plan()
    print(plan.to_text())
    return 0


def _trace_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run trace-demo: pip install torch")
        return 2

    from .torch_debugger import TorchTrace

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 16),
    )
    x = torch.randn(8, 16)
    with TorchTrace(model, record_leaf_only=not args.all_modules) as trace:
        _ = model(x)
    print(trace.to_text(show_shapes=True, show_stats=True))
    if args.html_out:
        from .html_report import write_trace_html

        path = write_trace_html(trace, args.html_out, title="llm-memlab trace demo")
        print(f"HTML report written to {path}")
    if args.timeline_out:
        from .html_report import write_timeline_html

        path = write_timeline_html(trace, args.timeline_out, title="llm-memlab trace timeline")
        print(f"Timeline report written to {path}")
    if args.interactive_out:
        from .html_report import write_interactive_html

        path = write_interactive_html(trace, args.interactive_out, title="llm-memlab interactive trace")
        print(f"Interactive report written to {path}")
    return 0


def _kernel_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run kernel-demo: pip install torch")
        return 2

    from .kernels import KernelConfig, chunked_cross_entropy, kernel, linear_cross_entropy
    from .report import make_table

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available.")
        return 2
    dtype = torch.float16 if device == "cuda" else torch.float32
    cfg = KernelConfig(compile=args.compile)

    rms = kernel("rms_norm", cfg)
    rms_manual = kernel("rms_norm_manual_backward", cfg)
    sdpa = kernel("scaled_dot_product_attention", cfg)
    swiglu_kernel = kernel("swiglu", cfg)
    qkv_attn = kernel("qkv_rope_attention", cfg)

    x = torch.randn(2, 128, 512, device=device, dtype=dtype)
    weight = torch.randn(512, device=device, dtype=dtype)
    gate = torch.randn(1536, 512, device=device, dtype=dtype)
    up = torch.randn(1536, 512, device=device, dtype=dtype)
    down = torch.randn(512, 1536, device=device, dtype=dtype)
    q = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    k = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    v = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    logits = torch.randn(2, 128, 2048, device=device, dtype=dtype)
    targets = torch.randint(0, 2048, (2, 128), device=device)
    lm_head = torch.randn(2048, 512, device=device, dtype=dtype)
    qkv_weight = torch.randn(1536, 512, device=device, dtype=dtype)
    out_weight = torch.randn(512, 512, device=device, dtype=dtype)
    cos = torch.randn(128, 64, device=device, dtype=dtype)
    sin = torch.randn(128, 64, device=device, dtype=dtype)

    checks = [
        ("rms_norm", tuple(rms(x, weight).shape)),
        ("rms_norm_manual_backward", tuple(rms_manual(x, weight).shape)),
        ("swiglu", tuple(swiglu_kernel(x, gate, up, down).shape)),
        ("sdpa", tuple(sdpa(q, k, v, is_causal=True).shape)),
        ("qkv_rope_attention", tuple(qkv_attn(x, qkv_weight, out_weight, cos=cos, sin=sin, num_heads=8).shape)),
        ("chunked_cross_entropy", tuple(chunked_cross_entropy(logits, targets, chunk_size=64).shape)),
        ("linear_cross_entropy", tuple(linear_cross_entropy(x, lm_head, targets, chunk_size=64).shape)),
    ]

    rows = []
    for name, fn in [
        ("rms_norm", lambda: rms(x, weight)),
        ("rms_norm_manual_backward", lambda: rms_manual(x, weight)),
        ("swiglu", lambda: swiglu_kernel(x, gate, up, down)),
        ("sdpa", lambda: sdpa(q, k, v, is_causal=True)),
        ("qkv_rope_attention", lambda: qkv_attn(x, qkv_weight, out_weight, cos=cos, sin=sin, num_heads=8)),
        ("chunked_cross_entropy", lambda: chunked_cross_entropy(logits, targets, chunk_size=64)),
        ("linear_cross_entropy", lambda: linear_cross_entropy(x, lm_head, targets, chunk_size=64)),
    ]:
        rows.append((name, f"{_bench(torch, fn, args.repeats):.3f} ms"))

    print(f"device={device} dtype={dtype} compile={args.compile}")
    print(make_table(("Kernel", "Output shape"), checks))
    print("")
    print(make_table(("Kernel", "Avg time"), rows))
    return 0


def _decode_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run decode-demo: pip install torch")
        return 2

    from .kv_cache import DecodeConfig, greedy_decode

    class TinyNextToken(torch.nn.Module):
        def __init__(self, vocab_size: int = 16):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size, device=input_ids.device)
            next_token = (input_ids[:, -1] + 1) % self.vocab_size
            logits[:, -1, :].scatter_(1, next_token[:, None], 1.0)
            return {"logits": logits, "past_key_values": past_key_values}

    prompt = torch.tensor([[1, 2, 3]])
    result = greedy_decode(TinyNextToken(), prompt, DecodeConfig(max_new_tokens=args.steps))
    print(result.to_text())
    return 0


def _cache_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run cache-demo: pip install torch")
        return 2

    from .kv_cache import KVCacheConfig, QuantizedStaticKVCache, StaticKVCache

    cfg = KVCacheConfig(num_layers=4, batch_size=1, num_heads=8, head_dim=64, max_seq_len=max(args.tokens, 1), dtype=torch.float16)
    cache = QuantizedStaticKVCache(cfg, quant_dtype=args.dtype) if args.quantized else StaticKVCache(cfg)
    for pos in range(args.tokens):
        key = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        value = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        for layer in range(cfg.num_layers):
            cache.append_layer(layer, key, value, position=pos)
    print(cache.stats().to_text())
    return 0


def _patch_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run patch-demo: pip install torch")
        return 2

    from .patchers import optimize_hf_model

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = 1e-6

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) * self.weight

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 16, bias=False)
            self.up_proj = torch.nn.Linear(8, 16, bias=False)
            self.down_proj = torch.nn.Linear(16, 8, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class TinyLlamaAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.head_dim = 4
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            self.o_proj = torch.nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

    class TinyHFBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = TinyRMSNorm(8)
            self.self_attn = TinyLlamaAttention()
            self.mlp = TinyMLP()

        def forward(self, x):
            x = self.input_layernorm(x)
            return self.mlp(x + self.self_attn(x))

    model = TinyHFBlock()
    _, report = optimize_hf_model(model, patch_attention=args.attention)
    print(report.to_text())
    print(model)
    return 0


def _benchmark_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run benchmark-demo: pip install torch")
        return 2

    from .benchmark import BenchmarkConfig, benchmark_callable, compare_benchmarks
    from .patchers import optimize_hf_model

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(64, 128, bias=False)
            self.up_proj = torch.nn.Linear(64, 128, bias=False)
            self.down_proj = torch.nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    x = torch.randn(8, 32, 64)
    baseline = TinyMLP().eval()
    optimized = TinyMLP().eval()
    optimized.load_state_dict(baseline.state_dict())
    optimize_hf_model(optimized)
    cfg = BenchmarkConfig(warmup=2, repeats=args.repeats)
    results = [
        benchmark_callable("baseline", lambda: baseline(x), cfg),
        benchmark_callable("patched", lambda: optimized(x), cfg),
    ]
    print(compare_benchmarks(results))
    return 0


def _compare_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run compare-demo: pip install torch")
        return 2

    from .benchmark import BenchmarkConfig, benchmark_callable
    from .compare_report import CompareReport, write_compare_html
    from .kv_quality import evaluate_attention_kv_quality
    from .memory_policy import choose_memory_policy
    from .optimization_report import OptimizationReport
    from .patchers import optimize_hf_model
    from .torch_debugger import TorchTrace

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(64, 128, bias=False)
            self.up_proj = torch.nn.Linear(64, 128, bias=False)
            self.down_proj = torch.nn.Linear(128, 64, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    x = torch.randn(8, 32, 64)
    baseline = TinyMLP().eval()
    optimized = TinyMLP().eval()
    optimized.load_state_dict(baseline.state_dict())
    _, patch_report = optimize_hf_model(optimized)
    cfg = BenchmarkConfig(warmup=2, repeats=args.repeats)
    benchmarks = [
        benchmark_callable("baseline", lambda: baseline(x), cfg),
        benchmark_callable("optimized", lambda: optimized(x), cfg),
    ]
    with TorchTrace(baseline) as baseline_trace:
        _ = baseline(x)
    with TorchTrace(optimized) as optimized_trace:
        _ = optimized(x)
    q = torch.randn(1, 4, 1, 32, dtype=torch.float16)
    k = torch.randn(1, 4, 16, 32, dtype=torch.float16)
    v = torch.randn(1, 4, 16, 32, dtype=torch.float16)
    kv_quality = evaluate_attention_kv_quality(q, k, v, quant_dtype=args.kv_dtype)
    policy = choose_memory_policy(max_vram="8GB", sequence_length=1024)
    opt_report = OptimizationReport(
        title="llm-memlab compare demo",
        benchmarks=benchmarks,
        patch_report=patch_report,
        memory_policy=policy,
        baseline_trace=baseline_trace,
        optimized_trace=optimized_trace,
        kv_quality=kv_quality,
    )
    path = write_compare_html(
        CompareReport(
            title="llm-memlab compare demo",
            benchmarks=benchmarks,
            patch_report=patch_report,
            baseline_trace=baseline_trace,
            optimized_trace=optimized_trace,
            kv_quality=kv_quality,
            memory_policy=policy,
            optimization_report=opt_report,
        ),
        args.out,
    )
    print(f"Compare HTML report written to {path}")
    return 0


def _policy_demo(args: argparse.Namespace) -> int:
    from .memory_policy import choose_memory_policy

    estimate = estimate_transformer_memory(preset_config(args.preset, sequence_length=args.seq, batch_size=1, dtype="fp16"))

    class TinyInfo:
        kv_cache_bytes_fp16 = estimate.kv_cache_bytes

    policy = choose_memory_policy(max_vram=args.max_vram, model_info=TinyInfo(), sequence_length=args.seq)
    print(policy.to_text())
    return 0


def _debug_hf(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("debug-hf requires: pip install torch transformers")
        return 2

    from .attention_debugger import attention_stats_to_text, collect_attention_stats
    from .html_report import write_trace_html
    from .inspector import load_hf_model
    from .torch_debugger import trace_forward

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        model = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.no_grad():
        _, trace = trace_forward(model, **encoded)
        _, attention_stats = collect_attention_stats(model, **encoded)
    print(trace.to_text(limit=24, show_shapes=True, show_stats=True))
    if attention_stats:
        print("")
        print("Attention debugger")
        print(attention_stats_to_text(attention_stats))
    if args.html_out:
        path = write_trace_html(trace, args.html_out, title=f"llm-memlab debug: {args.model}")
        print(f"HTML trace written to {path}")
    if args.timeline_out:
        from .html_report import write_timeline_html

        path = write_timeline_html(trace, args.timeline_out, title=f"llm-memlab timeline: {args.model}")
        print(f"Timeline report written to {path}")
    if args.interactive_out:
        from .html_report import write_interactive_html

        path = write_interactive_html(trace, args.interactive_out, title=f"llm-memlab interactive: {args.model}")
        print(f"Interactive report written to {path}")
    return 0


def _compare_hf(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("compare-hf requires: pip install torch transformers")
        return 2

    from .attention_debugger import collect_attention_stats
    from .benchmark import BenchmarkConfig, benchmark_callable
    from .compare_report import CompareReport, write_compare_html
    from .inspector import inspect_model, load_hf_model
    from .kv_quality import evaluate_attention_kv_quality
    from .memory_policy import choose_memory_policy
    from .optimization_report import OptimizationReport
    from .patchers import optimize_hf_model
    from .torch_debugger import TorchTrace

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        baseline = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
        optimized = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2

    _, patch_report = optimize_hf_model(optimized)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
    cfg = BenchmarkConfig(warmup=1, repeats=args.repeats)
    benchmarks = [
        benchmark_callable("hf-baseline-forward", lambda: baseline(**encoded), cfg),
        benchmark_callable("llm-memlab-optimized-forward", lambda: optimized(**encoded), cfg),
    ]
    with torch.no_grad():
        with TorchTrace(baseline) as baseline_trace:
            _ = baseline(**encoded)
        with TorchTrace(optimized) as optimized_trace:
            _ = optimized(**encoded)
        _, attention_stats = collect_attention_stats(baseline, **encoded)
    device = next(baseline.parameters()).device
    dtype = next(baseline.parameters()).dtype
    q = torch.randn(1, 4, 1, 32, device=device, dtype=dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32)
    k = torch.randn(1, 4, 16, 32, device=device, dtype=q.dtype)
    v = torch.randn(1, 4, 16, 32, device=device, dtype=q.dtype)
    kv_quality = evaluate_attention_kv_quality(q, k, v, quant_dtype=args.kv_dtype)
    info = inspect_model(baseline)
    policy = choose_memory_policy(
        max_vram=args.max_vram, model_info=info, sequence_length=getattr(encoded.get("input_ids"), "shape", [None, None])[-1]
    )
    opt_report = OptimizationReport(
        title=f"llm-memlab compare: {args.model}",
        benchmarks=benchmarks,
        patch_report=patch_report,
        memory_policy=policy,
        baseline_trace=baseline_trace,
        optimized_trace=optimized_trace,
        kv_quality=kv_quality,
        attention_stats=attention_stats,
    )
    path = write_compare_html(
        CompareReport(
            title=f"llm-memlab compare: {args.model}",
            benchmarks=benchmarks,
            patch_report=patch_report,
            baseline_trace=baseline_trace,
            optimized_trace=optimized_trace,
            kv_quality=kv_quality,
            memory_policy=policy,
            optimization_report=opt_report,
            attention_stats=attention_stats,
        ),
        args.out,
    )
    print(opt_report.to_text())
    print(f"Compare HTML report written to {path}")
    return 0


def _suite_hf(args: argparse.Namespace) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("suite-hf requires: pip install torch transformers")
        return 2

    from .benchmark import BenchmarkConfig
    from .benchmark_suite import benchmark_inference_suite
    from .inspector import load_hf_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        model = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
    result = benchmark_inference_suite(
        model, encoded, model_name=args.model, max_new_tokens=args.tokens, config=BenchmarkConfig(warmup=1, repeats=args.repeats)
    )
    if args.json_out or args.csv_out:
        from .benchmark_store import collect_run_metadata, records_from_suite, write_benchmark_csv, write_benchmark_json

        seq_len = int(getattr(encoded.get("input_ids"), "shape", [0, 0])[-1]) if isinstance(encoded, dict) else None
        metadata = collect_run_metadata(dtype=args.dtype, sequence_length=seq_len)
        records = records_from_suite(result, metadata=metadata)
        if args.json_out:
            print(f"Benchmark JSON written to {write_benchmark_json(records, args.json_out)}")
        if args.csv_out:
            print(f"Benchmark CSV written to {write_benchmark_csv(records, args.csv_out)}")
    print(result.to_text())
    return 0


def _drift_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run drift-demo: pip install torch")
        return 2

    from .drift_debugger import compare_layer_drift

    baseline = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
    candidate = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
    candidate.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        candidate[2].weight.add_(0.001)
    report = compare_layer_drift(baseline, candidate, torch.randn(2, 4, 8))
    print(report.to_text())
    return 0


def _drift_hf(args: argparse.Namespace) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("drift-hf requires: pip install torch transformers")
        return 2

    from .drift_debugger import compare_layer_drift
    from .inspector import load_hf_model
    from .patchers import optimize_hf_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        baseline = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
        candidate = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2
    optimize_hf_model(candidate)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
    report = compare_layer_drift(baseline, candidate, **encoded)
    print(report.to_text(limit=32))
    return 0


def _scoreboard_hf(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("scoreboard-hf requires: pip install torch transformers")
        return 2

    from .benchmark import BenchmarkConfig, benchmark_callable
    from .compare_report import scoreboard_to_html, write_scoreboard_html
    from .inspector import inspect_model, load_hf_model
    from .patchers import optimize_hf_model
    from .report import make_table

    rows = []
    cfg = BenchmarkConfig(warmup=1, repeats=args.repeats)
    for model_name in args.models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
            baseline = load_hf_model(model_name, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
            optimized = load_hf_model(model_name, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
            _, patch_report = optimize_hf_model(optimized)
            encoded = tokenizer(args.prompt, return_tensors="pt")
            if args.device:
                encoded = {key: value.to(args.device) for key, value in encoded.items()}
            baseline_bench = benchmark_callable(f"{model_name}:baseline", lambda: baseline(**encoded), cfg)
            optimized_bench = benchmark_callable(f"{model_name}:optimized", lambda: optimized(**encoded), cfg)
            info = inspect_model(baseline)
            speedup = baseline_bench.mean_ms / optimized_bench.mean_ms if optimized_bench.mean_ms else 0.0
            rows.append(
                {
                    "model": model_name,
                    "baseline_ms": baseline_bench.mean_ms,
                    "optimized_ms": optimized_bench.mean_ms,
                    "speedup": speedup,
                    "patched": patch_report.total_patched,
                    "params": info.parameter_count,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"model": model_name, "status": f"error: {exc}"})
    print(
        make_table(
            ("Model", "Status", "Base ms", "Opt ms", "Speed", "Patched"),
            [
                (
                    row.get("model"),
                    row.get("status"),
                    _fmt_float(row.get("baseline_ms")),
                    _fmt_float(row.get("optimized_ms")),
                    _fmt_speed(row.get("speedup")),
                    row.get("patched", ""),
                )
                for row in rows
            ],
        )
    )
    if args.json_out or args.csv_out:
        from .benchmark_store import BenchmarkRecord, write_benchmark_csv, write_benchmark_json

        records = [
            BenchmarkRecord(
                name=str(row.get("model")),
                kind="scoreboard",
                mean_ms=float(row.get("optimized_ms") or 0.0),
                min_ms=float(row.get("optimized_ms") or 0.0),
                max_ms=float(row.get("optimized_ms") or 0.0),
                extra=row,
            )
            for row in rows
        ]
        if args.json_out:
            print(f"Scoreboard JSON written to {write_benchmark_json(records, args.json_out)}")
        if args.csv_out:
            print(f"Scoreboard CSV written to {write_benchmark_csv(records, args.csv_out)}")
    path = write_scoreboard_html(rows, args.out, title="llm-memlab HF scoreboard")
    print(f"Scoreboard HTML written to {path}")
    return 0


def _run_hf(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("run-hf requires: pip install torch transformers")
        return 2

    from .hf_cache import plan_hf_cache
    from .inspector import inspect_model, load_hf_model
    from .memory_policy import choose_memory_policy
    from .oom_runner import OOMStrategy, run_with_oom_fallback
    from .patchers import optimize_hf_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        model = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2
    info = inspect_model(model)
    policy = choose_memory_policy(max_vram=args.max_vram, model_info=info, sequence_length=len(tokenizer(args.prompt).input_ids))
    cache_plan = plan_hf_cache(policy, model)
    optimize_hf_model(model)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}

    def generate_with_kwargs(**extra_kwargs):
        kwargs = cache_plan.generation_kwargs()
        kwargs.update(extra_kwargs)
        try:
            with torch.no_grad():
                return model.generate(**encoded, max_new_tokens=args.tokens, do_sample=False, **kwargs)
        except TypeError:
            kwargs.pop("cache_implementation", None)
            with torch.no_grad():
                return model.generate(**encoded, max_new_tokens=args.tokens, do_sample=False, **kwargs)

    result = run_with_oom_fallback(generate_with_kwargs, [OOMStrategy("policy-cache", {}), OOMStrategy("no-cache", {"use_cache": False})])
    output_ids = result.value
    print(policy.to_text())
    print("")
    print(cache_plan.to_text())
    print("")
    print(result.to_text())
    print("")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return 0


def _fmt_float(value) -> str:
    return "" if value is None else f"{value:.3f}"


def _fmt_speed(value) -> str:
    return "" if value is None else f"{value:.2f}x"


def _inspect_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run inspect-demo: pip install torch")
        return 2

    from .inspector import inspect_model

    class TinyConfig:
        model_type = "tiny-demo"
        num_hidden_layers = 2
        hidden_size = 16
        intermediate_size = 32
        num_attention_heads = 4
        num_key_value_heads = 4
        vocab_size = 128
        max_position_embeddings = 64

    class TinyRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(16))
            self.eps = 1e-6

    class TinyMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(16, 32, bias=False)
            self.up_proj = torch.nn.Linear(16, 32, bias=False)
            self.down_proj = torch.nn.Linear(32, 16, bias=False)

    class TinyModel(torch.nn.Module):
        config = TinyConfig()

        def __init__(self):
            super().__init__()
            self.norm = TinyRMSNorm()
            self.mlp = TinyMLP()

    print(inspect_model(TinyModel()).to_text())
    return 0


def _inspect_hf(args: argparse.Namespace) -> int:
    from .inspector import inspect_model, load_hf_model

    try:
        model = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model: {exc}")
        return 2
    print(inspect_model(model, max_seq_len=args.seq).to_text())
    return 0


def _benchmark_hf(args: argparse.Namespace) -> int:
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("benchmark-hf requires: pip install torch transformers")
        return 2

    from .benchmark import BenchmarkConfig, benchmark_callable, compare_benchmarks
    from .inspector import load_hf_model
    from .patchers import optimize_hf_model

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        baseline = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
        patched = load_hf_model(args.model, device=args.device, dtype=args.dtype, local_files_only=args.local_files_only)
    except Exception as exc:
        print(f"Could not load Hugging Face model/tokenizer: {exc}")
        return 2

    optimize_hf_model(patched)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    if args.device:
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
    cfg = BenchmarkConfig(warmup=1, repeats=args.repeats)

    def run(model):
        return model.generate(**encoded, max_new_tokens=args.tokens, do_sample=False)

    results = [
        benchmark_callable("hf-baseline-generate", lambda: run(baseline), cfg),
        benchmark_callable("llm-memlab-patched-generate", lambda: run(patched), cfg),
    ]
    print(compare_benchmarks(results))
    return 0


def _memory_first_hf_bench(args: argparse.Namespace) -> int:
    try:
        from .hf_runtime import assert_hf_benchmark_passed, benchmark_memory_first_hf_generate
    except RuntimeError as exc:
        print(str(exc))
        return 2

    try:
        result = benchmark_memory_first_hf_generate(
            args.model,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            adapter_tokens=args.adapter_tokens,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            cache=args.cache,
            quant_dtype=args.quant_dtype,
            allow_experimental_direct_cache=args.allow_experimental_direct_cache,
        )
    except Exception as exc:
        print(f"Could not run HF memory-first benchmark: {exc}")
        return 2
    print(result.to_text())
    if args.json_out:
        print(f"Benchmark JSON written to {result.write_json(args.json_out)}")
    if args.csv_out:
        print(f"Benchmark CSV written to {result.write_csv(args.csv_out)}")
    if args.fail_on_regression:
        try:
            assert_hf_benchmark_passed(result, min_token_agreement=args.min_token_agreement, max_slowdown_pct=args.max_slowdown_pct)
        except AssertionError as exc:
            print(str(exc))
            return 1
    return 0


def _serving_bench(args: argparse.Namespace) -> int:
    try:
        from .serving_benchmark import benchmark_serving_paths
    except RuntimeError as exc:
        print(str(exc))
        return 2

    try:
        result = benchmark_serving_paths(
            args.model,
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            adapter_tokens=args.adapter_tokens,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            cache=args.cache,
            quant_dtype=args.quant_dtype,
            include_vllm=not args.no_vllm,
            allow_experimental_direct_cache=args.allow_experimental_direct_cache,
        )
    except Exception as exc:
        print(f"Could not run serving benchmark: {exc}")
        return 2
    print(result.to_text())
    if args.json_out:
        print(f"Serving benchmark JSON written to {result.write_json(args.json_out)}")
    if args.csv_out:
        print(f"Serving benchmark CSV written to {result.write_csv(args.csv_out)}")
    if args.html_out:
        print(f"Serving dashboard HTML written to {result.write_html(args.html_out)}")
    if args.fail_on_regression and not result.passed:
        return 1
    return 0


def _hf_cache_certify(args: argparse.Namespace) -> int:
    try:
        from .hf_cache_certification import assert_hf_cache_certified, certify_hf_cache
    except RuntimeError as exc:
        print(str(exc))
        return 2

    try:
        report = certify_hf_cache(
            args.model,
            prompts=[item for item in args.prompts.split("|") if item],
            token_counts=_csv_ints(args.tokens),
            caches=[item.strip() for item in args.caches.split(",") if item.strip()],
            experimental_caches=[item.strip() for item in args.experimental_caches.split(",") if item.strip()],
            quant_dtypes=[item.strip() for item in args.quant_dtypes.split(",") if item.strip()],
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            allow_experimental_direct_cache=not args.no_experimental_direct_cache,
            max_logit_mean_abs=args.max_logit_mean_abs,
            min_logit_top1=args.min_logit_top1,
        )
    except Exception as exc:
        print(f"Could not run HF cache certification: {exc}")
        return 2
    print(report.to_text())
    if args.json_out:
        print(f"Certification JSON written to {report.write_json(args.json_out)}")
    if args.csv_out:
        print(f"Certification CSV written to {report.write_csv(args.csv_out)}")
    if args.html_out:
        print(f"Certification HTML written to {report.write_html(args.html_out)}")
    if args.fail_on_regression:
        try:
            assert_hf_cache_certified(report)
        except AssertionError as exc:
            print(str(exc))
            return 1
    return 0


def _certify_env(args: argparse.Namespace) -> int:
    from .env_certification import certify_environment

    try:
        report = certify_environment(
            model=args.model,
            prompts=(args.prompt,),
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            run_hf=not args.skip_hf,
            run_kernel=not args.skip_kernel,
        )
    except Exception as exc:
        print(f"Could not certify environment: {exc}")
        return 2
    print(report.to_text())
    if args.json_out:
        print(f"Environment certification JSON written to {report.write_json(args.json_out)}")
    if args.html_out:
        print(f"Environment certification HTML written to {report.write_html(args.html_out)}")
    if args.fail_on_regression and not report.passed:
        return 1
    return 0


def _kv_quality_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run kv-quality-demo: pip install torch")
        return 2

    from .kv_quality import evaluate_attention_kv_quality, evaluate_kv_quantization_quality

    x = torch.randn(1, args.heads, args.tokens, args.head_dim, dtype=torch.float16)
    if args.attention:
        q = torch.randn(1, args.heads, 1, args.head_dim, dtype=torch.float16)
        k = torch.randn(1, args.heads, args.tokens, args.head_dim, dtype=torch.float16)
        v = torch.randn(1, args.heads, args.tokens, args.head_dim, dtype=torch.float16)
        print(evaluate_attention_kv_quality(q, k, v, quant_dtype=args.dtype).to_text())
    else:
        print(evaluate_kv_quantization_quality(x, quant_dtype=args.dtype).to_text())
    return 0


def _bench(torch, fn, repeats: int) -> float:
    repeats = max(1, repeats)
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def _toy_block_graph(seq: int, hidden: int, intermediate: int, dtype: str) -> GraphSpec:
    graph = GraphSpec(inputs=("x",), outputs=("out",))
    for tensor in [
        TensorSpec.from_shape("x", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("qkv", (1, seq, hidden * 3), dtype=dtype, role="activation"),
        TensorSpec.from_shape("attn", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("mlp_up", (1, seq, intermediate * 2), dtype=dtype, role="activation"),
        TensorSpec.from_shape("mlp_down", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("out", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("w_qkv", (hidden, hidden * 3), dtype=dtype, role="parameter"),
        TensorSpec.from_shape("w_mlp_up", (hidden, intermediate * 2), dtype=dtype, role="parameter"),
        TensorSpec.from_shape("w_mlp_down", (intermediate, hidden), dtype=dtype, role="parameter"),
    ]:
        graph.add_tensor(tensor)
    graph.add_op(OperationSpec.make("qkv_proj", "linear", ("x", "w_qkv"), ("qkv",)))
    graph.add_op(OperationSpec.make("attention", "flash_attention", ("qkv",), ("attn",)))
    graph.add_op(OperationSpec.make("mlp_up", "swiglu_up", ("attn", "w_mlp_up"), ("mlp_up",)))
    graph.add_op(OperationSpec.make("mlp_down", "linear", ("mlp_up", "w_mlp_down"), ("mlp_down",)))
    graph.add_op(OperationSpec.make("residual", "add", ("x", "mlp_down"), ("out",)))
    return graph
