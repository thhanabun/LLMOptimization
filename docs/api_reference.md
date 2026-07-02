# API Reference

This is a compact reference for the stable public surface in `llm_memlab.production`. Experimental kernels and backend-specific internals should stay behind explicit opt-in paths.

## Import Pattern

```python
from llm_memlab.production import (
    benchmark_serving_paths,
    benchmark_memory_first_hf_generate,
    certify_hf_cache,
    certify_model_matrix,
    default_kernel_policy,
    detect_hardware_profile,
    select_hf_cache_policy,
    select_kernel_policy,
)
```

## Runtime And Backend Detection

```python
from llm_memlab.production import detect_hardware_profile
from llm_memlab.backend_registry import default_backend_registry

hardware = detect_hardware_profile()
registry = default_backend_registry()
print(hardware.to_text())
print(registry.to_text())
```

Use this before selecting CUDA/Triton/vLLM paths.

## Memory Estimation

```python
from llm_memlab import estimate_transformer_memory, preset_config

cfg = preset_config("7b-like", sequence_length=4096, batch_size=1)
estimate = estimate_transformer_memory(cfg)
print(estimate.to_text())
```

## HF Memory-First Benchmark

```python
from llm_memlab.production import benchmark_memory_first_hf_generate

result = benchmark_memory_first_hf_generate(
    "/models/TinyLlama-1.1B-Chat-v1.0",
    prompt="hello",
    max_new_tokens=1,
    device="auto",
    dtype="auto",
    local_files_only=True,
    cache="paged",
)
print(result.to_text())
result.write_json("memory_first_hf.json")
result.write_csv("memory_first_hf.csv")
```

## Serving Benchmark

```python
from llm_memlab.production import benchmark_serving_paths

result = benchmark_serving_paths(
    "/models/TinyLlama-1.1B-Chat-v1.0",
    prompt="hello",
    max_new_tokens=1,
    device="auto",
    dtype="auto",
    local_files_only=True,
    cache="paged",
    include_vllm=True,
)
result.write_json("serving_bench.json")
result.write_csv("serving_bench.csv")
result.write_html("serving_dashboard.html")
```

## HF Cache Certification

```python
from llm_memlab.production import certify_hf_cache

report = certify_hf_cache(
    "/models/TinyLlama-1.1B-Chat-v1.0",
    prompts=("hello",),
    token_counts=(1,),
    caches=("paged",),
    device="auto",
    dtype="auto",
    local_files_only=True,
)
print(report.to_text())
report.write_json("hf_cache_cert.json")
report.write_html("hf_cache_cert.html")
```

## Model Matrix

```python
from llm_memlab.production import ModelCertificationTarget, certify_model_matrix

targets = (
    ModelCertificationTarget("tinyllama", "llama", "/models/TinyLlama-1.1B-Chat-v1.0", local_files_only=True),
)
matrix = certify_model_matrix(targets, prompts=("hello",), device="auto", dtype="auto")
matrix.write_json("model_matrix.json")
matrix.write_profiles("profiles.json")
```

## Policy Selection

```python
from llm_memlab.production import HFCachePolicy, select_hf_cache_policy

decision = select_hf_cache_policy(
    family="llama",
    prompt_tokens=32,
    policy=HFCachePolicy(requested_cache="quantized", quantized_profile_paths=("profiles.json",)),
)
print(decision.to_text())
```

## Stability Notes

- `llm_memlab.production` is the recommended stable import surface for v0.1.x.
- `llm_memlab.experimental` and Triton/CuTile fused paths require explicit certification before production use.
- Generated JSON/CSV/HTML artifacts are designed for CI evidence and debugging reports.
