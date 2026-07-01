# llm-memlab

`llm-memlab` is a memory-first LLM architecture prototype:

- A small graph IR that tracks tensor shape, dtype, role, and lifetime.
- A transformer memory estimator for inference, LoRA, and full fine-tuning.
- A buffer planner that shows peak live memory and possible reuse.
- Optional PyTorch hooks for module-level memory and numerical debugging.

This is intentionally not a PyTorch replacement yet. It is the first useful layer:
keep PyTorch compatibility, make memory visible, then move hot paths into custom
backends or Triton kernels later.

## How To Use In 5 Minutes

From this checkout:

```powershell
$env:PYTHONPATH = "src"
python -m llm_memlab backend-demo
python -m llm_memlab fused-decode-bench --q-heads 4 --kv-heads 2 --tokens 64 --head-dim 32 --repeats 3 --json-out fused.json --csv-out fused.csv --html-out fused.html
python examples/memory_first_generate.py
python examples/deep_debug_report.py
python examples/regression_benchmark_ci.py
```

For CUDA/Triton validation on a GPU machine:

```powershell
python -m unittest discover -s tests -p test_cuda_triton.py
python -m llm_memlab kernel-demo --device cuda --repeats 3
```

### Production Docs

- [Stable vs Experimental](docs/stable_vs_experimental.md)
- [What Is Stable Today](docs/stable_today.md)
- [Known Limitations](docs/known_limitations.md)
- [Use In 5 Minutes](docs/quickstart_5_minutes.md)
- [Production Integration Guide](docs/production_integration_guide.md)
- [Production Checklist](docs/production_checklist.md)
- [How To Integrate In CI](docs/ci_integration.md)
- [HF Adapter Limitations](docs/hf_adapter_limitations.md)
- [Kernel Policy Decision Table](docs/kernel_policy_decision_table.md)
- [vLLM vs llm-memlab](docs/vllm_vs_llm_memlab.md)
- [Deprecation And Versioning Policy](docs/versioning.md)

Use `llm_memlab.production` for stable APIs and `llm_memlab.experimental` only for opt-in certified kernel paths.
### Production Notes

For production-oriented use, keep the stable path conservative:

- Use `benchmark-compare` and `BenchmarkGateConfig` in CI to fail on latency or quality regressions.
- Store benchmark artifacts with schema version, hardware metadata, warmup/repeats, seed, backend, dtype, and sequence length.
- Use `QualityThresholds` and `assert_quality_regression()` as a required merge gate for optimized model paths.
- Use `select_kernel_policy()` with the default policy for stable fallback. Paged fused Triton kernels are labeled `triton-experimental` and require `KernelPolicy(allow_experimental=True)` or `backend="triton-experimental"`.
- Use `profile_decode_memory()`/`write_memory_profile_json()` for observability artifacts; the profiler does not alter model numerics.
- Use `select_memory_adapter()` for family-specific HF adapter selection (`LlamaMemoryAdapter`, `QwenMemoryAdapter`, `MistralMemoryAdapter`) and keep model-family integration tests local to the adapter being promoted.
- Use `serving-bench` to compare Hugging Face generate, llm-memlab memory-first HF, and optional `vllm-serving` in one JSON/CSV/HTML dashboard.
### Runtime v3 Features

- Paged fused decode v3 supports int8 and uint8 paged K/V, GQA/MQA, batch-specific page tables shaped `[batch, logical_page]`, variable sequence lengths shaped `[batch]`, and a streaming softmax fallback for long contexts. It is intentionally exposed as `triton-experimental` until multi-GPU coverage is broad enough.
- `MemoryFirstTransformersCache` implements a small Transformers Cache-compatible facade with `update()`, `get_seq_length()`, `to_legacy_cache()`, and `reorder_cache()` for Llama/Qwen/Mistral-style decoders.
- `install_memory_first_generate()` can conservatively inject a memory-first cache into `model.generate()`, falling back to the original call when a model rejects external cache objects.
- `fused-decode-bench` runs a CUDA benchmark matrix and can write JSON/CSV/HTML plus fail on speed or quality regression.
- `profile_decode_memory()` records CUDA allocated/reserved/peak memory around decode steps and exports HTML.
- `run_quality_regression()` checks logits, top-k overlap, generated-token agreement, and next-token-loss delta.
- `select_kernel_policy()` explains why it selected torch, Triton, dense fused decode, or paged fused decode.
## Quick Start

Use it directly from this checkout:

```powershell
$env:PYTHONPATH = "src"
```

Run an estimate without installing PyTorch:

```powershell
python -m llm_memlab estimate --preset 7b-like --seq 2048 --batch 1 --training lora --lora-rank 16
```

Run the built-in planner demo:

```powershell
python -m llm_memlab plan-demo
```

Trace a tiny PyTorch model if `torch` is installed:

```powershell
python -m llm_memlab trace-demo
```

## Python API

```python
from llm_memlab import TransformerConfig, estimate_transformer_memory

cfg = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    intermediate_size=11008,
    num_attention_heads=32,
    vocab_size=32000,
    sequence_length=2048,
    batch_size=1,
    training="lora",
    lora_rank=16,
)

estimate = estimate_transformer_memory(cfg)
print(estimate.to_text())
```


## Optimized Kernels

The kernel layer provides PyTorch-compatible building blocks for transformer hot paths:

- `rms_norm` with fp32 variance and output dtype preservation.
- `rms_norm_manual_backward` for compact PyTorch autograd storage during training.
- Optional `triton_rms_norm`, `triton_apply_rope`, `triton_swiglu_activation`, and real int8/uint8 per-token quantize/dequantize Triton kernels with PyTorch fallback.
- `apply_rope` for interleaved rotary embeddings.
- `swiglu` for gated MLP blocks.
- `scaled_dot_product_attention`, dispatching to PyTorch SDPA and FlashAttention-style kernels where available.
- `chunked_cross_entropy` to reduce loss-time memory spikes on long sequences.
- `linear_cross_entropy` to compute LM-head projection and CE in chunks without materializing full `[batch, seq, vocab]` logits.
- `qkv_rope_attention_cached` to write K/V into `StaticKVCache` or `QuantizedStaticKVCache` and attend over cached tokens during decode.

Run a local microbenchmark:

```powershell
python -m llm_memlab kernel-demo --device auto --repeats 20
```

Use the kernels directly:

```python
from llm_memlab.kernels import linear_cross_entropy, rms_norm_manual_backward, scaled_dot_product_attention

y = rms_norm_manual_backward(x, weight)
out = scaled_dot_product_attention(q, k, v, is_causal=True)
loss = linear_cross_entropy(hidden, lm_head.weight, labels, chunk_size=512)
```


Cache-aware decoder block:

```python
import torch
from llm_memlab.kv_cache import KVCacheConfig, StaticKVCache
from llm_memlab.modules import OptimizedDecoderBlock, build_rope_cache

block = OptimizedDecoderBlock(
    hidden_size=4096,
    intermediate_size=11008,
    num_heads=32,
    layer_idx=0,
    use_triton=True,
)
cache = StaticKVCache(KVCacheConfig(
    num_layers=1,
    batch_size=1,
    num_heads=32,
    head_dim=128,
    max_seq_len=4096,
    dtype=torch.float16,
    device="cuda",
))
cos, sin = build_rope_cache(4096, 128, dtype=torch.float16, device="cuda")

# Decode token at position t without reallocating K/V tensors.
y = block(x_t, cos=cos[t:t+1], sin=sin[t:t+1], kv_cache=cache, cache_position=t)
```

Triton is optional. If `triton` is not installed or tensors are on CPU, the Triton wrapper functions fall back to the PyTorch reference path.
Use `torch.compile` when available:

```python
from llm_memlab.kernels import KernelConfig, kernel

rms = kernel("rms_norm", KernelConfig(compile=True))
y = rms(x, weight)
```


## KV Cache Decode

The inference layer includes a static KV cache and a generic HuggingFace-style decode loop:

```python
import torch
from llm_memlab.kv_cache import DecodeConfig, KVCacheConfig, StaticKVCache, greedy_decode

cache = StaticKVCache(KVCacheConfig(
    num_layers=32,
    batch_size=1,
    num_heads=32,
    head_dim=128,
    max_seq_len=4096,
    dtype=torch.float16,
    device="cuda",
))

result = greedy_decode(model, input_ids, DecodeConfig(max_new_tokens=64))
print(result.to_text())
print(cache.stats().to_text())
```

Run the tiny local demo:

```powershell
python -m llm_memlab decode-demo --steps 8
```

This makes inference less black-box by reporting per-token latency, throughput, generated token IDs, and cache length when the model returns `past_key_values`.



## Architecture Inspector

Inspect a local model object or a Hugging Face causal LM:

```powershell
python -m llm_memlab inspect-demo
python -m llm_memlab inspect-hf --model Qwen/Qwen2.5-0.5B --local-files-only
```

The inspector reports core architecture fields, parameter counts, dtype/device summaries, patchable RMSNorm/MLP modules, attention candidates, and fp16/int8 KV cache estimates.

## Hugging Face Benchmark

Benchmark `generate()` before and after conservative llm-memlab patching:

```powershell
python -m llm_memlab benchmark-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --tokens 32 --repeats 3
```

Use `--local-files-only` when the model is already cached locally.

## Quantized KV Quality

Measure KV cache storage error and compression on random K/V-like tensors. Common storage dtypes are supported: `int8`, `uint8`, `fp16`, `bf16`, `fp32`, and `fp8_e4m3fn` when the local PyTorch build exposes it.

```powershell
python -m llm_memlab kv-quality-demo --tokens 128 --heads 32 --head-dim 128 --dtype int8
python -m llm_memlab kv-quality-demo --tokens 128 --heads 32 --head-dim 128 --dtype uint8
```

Use `--attention` to compare real SDPA output before and after quantizing/dequantizing K/V, which is closer to the inference quality question than tensor reconstruction alone:

```powershell
python -m llm_memlab kv-quality-demo --tokens 128 --heads 32 --head-dim 128 --dtype int8 --attention
```

## Model Patching

Patch Hugging Face-style Llama/Qwen modules conservatively:

```python
from llm_memlab.patchers import optimize_hf_model

model, report = optimize_hf_model(model, use_triton=True)
print(report.to_text())
```

The patcher replaces RMSNorm-like modules and SwiGLU MLP modules when their local interfaces match. Attention replacement is opt-in with `patch_attention=True` or `optimize_llama_qwen_attention()` because Hugging Face attention signatures vary across model families, cache implementations, masks, RoPE, and grouped-query attention layouts. The current attention adapter targets simple Llama/Qwen-like equal-width q/k/v/o modules and packs q/k/v into one projection before PyTorch SDPA.

Try the local demo:

```powershell
python -m llm_memlab patch-demo
python -m llm_memlab patch-demo --attention
```

## Benchmarking

Benchmark any callable or model forward pass:

```python
from llm_memlab.benchmark import BenchmarkConfig, benchmark_callable

result = benchmark_callable("forward", lambda: model(input_ids), BenchmarkConfig(warmup=3, repeats=20))
print(result.to_text())
```

Run the built-in patcher comparison:

```powershell
python -m llm_memlab benchmark-demo --repeats 20
```

Write one HTML report that combines baseline vs optimized benchmark, patch coverage, trace summary, and KV attention quality:

```powershell
python -m llm_memlab compare-demo --out compare_demo.html --repeats 5 --kv-dtype int8
```

## Visual Debugger

Export a layer-level HTML report from a trace:

```python
from llm_memlab.html_report import write_trace_html
from llm_memlab.torch_debugger import trace_forward

_, trace = trace_forward(model, x)
write_trace_html(trace, "trace.html")
```

Or via CLI:

```powershell
python -m llm_memlab trace-demo --html-out trace_demo.html
```


### Quantized KV Cache

Use `QuantizedStaticKVCache` to store K/V tensors as int8 with one scale per `[batch, head, token]` vector. Reads dequantize back to the configured dtype, so cache-aware attention can use the same API:

```python
import torch
from llm_memlab.kv_cache import KVCacheConfig, QuantizedStaticKVCache

cache = QuantizedStaticKVCache(KVCacheConfig(
    num_layers=32,
    batch_size=1,
    num_heads=32,
    head_dim=128,
    max_seq_len=4096,
    dtype=torch.float16,
    device="cuda",
))

k, v = cache.append_layer(layer_idx, k_new, v_new, position=t)
print(cache.stats().to_text())
```

Compare fp and quantized cache memory locally:

```powershell
python -m llm_memlab cache-demo --tokens 128
python -m llm_memlab cache-demo --tokens 128 --quantized --dtype int8
python -m llm_memlab cache-demo --tokens 128 --quantized --dtype uint8
```

The int8/uint8 cache is lossy but usually much smaller. Compression is below the ideal 2x for fp16 because per-token metadata is stored alongside K/V. Floating storage modes (`fp16`, `bf16`, `fp32`) are also available for controlled quality and allocator experiments.

## PyTorch Runtime Debugging

```python
import torch
from llm_memlab.torch_debugger import TorchTrace

model = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 16),
)

x = torch.randn(8, 16)

with TorchTrace(model) as trace:
    y = model(x)

print(trace.to_text())
```

The trace records module runtime, input/output tensor bytes, parameter counts, input/output shapes, activation statistics, CUDA allocator deltas when CUDA is available, NaN/Inf flags, hot layers, and optional gradient statistics.


## Memory-First Runtime Policy

Choose an optimization policy from a VRAM budget. The policy recommends KV dtype, static vs paged cache, chunk size, and attention backend tradeoffs:

```powershell
python -m llm_memlab policy-demo --max-vram 8GB --preset 7b-like --seq 4096
```

From Python:

```python
from llm_memlab import choose_memory_policy

policy = choose_memory_policy(max_vram="8GB", model_info=info, sequence_length=4096)
print(policy.to_text())
```

## Explainable Optimization Reports

`compare-demo` now writes a single HTML report with benchmark results, patch coverage, trace summary, KV attention quality, memory policy, and inferred findings:

```powershell
python -m llm_memlab compare-demo --out compare_demo.html --repeats 5 --kv-dtype int8
```

For cached/local Hugging Face models, compare baseline vs optimized forward passes:

```powershell
python -m llm_memlab compare-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --out compare_hf.html --local-files-only
```

## HF Debugger

Trace a Hugging Face causal LM forward pass and collect attention entropy/head statistics:

```powershell
python -m llm_memlab debug-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --html-out debug.html --local-files-only
```

The debugger reports layer time, shapes, bytes, activation stats, NaN/Inf flags, hot layers, and attention stats such as entropy, maximum attention probability, and dead-head fraction.

## Paged KV Cache and Quantized Attention

`PagedKVCache` stores decode tokens in fixed-size pages while keeping the same `append_layer/get_layer` API as `StaticKVCache`:

```python
from llm_memlab import KVCacheConfig, PagedKVCache

cache = PagedKVCache(KVCacheConfig(num_layers=32, batch_size=1, num_heads=32, head_dim=128, max_seq_len=4096), page_size=32)
```

`quantized_kv_attention` exposes the quantized-KV attention contract with a portable dequant+SDPA fallback. On CUDA with Triton, single-token non-causal decode (`q` shaped `[batch, heads, 1, head_dim]`) can use the fused decode path that dequantizes K/V inside the attention kernel:

```python
from llm_memlab import quantized_kv_attention

out = quantized_kv_attention(q, k, v, quant_dtype="int8")
```

## HF Scoreboard and Memory-Budget Run

Benchmark one or more cached Hugging Face models and write a scoreboard HTML report:

```powershell
python -m llm_memlab scoreboard-hf --models Qwen/Qwen2.5-0.5B TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello" --out scoreboard_hf.html --local-files-only
```

Run generation with a memory-first policy summary before the decoded text:

```powershell
python -m llm_memlab run-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --tokens 32 --max-vram 8GB --local-files-only
```

## Timeline Debugger

The debugger can now write a timeline-style HTML view in addition to the table report:

```powershell
python -m llm_memlab trace-demo --timeline-out trace_timeline.html
python -m llm_memlab debug-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --timeline-out debug_timeline.html --local-files-only
```

## Paged KV Cache v2

`PagedKVCache` now tracks a logical page table, free pages, tail-page release, and fragmentation:

```python
from llm_memlab import KVCacheConfig, PagedKVCache

cache = PagedKVCache(KVCacheConfig(num_layers=1, batch_size=1, num_heads=8, head_dim=64, max_seq_len=4096), page_size=32)
print(cache.page_table)
print(cache.fragmentation_report())
cache.release_pages(1)
```

## Attention Patcher v2

The opt-in attention patcher now supports simple GQA/MQA-style layouts where K/V have fewer heads than Q, repeats K/V heads for SDPA, accepts tuple `position_embeddings=(cos, sin)` for RoPE best-effort application, and reports clearer skip reasons.

```python
from llm_memlab import optimize_llama_qwen_attention

model, report = optimize_llama_qwen_attention(model)
print(report.to_text())
```

## Quantized Attention Backend Selector

`quantized_kv_attention` now accepts `backend="auto" | "torch" | "triton"`, and `select_quantized_attention_backend()` reports the selected backend, implementation label, quant dtype, and fallback reason. The current implementation uses fused Triton decode attention when the shape is supported, and falls back to Triton quant/dequant+PyTorch SDPA or portable PyTorch dequant+SDPA otherwise.

```python
from llm_memlab import quantized_kv_attention

out = quantized_kv_attention(q, k, v, quant_dtype="int8", backend="auto")
```

## Benchmark Suite and Drift Debugger

Run a prefill/generate/VRAM benchmark suite for a cached Hugging Face model:

```powershell
python -m llm_memlab suite-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --tokens 16 --local-files-only
```

The suite reports prompt tokens, prefill latency, generate latency, prefill tok/s, decode tok/s, and peak CUDA memory when CUDA is available.

Compare layer-by-layer output drift between two model paths:

```powershell
python -m llm_memlab drift-demo
python -m llm_memlab drift-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --local-files-only
```

From Python:

```python
from llm_memlab import benchmark_inference_suite, compare_layer_drift

suite = benchmark_inference_suite(model, encoded, model_name="my-model", max_new_tokens=16)
print(suite.to_text())

drift = compare_layer_drift(baseline_model, candidate_model, **encoded)
print(drift.to_text(limit=32))
```

## CI and Examples

The repository includes GitHub Actions CI for Python 3.10-3.12 and small examples:

```powershell
python examples/memory_policy_example.py
python examples/drift_debugger_example.py
python examples/hf_suite_example.py Qwen/Qwen2.5-0.5B
```

The HF example expects the model to be cached locally.

## Output Quality, Benchmark Store, and Runtime Backends

Measure output drift beyond raw latency:

```powershell
python -m llm_memlab quality-demo
```

```python
from llm_memlab import compare_logits, compare_token_sequences

logit_quality = compare_logits(baseline_logits, candidate_logits, top_k=5)
token_quality = compare_token_sequences(baseline_ids, candidate_ids)
print(logit_quality.to_text())
print(token_quality.to_text())
```

Persist benchmark results as JSON or CSV so multiple runs can be compared later:

```powershell
python -m llm_memlab suite-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --local-files-only --json-out suite.json --csv-out suite.csv
python -m llm_memlab scoreboard-hf --models Qwen/Qwen2.5-0.5B --local-files-only --json-out scoreboard.json --csv-out scoreboard.csv
```

The backend registry reports runtime availability and priority for torch, CUDA, and Triton:

```powershell
python -m llm_memlab backend-demo
```

```python
from llm_memlab import default_backend_registry

registry = default_backend_registry()
print(registry.best("triton", "cuda", "torch"))
```

## Interactive Debugger and OOM-Aware HF Runs

The debugger can emit a sortable/filterable HTML table for layer-level inspection:

```powershell
python -m llm_memlab trace-demo --interactive-out trace_interactive.html
python -m llm_memlab debug-hf --model Qwen/Qwen2.5-0.5B --prompt "Hello" --interactive-out debug_interactive.html --local-files-only
```

`run-hf` now combines the memory policy, HF cache hints, conservative patching, and OOM fallback. If the policy cache path raises a CUDA OOM, it retries with cache disabled and reports the selected strategy:

```powershell
python -m llm_memlab run-hf --model Qwen/Qwen2.5-0.5B --tokens 32 --max-vram 8GB --local-files-only
```

From Python:

```python
from llm_memlab import OOMStrategy, plan_hf_cache, run_with_oom_fallback

cache_plan = plan_hf_cache(policy, model)
result = run_with_oom_fallback(
    lambda **kw: model.generate(**encoded, **{**cache_plan.generation_kwargs(), **kw}),
    [OOMStrategy("policy-cache", {}), OOMStrategy("no-cache", {"use_cache": False})],
)
```

## Triton Quantization Kernels

The Triton layer now includes real generic last-dimension CUDA kernels for per-token `int8` and asymmetric `uint8` quantize/dequantize. They accept tensors shaped like `[batch, heads, tokens, head_dim]` or any tensor where the last dimension is the vector to quantize. CPU, missing-Triton, and oversized hidden dimensions fall back to the PyTorch reference path.

```python
from llm_memlab import (
    triton_dequantize_int8_per_token,
    triton_dequantize_uint8_per_token,
    triton_quantize_int8_per_token,
    triton_quantize_uint8_per_token,
)

q8, scale = triton_quantize_int8_per_token(k)
k_roundtrip = triton_dequantize_int8_per_token(q8, scale, dtype=k.dtype)

qu8, scale, zero_point = triton_quantize_uint8_per_token(k)
k_uint8 = triton_dequantize_uint8_per_token(qu8, scale, zero_point, dtype=k.dtype)
```

`quantized_kv_attention(..., backend="triton")` now uses fused Triton decode attention for single-token decode, so K/V are dequantized inside the softmax attention kernel instead of materializing full dequantized K/V tensors first. Causal mode, non-decode shapes, masks, dropout, and oversized head/token blocks safely fall back to Triton quant/dequant + PyTorch SDPA.
Run CUDA/Triton validation on a GPU machine:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p test_cuda_triton.py
python -m llm_memlab kernel-demo --device cuda --repeats 3
```

## Memory-First Runtime v2

Fused Triton decode v2 supports single-token non-causal decode for MHA, GQA, and MQA layouts where `q_heads` is divisible by `kv_heads`. The int8 paged path reads K/V through a page table, so decode can avoid materializing a full dequantized K/V tensor before attention:

```python
from llm_memlab import benchmark_fused_decode_attention, quantized_kv_attention

out = quantized_kv_attention(q, k, v, quant_dtype="int8", backend="triton")
bench = benchmark_fused_decode_attention(q_heads=8, kv_heads=2, tokens=128, head_dim=64, repeats=20)
print(bench.to_text())
```

Paged fused decode now covers int8 and uint8 K/V with `head_dim <= 256`; long contexts use the streaming paged path to avoid materializing full dequantized K/V. Unsupported masks, dropout, causal multi-token paths, or CPU tensors fall back to portable reference paths.

## Memory-First Hugging Face Adapter

`MemoryFirstHFAdapter` provides a real custom generate loop for HF-style causal LMs that accept and return legacy tuple `past_key_values`. It stores returned K/V in `QuantizedStaticKVCache` or `PagedKVCache` between decode steps:

```python
from llm_memlab import MemoryFirstHFConfig, memory_first_generate

result = memory_first_generate(
    model,
    input_ids,
    MemoryFirstHFConfig(cache="quantized", quant_dtype="int8", max_new_tokens=32),
)
print(result.to_text())
```

This is intentionally conservative: newer model-specific HF `Cache` objects may still need dedicated adapters.

## Deep Debugger

The deep debugger compares baseline vs optimized models layer-by-layer, reports output quality drift, flags NaN/Inf, detects collapsed/dead attention heads when q/k projections are visible, and writes one interactive HTML page with timeline, memory, quality, and clickable layer details:

```python
from llm_memlab import build_deep_debug_report, write_deep_debug_html

report = build_deep_debug_report(baseline_model, optimized_model, **encoded)
write_deep_debug_html(report, "deep_debug.html")
```

## Benchmark Database v2

Benchmark records can now carry run metadata such as GPU name, torch version, Triton version, dtype, sequence length, and git commit. JSON/CSV records can be compared across runs, and regressions can fail a test when slowdown exceeds a threshold:

```python
from llm_memlab import (
    assert_no_regressions,
    collect_run_metadata,
    compare_record_sets,
    read_benchmark_json,
    records_from_suite,
    write_benchmark_json,
)

metadata = collect_run_metadata(dtype="fp16", sequence_length=4096)
write_benchmark_json(records_from_suite(suite, metadata=metadata), "candidate.json")
comparisons = compare_record_sets(read_benchmark_json("baseline.json"), read_benchmark_json("candidate.json"), max_slowdown_pct=10.0)
assert_no_regressions(comparisons)
```

## Library API Polish

The package now exposes `llm_memlab.backends` for backend availability/selection and `KernelPolicy` for choosing kernel backend, quant dtype, fused decode preference, and paged-attention limits:

```python
from llm_memlab.backends import default_backend_registry
from llm_memlab import KernelPolicy

registry = default_backend_registry()
policy = KernelPolicy(backend="triton", quant_dtype="int8", prefer_fused_decode=True)
policy.validate()
```

CI runs Python 3.10-3.12 and includes a CUDA-optional test profile that skips GPU tests when CUDA/Triton are unavailable.
## Roadmap

1. Add model importers for Hugging Face transformer blocks.
2. Add fuller Triton kernels for backward paths, attention variants, and cross entropy.
3. Add graph rewrites for activation checkpointing and CPU/NVMe offload.
4. Add a `torch.compile` backend that consumes this IR.
5. Add a browser timeline for tensor lifetimes and allocator snapshots.

