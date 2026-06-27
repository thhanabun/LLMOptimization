# llm-memlab

`llm-memlab` is a memory-first LLM architecture prototype:

- A small graph IR that tracks tensor shape, dtype, role, and lifetime.
- A transformer memory estimator for inference, LoRA, and full fine-tuning.
- A buffer planner that shows peak live memory and possible reuse.
- Optional PyTorch hooks for module-level memory and numerical debugging.

This is intentionally not a PyTorch replacement yet. It is the first useful layer:
keep PyTorch compatibility, make memory visible, then move hot paths into custom
backends or Triton kernels later.

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
- Optional `triton_rms_norm`, `triton_apply_rope`, `triton_swiglu_activation`, and int8 quantize/dequantize hooks with PyTorch fallback.
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

## Roadmap

1. Add model importers for Hugging Face transformer blocks.
2. Add fuller Triton kernels for backward paths, attention variants, and cross entropy.
3. Add graph rewrites for activation checkpointing and CPU/NVMe offload.
4. Add a `torch.compile` backend that consumes this IR.
5. Add a browser timeline for tensor lifetimes and allocator snapshots.











