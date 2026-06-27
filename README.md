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
- `apply_rope` for interleaved rotary embeddings.
- `swiglu` for gated MLP blocks.
- `scaled_dot_product_attention`, dispatching to PyTorch SDPA and FlashAttention-style kernels where available.
- `chunked_cross_entropy` to reduce loss-time memory spikes on long sequences.

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
2. Add Triton fused kernels for RMSNorm, RoPE, SwiGLU, and cross entropy.
3. Add graph rewrites for activation checkpointing and CPU/NVMe offload.
4. Add a `torch.compile` backend that consumes this IR.
5. Add a browser timeline for tensor lifetimes and allocator snapshots.




