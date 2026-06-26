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

The trace records module runtime, output tensor bytes, parameter bytes,
CUDA allocator deltas when CUDA is available, and NaN/Inf flags.

## Roadmap

1. Add model importers for Hugging Face transformer blocks.
2. Add Triton fused kernels for RMSNorm, RoPE, SwiGLU, and cross entropy.
3. Add graph rewrites for activation checkpointing and CPU/NVMe offload.
4. Add a `torch.compile` backend that consumes this IR.
5. Add a browser timeline for tensor lifetimes and allocator snapshots.
