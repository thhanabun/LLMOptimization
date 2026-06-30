# CuTile Backend

CuTile support is experimental and production-gated.

## Policy

- `torch` remains the correctness fallback.
- `triton` remains the portable CUDA default.
- `cutile-experimental` is considered only when a CuTile runtime is importable and the GPU is Hopper/Blackwell-class.
- Ampere and older GPUs, including RTX 30-series gaming cards, fall back to Triton or Torch by default.

## Current Kernel Contract

`llm_memlab.backends.cutile.cutile_fused_decode_attention()` exposes a single-token paged decode attention contract for fp16/bf16/fp32 K/V:

- batch = 1
- decode step = 1
- GQA/MQA supported when `q_heads % kv_heads == 0`
- paged K/V layout `[batch, kv_heads, pages, page_size, head_dim]`

If a compatible CuTile runtime op is not available, the function uses PyTorch SDPA fallback and reports `backend_used="torch-fallback"`.

## Certification

Use:

```bash
python -m llm_memlab cutile-certify --tokens 128 --q-heads 8 --kv-heads 2
```

Certification passes only when:

- the CuTile runtime path is actually used,
- the detected GPU/runtime is a production candidate,
- output quality passes the configured logits/top-k thresholds.

This keeps CuTile useful for research without silently promoting an unverified backend into production.
