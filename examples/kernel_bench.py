import time

import torch

from llm_memlab.kernels import (
    chunked_cross_entropy,
    linear_cross_entropy,
    qkv_rope_attention,
    rms_norm,
    rms_norm_manual_backward,
    scaled_dot_product_attention,
    swiglu,
)
from llm_memlab.modules import OptimizedDecoderBlock, build_rope_cache


def bench(name, fn, repeats=25):
    for _ in range(3):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / repeats
    print(f"{name}: {elapsed:.3f} ms")


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

x = torch.randn(2, 256, 1024, device=device, dtype=dtype)
weight = torch.randn(1024, device=device, dtype=dtype)
gate = torch.randn(4096, 1024, device=device, dtype=dtype)
up = torch.randn(4096, 1024, device=device, dtype=dtype)
down = torch.randn(1024, 4096, device=device, dtype=dtype)
q = torch.randn(2, 8, 256, 64, device=device, dtype=dtype)
k = torch.randn(2, 8, 256, 64, device=device, dtype=dtype)
v = torch.randn(2, 8, 256, 64, device=device, dtype=dtype)
qkv_weight = torch.randn(3072, 1024, device=device, dtype=dtype)
out_weight = torch.randn(1024, 1024, device=device, dtype=dtype)
logits = torch.randn(2, 256, 4096, device=device, dtype=dtype)
targets = torch.randint(0, 4096, (2, 256), device=device)
lm_head = torch.randn(4096, 1024, device=device, dtype=dtype)
block = OptimizedDecoderBlock(1024, 4096, 8).to(device=device, dtype=dtype)
cos, sin = build_rope_cache(256, 128, device=device, dtype=dtype)

bench("rms_norm", lambda: rms_norm(x, weight))
bench("rms_norm_manual_backward", lambda: rms_norm_manual_backward(x, weight))
bench("swiglu", lambda: swiglu(x, gate, up, down), repeats=5)
bench("sdpa", lambda: scaled_dot_product_attention(q, k, v, is_causal=True))
bench("qkv_rope_attention", lambda: qkv_rope_attention(x, qkv_weight, out_weight, cos=cos, sin=sin, num_heads=8), repeats=5)
bench("chunked_cross_entropy", lambda: chunked_cross_entropy(logits, targets, chunk_size=128))
bench("linear_cross_entropy", lambda: linear_cross_entropy(x, lm_head, targets, chunk_size=128))
bench("optimized_decoder_block", lambda: block(x, cos=cos, sin=sin), repeats=3)
