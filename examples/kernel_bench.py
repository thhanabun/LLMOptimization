import time

import torch

from llm_memlab.kernels import chunked_cross_entropy, rms_norm, scaled_dot_product_attention, swiglu


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
logits = torch.randn(2, 256, 4096, device=device, dtype=dtype)
targets = torch.randint(0, 4096, (2, 256), device=device)

bench("rms_norm", lambda: rms_norm(x, weight))
bench("swiglu", lambda: swiglu(x, gate, up, down), repeats=5)
bench("sdpa", lambda: scaled_dot_product_attention(q, k, v, is_causal=True))
bench("chunked_cross_entropy", lambda: chunked_cross_entropy(logits, targets, chunk_size=128))
