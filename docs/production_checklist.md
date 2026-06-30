# Production Checklist

Use this checklist before shipping an optimized path:

- Run CPU tests and optional CUDA tests.
- Run HF adapter matrix tests for each supported model family.
- Run `hf-cache-certify` for each production cache path. Put risky cache modes in `--experimental-caches` so they report drift without failing the production gate.
- Run `kernel-certify --quick` in normal CI and the full certification matrix on a GPU runner.
- Store JSON/CSV benchmark records with schema version, commit hash, GPU, torch version, triton version, dtype, sequence length, warmup, repeats, and seed.
- Compare benchmark records with a slowdown threshold and fail CI on regression.
- Run quality regression before accepting speedups.
- Keep `KernelPolicy` conservative. If the selector is unsure, it should fall back to torch/dequant+SDPA.
- Treat paged fused Triton kernels as experimental until the exact shape/dtype/GPU family has passed certification.
- Treat CuTile as `cutile-experimental`; on Ampere/older GPUs use Triton/Torch fallback unless a certified CuTile runtime is present.
- Keep real HF model smoke tests local-files-only and pinned by environment variable or local path.
- Export memory profiles for baseline and optimized paths when investigating deployment VRAM.
