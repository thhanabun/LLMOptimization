# Stable vs Experimental

`llm_memlab.production` is the stable import surface. Use it for CI, application integration, benchmark gates, quality gates, memory profiling, kernel policy selection, and conservative HF cache adapters.

Stable APIs:

- Benchmark DB and regression gate: `BenchmarkRecord`, `BenchmarkGateConfig`, `benchmark_gate`, `assert_no_regressions`.
- Quality safety rail: `QualityThresholds`, `run_quality_regression`, `assert_quality_regression`.
- Conservative kernel selection: `KernelPolicy`, `select_kernel_policy`.
- Memory observability: `profile_decode_memory`, `write_memory_profile_json`, `write_memory_profile_html`.
- HF family adapter entrypoints: `select_memory_adapter`, `memory_first_generate_hf`, `install_memory_first_generate`.

`llm_memlab.experimental` is opt-in. It exposes fused and paged Triton kernels plus CuTile backend contracts that need per-shape certification before production use.

Experimental APIs:

- Dense fused int8/uint8 decode kernels.
- Paged fused int8/uint8 decode kernels.
- CuTile detection and backend `cutile-experimental`.
- `experimental_kernel_policy()` and backend `triton-experimental`.

Production rule: start from `llm_memlab.production`; enable `llm_memlab.experimental` only for shapes that pass kernel certification on your deployment GPU class.
