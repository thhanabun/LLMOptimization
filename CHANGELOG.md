# Changelog

## 0.1.1 - Unreleased

Documentation, notebook, and portability polish for the first patch release.

### Added

- Real Colab and Kaggle notebook starters under `notebooks/`.
- `examples/cloud_smoke.py`, a portable platform smoke script for backend detection, memory estimates, optional HF model benchmarks, serving benchmark export, and dashboard generation.
- Docs for certifying local models, comparing against vLLM, and the stable public API surface.

### Changed

- README release status now targets `v0.1.1`.
- Cloud notebook docs now point to runnable notebooks and the shared cloud smoke script.
- Examples prefer `LLM_MEMLAB_MODEL` and `LLM_MEMLAB_MODEL_ROOT` instead of hardcoded local paths.

## 0.1.0 - 2026-07-01

Initial public release of `llm-memlab`, a memory-first LLM analysis, debugging, certification, and optimization lab.

### Stable

- Transformer memory estimator for inference, LoRA, and full fine-tuning.
- Tensor lifetime IR and buffer planner for peak-memory visibility.
- PyTorch module tracing, interactive HTML reports, and layer/debugger utilities.
- Benchmark database with JSON/CSV records, metadata, comparison gates, and HTML dashboard.
- Quality regression metrics for logits, top-k overlap, token match, and drift checks.
- OOM-aware runner and CUDA memory profiler for observability workflows.
- Hugging Face adapter contract and conservative family adapters for Llama, Qwen/Qwen3, Mistral/Mixtral, Gemma, Phi, DeepSeek, GPT-NeoX, and Falcon style models.
- Production API surface in `llm_memlab.production`.
- Local model harness and certification matrix scaffolding for real-model cache profiles.
- `vllm-serving` backend detector and serving benchmark comparison against HF generate and llm-memlab memory-first HF.

### Experimental

- Triton fused decode and paged KV paths.
- Quantized KV cache direct path for selected/certified model-family profiles.
- CuTile backend contract and fallback dispatcher.
- vLLM serving path execution; detection and fallback are stable, direct vLLM execution depends on a compatible Linux/WSL/Docker CUDA environment.

### Known Limitations

- This release is not a PyTorch replacement or a serving engine.
- Qwen3/Gemma4 full certification is not claimed unless users run local certification on their target hardware.
- Native Windows is supported for development, analysis, and HF fallback workflows; production vLLM serving should be validated on Linux/WSL/Docker.
- Experimental kernels require shape, dtype, long-context, quality, and hardware certification before production promotion.
