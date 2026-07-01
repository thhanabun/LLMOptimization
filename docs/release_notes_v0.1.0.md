# llm-memlab v0.1.0 Release Notes

`llm-memlab` v0.1.0 is the first public baseline for a memory-first LLM debugging and optimization library.

## Highlights

- Memory estimator, tensor lifetime IR, and buffer planner.
- PyTorch and Hugging Face model inspection/debug reports.
- KV cache experiments, quality metrics, benchmark database, and HTML dashboards.
- Conservative HF memory-first generate path with fallback behavior.
- Production API surface via `llm_memlab.production`.
- Backend registry with Torch, CUDA, Triton, CuTile, optional plugin backends, and `vllm-serving`.
- Serving benchmark that compares HF generate, llm-memlab memory-first HF, and optional vLLM.
- Release docs for production status, limitations, CI integration, and notebook/cloud usage.

## Stable Today

- Analysis, estimation, tracing, reporting, benchmark storage, quality gates, and conservative HF fallback workflows.
- Backend and hardware detection with explicit fallback reasons.
- CI-friendly commands that skip unsupported GPU/runtime paths.

## Experimental

- Triton fused/paged decode kernels.
- Quantized direct cache paths.
- CuTile runtime integration.
- Direct vLLM execution path; detection/fallback is stable, serving execution must be certified on the target environment.

## Validation Used For This Release

- `ruff check src tests examples`
- `python -m unittest discover -s tests`
- selected `mypy` checks for new backend/serving modules
- `python -m build`
- wheel install smoke
- TinyLlama local serving smoke on CPU with vLLM fallback reporting

## Honest Scope

This release is not intended to replace PyTorch, Transformers, vLLM, or production inference fleets. It is a memory-first lab for seeing what your model is doing, certifying optimized paths, and making safer backend decisions.
