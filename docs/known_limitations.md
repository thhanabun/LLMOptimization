# Known Limitations

`llm-memlab` is production-oriented, but not every backend is production-ready today. The project intentionally labels risky paths as experimental until certification data is broad enough.

## Not Production Yet

- Triton paged fused decode is still experimental unless a full kernel certification matrix passes for the target GPU.
- CuTile native decode is a contract/fallback path unless CuTile runtime and Hopper/Blackwell-class certification are available.
- Quantized direct HF cache is disabled by default unless a model-family profile certifies it.
- Mamba/SSM architecture support is not implemented yet. It is a future experimental architecture path, not a current production feature.
- Real model certification does not download models by default. CI should provide local cached models and use `--require-real-models` or `--strict`.

## Common Fallback Reasons

- Missing local model fixtures.
- Prompt length exceeds a certified quantized-cache profile.
- GPU architecture does not match the certification profile.
- Triton or CUDA is unavailable.
- HF `generate()` rejects external cache objects or runtime cache/device/shape checks fail.

## Kernel Promotion Requirements

`kernel-promote` should only produce a production decision after coverage includes:

- batch variation,
- Q heads and KV heads,
- GQA and MQA shapes,
- head dimension variation,
- sequence length variation,
- page size variation,
- int8 and uint8 quantized K/V,
- fp16 and bf16 compute,
- and long-context coverage.

Quick smoke runs are useful for developer feedback, but they should not be treated as production certification.

