# Production Integration Guide

`llm-memlab` is designed to be memory-first and conservative in production. The stable path measures quality, memory, and latency first, then promotes optimized backends only when certification data exists for the model family and hardware.

## Recommended Stable Path

1. Import stable APIs from `llm_memlab.production`.
2. Run `certify-model-matrix` against local representative models and commit the resulting `profiles.json`.
3. Use `profile explain` in debugging and CI logs so users can see why a cache backend was selected.
4. Run benchmark gates from JSON/CSV history before merging optimization changes.
5. Keep Triton/CuTile fused kernels behind `triton-experimental` or `cutile-experimental` until promotion gates pass on the target GPU class.

## Hugging Face Generate Integration

Family adapters prepare `past_key_values`, `attention_mask`, and `cache_position` only when the model's `generate()` signature accepts them. This keeps integration compatible across multiple Transformers versions and lets unsupported models fall back to the original generate path.

Adapters currently cover Llama, Qwen/Qwen3, Mistral/Mixtral, Gemma, Phi, DeepSeek, GPT-NeoX, and Falcon-style configs. For a new family, implement `MemoryAdapterProtocol`, register the adapter, and add a tiny-model contract test before using it in production.

## Certification Profiles

Profiles are data, not hardcoded policy. A profile records model family, model name, Transformers version, Torch version, GPU architecture, safe prompt length, quant dtype, and certified backend. Production policy only enables quantized direct cache when the matching profile is production-certified.

Useful commands:

```bash
llm-memlab profile export --out profiles.default.json
llm-memlab profile merge --inputs profiles.default.json profiles.local.json --out profiles.json
llm-memlab profile explain --family qwen3 --model Qwen3 --prompt-tokens 256 --profile profiles.json
```

## Kernel Promotion

Kernel certification checks correctness and performance across shapes, dtypes, GQA/MQA, page size, and long context. Promotion is intentionally strict:

- Triton can be promoted only after int8 and uint8 cases pass with fp16 or bf16 compute.
- CuTile remains experimental unless hardware is Hopper or Blackwell and certification passes.
- Long-context production requires sequence length `>=4096` in the certification report.

## CI Pattern

```bash
python -m unittest discover -s tests
llm-memlab certify-model-matrix --local-root ./models --profiles-out profiles.json --json-out matrix.json --fail-on-regression
llm-memlab benchmark-compare --baseline baseline.json --candidate candidate.json --fail-on-regression
llm-memlab benchmark-dashboard --inputs baseline.json candidate.json --out dashboard.html
```

Experimental paths should report results, not fail production CI, until they have enough coverage to promote.

