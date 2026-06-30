# HF Adapter Limitations

The HF integration is conservative and family-aware, but production rollout still depends on model-family tests.

Supported local families:

- Llama-like: cache positions, attention masks, RoPE-oriented decoder behavior.
- Qwen-like: cache positions, attention masks, GQA/MQA-aware config detection.
- Mistral-like: cache positions, attention masks, grouped-query heads, sliding-window hint propagation.

Current limits:

- The adapter prefers Transformers `Cache`-style injection when the model accepts an external cache object.
- If a model rejects the external cache object, the wrapper falls back to the original `generate()` or to the custom memory-first loop depending on the entrypoint.
- Real production tests should pin a specific `transformers` version and run local-files-only smoke tests for each target model.
- Beam search, speculative decoding, tensor parallel cache sharding, and custom attention subclasses need per-family validation.
- Model downloads are not part of default tests. Set `LLM_MEMLAB_HF_SMOKE_MODEL` to a cached local model path/name to enable real HF smoke tests.

Qwen3 cache certification:

- Qwen3 uses a stricter Transformers `Cache` mask contract. `MemoryFirstTransformersCache.get_mask_sizes(query_length, layer_idx)` follows the current `(kv_length, kv_offset)` API.
- Qwen3 is handled by a dedicated `Qwen3MemoryAdapter`, separate from Qwen2-style behavior.
- Paged direct cache can be certified for multi-token prefill with exact generated-token agreement.
- Quantized direct cache remains experimental for Qwen3 until prefill-logit drift passes the configured threshold. Production policy falls back to paged direct cache for multi-token Qwen3 prefill.
- Default Qwen3 generation remains quality-gated unless direct cache is explicitly enabled and certified in the target environment.
- `hf-cache-certify` v2 records requested/effective cache, direct/fallback status, prompt length, token match, prefill logit drift, K/V drift, peak CUDA memory, and Transformers version.
