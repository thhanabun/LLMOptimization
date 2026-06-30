from llm_memlab import choose_memory_policy, estimate_transformer_memory, preset_config

estimate = estimate_transformer_memory(preset_config("7b-like", sequence_length=4096, dtype="fp16"))


class Info:
    kv_cache_bytes_fp16 = estimate.kv_cache_bytes


policy = choose_memory_policy(max_vram="8GB", model_info=Info(), sequence_length=4096)
print(policy.to_text())
