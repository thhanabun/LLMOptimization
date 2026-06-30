from llm_memlab import benchmark_fused_decode_attention


if __name__ == "__main__":
    result = benchmark_fused_decode_attention(q_heads=4, kv_heads=2, tokens=64, head_dim=32, repeats=5)
    print(result.to_text())
