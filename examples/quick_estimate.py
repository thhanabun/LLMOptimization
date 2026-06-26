from llm_memlab import TransformerConfig, estimate_transformer_memory


cfg = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    intermediate_size=11008,
    num_attention_heads=32,
    vocab_size=32000,
    sequence_length=2048,
    training="lora",
    lora_rank=16,
    activation_checkpointing="selective",
)

print(estimate_transformer_memory(cfg).to_text())
