import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab import TransformerConfig, estimate_transformer_memory, preset_config


class EstimateTests(unittest.TestCase):
    def test_tiny_estimate_is_positive(self):
        cfg = preset_config("tiny", sequence_length=128, batch_size=2)
        estimate = estimate_transformer_memory(cfg)
        self.assertGreater(estimate.parameter_count, 0)
        self.assertGreater(estimate.peak_bytes, estimate.parameter_bytes)

    def test_lora_trains_far_fewer_parameters_than_full(self):
        base = TransformerConfig(
            num_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_attention_heads=8,
            vocab_size=4096,
            sequence_length=128,
            training="full",
        )
        full = estimate_transformer_memory(base)
        lora = estimate_transformer_memory(TransformerConfig(**{**base.__dict__, "training": "lora", "lora_rank": 8}))
        self.assertLess(lora.trainable_parameter_count, full.trainable_parameter_count)
        self.assertLess(lora.optimizer_bytes, full.optimizer_bytes)


if __name__ == "__main__":
    unittest.main()
