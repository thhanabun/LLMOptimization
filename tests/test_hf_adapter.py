import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.hf_adapter import MemoryFirstHFAdapter, MemoryFirstHFConfig, memory_first_generate
from llm_memlab.kv_cache import QuantizedStaticKVCache


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HFAdapterTests(unittest.TestCase):
    def test_memory_first_generate_uses_quantized_cache(self):
        class TinyHF(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vocab_size = 8

            def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
                batch, seq = input_ids.shape
                logits = torch.zeros(batch, seq, self.vocab_size)
                logits[:, -1, (input_ids[:, -1] + 1) % self.vocab_size] = 1.0
                length = seq if past_key_values is None else past_key_values[0][0].shape[-2] + seq
                key = torch.randn(batch, 2, length, 4)
                value = torch.randn(batch, 2, length, 4)
                return {"logits": logits, "past_key_values": ((key, value),)}

        result = memory_first_generate(TinyHF(), torch.tensor([[1, 2]]), MemoryFirstHFConfig(max_new_tokens=2, cache="quantized"))
        self.assertEqual(tuple(result.sequences.shape), (1, 4))
        self.assertIsInstance(result.cache, QuantizedStaticKVCache)
        self.assertEqual(result.steps, 2)


if __name__ == "__main__":
    unittest.main()
