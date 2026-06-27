import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.inspector import inspect_model


@unittest.skipIf(torch is None, "PyTorch is not installed")
class InspectorTests(unittest.TestCase):
    def test_inspect_tiny_model(self):
        class Config:
            model_type = "tiny"
            num_hidden_layers = 2
            hidden_size = 8
            intermediate_size = 16
            num_attention_heads = 2
            num_key_value_heads = 2
            vocab_size = 32
            max_position_embeddings = 64

        class TinyRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8))
                self.eps = 1e-6

        class TinyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(8, 16, bias=False)
                self.up_proj = torch.nn.Linear(8, 16, bias=False)
                self.down_proj = torch.nn.Linear(16, 8, bias=False)

        class TinyModel(torch.nn.Module):
            config = Config()

            def __init__(self):
                super().__init__()
                self.norm = TinyRMSNorm()
                self.mlp = TinyMLP()

        info = inspect_model(TinyModel())
        self.assertEqual(info.model_type, "tiny")
        self.assertEqual(info.num_layers, 2)
        self.assertEqual(info.patchable_mlps, 1)
        self.assertGreater(info.kv_cache_bytes_fp16, 0)
        self.assertIn("KV cache", info.to_text())


if __name__ == "__main__":
    unittest.main()
