import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.attention_debugger import analyze_qk_attention, attention_stats_to_text, collect_attention_stats


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AttentionDebuggerTests(unittest.TestCase):
    def test_analyze_qk_attention(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.q_proj = torch.nn.Linear(8, 8, bias=False)
                self.k_proj = torch.nn.Linear(8, 8, bias=False)

        stat = analyze_qk_attention(TinyAttention(), torch.randn(1, 4, 8), name="attn")
        self.assertIsNotNone(stat)
        self.assertEqual(stat.name, "attn")
        self.assertIn("Entropy", stat.to_text())

    def test_collect_attention_stats(self):
        class TinyAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.q_proj = torch.nn.Linear(8, 8, bias=False)
                self.k_proj = torch.nn.Linear(8, 8, bias=False)
                self.v_proj = torch.nn.Linear(8, 8, bias=False)
                self.o_proj = torch.nn.Linear(8, 8, bias=False)

            def forward(self, x):
                return self.o_proj(self.v_proj(x))

        model = torch.nn.Sequential(TinyAttention())
        output, stats = collect_attention_stats(model, torch.randn(1, 4, 8))
        self.assertEqual(output.shape, (1, 4, 8))
        self.assertEqual(len(stats), 1)
        self.assertIn("Dead heads", attention_stats_to_text(stats))


if __name__ == "__main__":
    unittest.main()
