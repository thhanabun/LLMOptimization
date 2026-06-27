import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.patchers import PackedQKVAttentionAdapter, optimize_hf_model, optimize_llama_qwen_attention


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PatcherTests(unittest.TestCase):
    def test_patch_rmsnorm_and_swiglu_mlp(self):
        class TinyRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(8))
                self.variance_epsilon = 1e-6

            def forward(self, x):
                return x * self.weight

        class TinyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(8, 16, bias=False)
                self.up_proj = torch.nn.Linear(8, 16, bias=False)
                self.down_proj = torch.nn.Linear(16, 8, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = TinyRMSNorm()
                self.mlp = TinyMLP()

            def forward(self, x):
                return self.mlp(self.norm(x))

        model = TinyModel()
        _, report = optimize_hf_model(model)
        self.assertEqual(report.patched_norms, 1)
        self.assertEqual(report.patched_mlps, 1)
        self.assertEqual(model(torch.randn(2, 3, 8)).shape, (2, 3, 8))

    def test_attention_patch_is_opt_in_and_forwardable(self):
        class TinyLlamaAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.head_dim = 4
                self.q_proj = torch.nn.Linear(8, 8, bias=False)
                self.k_proj = torch.nn.Linear(8, 8, bias=False)
                self.v_proj = torch.nn.Linear(8, 8, bias=False)
                self.o_proj = torch.nn.Linear(8, 8, bias=False)

            def forward(self, x):
                return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = TinyLlamaAttention()

            def forward(self, x):
                return self.self_attn(x)

        model = TinyModel()
        _, report = optimize_hf_model(model, patch_attention=True)
        self.assertEqual(report.patched_attentions, 1)
        self.assertIsInstance(model.self_attn, PackedQKVAttentionAdapter)
        self.assertEqual(model(torch.randn(2, 3, 8)).shape, (2, 3, 8))

    def test_attention_helper_only_patches_attention(self):
        class TinyQwenAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2
                self.q_proj = torch.nn.Linear(8, 8, bias=False)
                self.k_proj = torch.nn.Linear(8, 8, bias=False)
                self.v_proj = torch.nn.Linear(8, 8, bias=False)
                self.o_proj = torch.nn.Linear(8, 8, bias=False)

        model = torch.nn.Sequential(TinyQwenAttention())
        _, report = optimize_llama_qwen_attention(model)
        self.assertEqual(report.patched_attentions, 1)

    def test_dry_run_does_not_replace(self):
        class TinyRMSNorm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4))
                self.eps = 1e-6

        model = torch.nn.Sequential(TinyRMSNorm())
        original_type = type(model[0])
        _, report = optimize_hf_model(model, dry_run=True)
        self.assertEqual(report.patched_norms, 1)
        self.assertIs(type(model[0]), original_type)


if __name__ == "__main__":
    unittest.main()
