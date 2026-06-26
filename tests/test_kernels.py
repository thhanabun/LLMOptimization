import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.kernels import (
    apply_rope,
    chunked_cross_entropy,
    rms_norm,
    scaled_dot_product_attention,
    swiglu,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KernelTests(unittest.TestCase):
    def test_rms_norm_matches_reference(self):
        x = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        actual = rms_norm(x, weight, eps=1e-5)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_rope_preserves_shape(self):
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 6, 8)
        cos = torch.randn(6, 4)
        sin = torch.randn(6, 4)
        q_out, k_out = apply_rope(q, k, cos, sin)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_swiglu_matches_reference(self):
        x = torch.randn(2, 3, 8)
        gate = torch.randn(16, 8)
        up = torch.randn(16, 8)
        down = torch.randn(8, 16)
        actual = swiglu(x, gate, up, down)
        expected = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x, gate)) * torch.nn.functional.linear(x, up), down)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_chunked_cross_entropy_matches_reference(self):
        logits = torch.randn(2, 5, 11)
        targets = torch.randint(0, 11, (2, 5))
        actual = chunked_cross_entropy(logits, targets, chunk_size=3)
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_sdpa_preserves_shape(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(out.shape, q.shape)


if __name__ == "__main__":
    unittest.main()
