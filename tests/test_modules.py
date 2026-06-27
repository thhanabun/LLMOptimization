import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OptimizedModuleTests(unittest.TestCase):
    def test_decoder_block_forward_backward(self):
        from llm_memlab.modules import OptimizedDecoderBlock, build_rope_cache

        block = OptimizedDecoderBlock(hidden_size=16, intermediate_size=32, num_heads=4)
        x = torch.randn(2, 5, 16, requires_grad=True)
        cos, sin = build_rope_cache(seq_len=5, head_dim=4, dtype=x.dtype)
        y = block(x, cos=cos, sin=sin)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        self.assertIsNotNone(x.grad)

    def test_rope_cache_shape(self):
        from llm_memlab.modules import build_rope_cache

        cos, sin = build_rope_cache(seq_len=7, head_dim=8)
        self.assertEqual(cos.shape, (7, 8))
        self.assertEqual(sin.shape, (7, 8))


if __name__ == "__main__":
    unittest.main()
