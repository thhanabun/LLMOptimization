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
    linear_cross_entropy,
    qkv_rope_attention,
    qkv_rope_attention_cached,
    quantized_kv_attention,
    rms_norm,
    rms_norm_manual_backward,
    scaled_dot_product_attention,
    swiglu,
    triton_apply_rope,
    triton_rms_norm,
    triton_dequantize_int8_per_token,
    triton_quantize_int8_per_token,
    triton_swiglu_activation,
)
from llm_memlab.kv_cache import KVCacheConfig, StaticKVCache


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KernelTests(unittest.TestCase):
    def test_rms_norm_matches_reference(self):
        x = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        actual = rms_norm(x, weight, eps=1e-5)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_triton_rms_norm_falls_back_on_cpu(self):
        x = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        actual = triton_rms_norm(x, weight, eps=1e-5)
        expected = rms_norm(x, weight, eps=1e-5)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_manual_rms_norm_backward_matches_reference(self):
        x = torch.randn(2, 3, 8, requires_grad=True)
        weight = torch.randn(8, requires_grad=True)
        bias = torch.randn(8, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()
        ref_bias = bias.detach().clone().requires_grad_()

        actual = rms_norm_manual_backward(x, weight, eps=1e-5, bias=bias).sum()
        expected = rms_norm(ref_x, ref_weight, eps=1e-5, bias=ref_bias).sum()
        actual.backward()
        expected.backward()

        self.assertTrue(torch.allclose(x.grad, ref_x.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(weight.grad, ref_weight.grad, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(bias.grad, ref_bias.grad, atol=1e-6, rtol=1e-6))

    def test_rope_preserves_shape(self):
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 6, 8)
        cos = torch.randn(6, 4)
        sin = torch.randn(6, 4)
        q_out, k_out = apply_rope(q, k, cos, sin)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(k_out.shape, k.shape)

    def test_triton_rope_falls_back_on_cpu(self):
        q = torch.randn(2, 4, 6, 8)
        k = torch.randn(2, 4, 6, 8)
        cos = torch.randn(6, 4)
        sin = torch.randn(6, 4)
        actual_q, actual_k = triton_apply_rope(q, k, cos, sin)
        expected_q, expected_k = apply_rope(q, k, cos, sin)
        self.assertTrue(torch.allclose(actual_q, expected_q))
        self.assertTrue(torch.allclose(actual_k, expected_k))

    def test_swiglu_matches_reference(self):
        x = torch.randn(2, 3, 8)
        gate = torch.randn(16, 8)
        up = torch.randn(16, 8)
        down = torch.randn(8, 16)
        actual = swiglu(x, gate, up, down)
        expected = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, gate)) * torch.nn.functional.linear(x, up), down
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_triton_swiglu_activation_falls_back_on_cpu(self):
        gate = torch.randn(2, 3, 8)
        up = torch.randn(2, 3, 8)
        actual = triton_swiglu_activation(gate, up)
        expected = torch.nn.functional.silu(gate) * up
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_triton_quantize_dequantize_falls_back_on_cpu(self):
        x = torch.randn(2, 3, 4, 8)
        q, scale = triton_quantize_int8_per_token(x)
        y = triton_dequantize_int8_per_token(q, scale, dtype=torch.float32)
        self.assertEqual(q.dtype, torch.int8)
        self.assertLess((x - y).abs().mean().item(), 0.02)

    def test_chunked_cross_entropy_matches_reference(self):
        logits = torch.randn(2, 5, 11)
        targets = torch.randint(0, 11, (2, 5))
        actual = chunked_cross_entropy(logits, targets, chunk_size=3)
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), targets.reshape(-1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_linear_cross_entropy_matches_reference_and_grads(self):
        hidden = torch.randn(2, 5, 7, requires_grad=True)
        weight = torch.randn(13, 7, requires_grad=True)
        bias = torch.randn(13, requires_grad=True)
        targets = torch.randint(0, 13, (2, 5))
        ref_hidden = hidden.detach().clone().requires_grad_()
        ref_weight = weight.detach().clone().requires_grad_()
        ref_bias = bias.detach().clone().requires_grad_()

        actual = linear_cross_entropy(hidden, weight, targets, bias=bias, chunk_size=3)
        logits = torch.nn.functional.linear(ref_hidden, ref_weight, ref_bias)
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 13), targets.reshape(-1))
        actual.backward()
        expected.backward()

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(hidden.grad, ref_hidden.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(weight.grad, ref_weight.grad, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(bias.grad, ref_bias.grad, atol=1e-6, rtol=1e-6))

    def test_qkv_rope_attention_preserves_shape(self):
        x = torch.randn(2, 6, 16)
        qkv = torch.randn(48, 16)
        out = torch.randn(16, 16)
        cos = torch.randn(6, 8)
        sin = torch.randn(6, 8)
        y = qkv_rope_attention(x, qkv, out, cos=cos, sin=sin, num_heads=2)
        self.assertEqual(y.shape, x.shape)

    def test_qkv_rope_attention_cached_updates_cache(self):
        cache = StaticKVCache(KVCacheConfig(num_layers=1, batch_size=2, num_heads=2, head_dim=8, max_seq_len=8, dtype=torch.float32))
        qkv = torch.randn(48, 16)
        out = torch.randn(16, 16)
        x0 = torch.randn(2, 1, 16)
        y0 = qkv_rope_attention_cached(x0, qkv, out, kv_cache=cache, layer_idx=0, cache_position=0, num_heads=2)
        self.assertEqual(y0.shape, x0.shape)
        self.assertEqual(cache.length, 1)
        x1 = torch.randn(2, 1, 16)
        y1 = qkv_rope_attention_cached(x1, qkv, out, kv_cache=cache, layer_idx=0, cache_position=1, num_heads=2)
        self.assertEqual(y1.shape, x1.shape)
        self.assertEqual(cache.length, 2)


    def test_quantized_kv_attention_preserves_shape(self):
        q = torch.randn(1, 2, 1, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = quantized_kv_attention(q, k, v, quant_dtype="int8", backend="auto")
        self.assertEqual(out.shape, q.shape)
        with self.assertRaises(ValueError):
            quantized_kv_attention(q, k, v, backend="bad")
    def test_sdpa_preserves_shape(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(out.shape, q.shape)
        with self.assertRaises(ValueError):
            quantized_kv_attention(q, k, v, backend="bad")


if __name__ == "__main__":
    unittest.main()



