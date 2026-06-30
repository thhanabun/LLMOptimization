import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.kv_cache import (
    DecodeConfig,
    KVCacheConfig,
    PagedKVCache,
    QuantizedStaticKVCache,
    StaticKVCache,
    dequantize_int8_per_token,
    dequantize_uint8_per_token,
    greedy_decode,
    quantize_int8_per_token,
    quantize_uint8_per_token,
    sample_next_token,
)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KVCacheTests(unittest.TestCase):
    def test_static_cache_append_and_stats(self):
        cache = StaticKVCache(KVCacheConfig(num_layers=2, batch_size=1, num_heads=3, head_dim=4, max_seq_len=8, dtype=torch.float32))
        key = torch.randn(1, 3, 2, 4)
        value = torch.randn(1, 3, 2, 4)
        cached_key, cached_value = cache.append_layer(0, key, value)
        self.assertEqual(cached_key.shape, (1, 3, 2, 4))
        self.assertEqual(cached_value.shape, (1, 3, 2, 4))
        self.assertEqual(cache.length, 2)
        self.assertGreater(cache.stats().bytes_allocated, cache.stats().bytes_used)

    def test_quantize_dequantize_roundtrip(self):
        x = torch.randn(2, 3, 4, 8)
        q, scale = quantize_int8_per_token(x)
        y = dequantize_int8_per_token(q, scale, dtype=torch.float32)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(scale.shape, (2, 3, 4, 1))
        self.assertLess((x - y).abs().mean().item(), 0.02)

    def test_uint8_quantize_dequantize_roundtrip(self):
        x = torch.randn(2, 3, 4, 8)
        q, scale, zero_point = quantize_uint8_per_token(x)
        y = dequantize_uint8_per_token(q, scale, zero_point, dtype=torch.float32)
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(zero_point.dtype, torch.uint8)
        self.assertLess((x - y).abs().mean().item(), 0.02)

    def test_quantized_cache_append_dequantizes_and_compresses(self):
        cfg = KVCacheConfig(num_layers=2, batch_size=1, num_heads=3, head_dim=16, max_seq_len=8, dtype=torch.float32)
        cache = QuantizedStaticKVCache(cfg)
        fp_cache = StaticKVCache(cfg)
        key = torch.randn(1, 3, 2, 16)
        value = torch.randn(1, 3, 2, 16)
        q_key, q_value = cache.append_layer(0, key, value)
        fp_cache.append_layer(0, key, value)
        self.assertEqual(q_key.shape, key.shape)
        self.assertEqual(q_value.shape, value.shape)
        self.assertEqual(cache.length, 2)
        self.assertLess(cache.nbytes, fp_cache.nbytes)
        self.assertGreater(cache.stats().compression_ratio, 1.0)
        self.assertLess((q_key - key).abs().mean().item(), 0.02)

    def test_quantized_cache_supports_common_storage_dtypes(self):
        cfg = KVCacheConfig(num_layers=1, batch_size=1, num_heads=2, head_dim=8, max_seq_len=4, dtype=torch.float32)
        key = torch.randn(1, 2, 2, 8)
        value = torch.randn(1, 2, 2, 8)
        for dtype_name in ("int8", "uint8", "fp16", "bf16", "fp32"):
            with self.subTest(dtype_name=dtype_name):
                cache = QuantizedStaticKVCache(cfg, quant_dtype=dtype_name)
                cached_key, cached_value = cache.append_layer(0, key, value)
                self.assertEqual(cached_key.shape, key.shape)
                self.assertEqual(cached_value.dtype, torch.float32)
                self.assertIn(dtype_name, cache.stats().to_text())

    def test_paged_cache_append_and_read(self):
        cfg = KVCacheConfig(num_layers=1, batch_size=1, num_heads=2, head_dim=4, max_seq_len=5, dtype=torch.float32)
        cache = PagedKVCache(cfg, page_size=2)
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 3, 4)
        cached_key, cached_value = cache.append_layer(0, key, value)
        self.assertEqual(cached_key.shape, key.shape)
        self.assertTrue(torch.allclose(cached_key, key))
        self.assertTrue(torch.allclose(cached_value, value))
        self.assertEqual(cache.allocated_pages, 2)
        self.assertIn("paged", cache.stats().storage_dtype)
        self.assertGreater(cache.free_pages, 0)
        self.assertIn("Fragmentation", cache.fragmentation_report())
        freed = cache.release_pages(1)
        self.assertEqual(freed, 1)
        self.assertEqual(cache.allocated_pages, 1)

    def test_sample_next_token_greedy(self):
        logits = torch.tensor([[0.1, 2.0, 0.3]])
        token = sample_next_token(logits)
        self.assertEqual(int(token.item()), 1)

    def test_greedy_decode_dict_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
                self.calls += 1
                vocab = 8
                logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
                next_token = (input_ids[:, -1] + 1) % vocab
                logits[:, -1, :].scatter_(1, next_token[:, None], 1.0)
                past = ((torch.zeros(1, 1, self.calls, 1), torch.zeros(1, 1, self.calls, 1)),)
                return {"logits": logits, "past_key_values": past}

        prompt = torch.tensor([[1, 2]])
        result = greedy_decode(TinyModel(), prompt, DecodeConfig(max_new_tokens=3))
        self.assertEqual(result.sequences.tolist(), [[1, 2, 3, 4, 5]])
        self.assertEqual(len(result.steps), 3)
        self.assertIn("Throughput", result.to_text())


if __name__ == "__main__":
    unittest.main()
