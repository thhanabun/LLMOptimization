import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.kv_quality import evaluate_attention_kv_quality, evaluate_int8_kv_quality, evaluate_kv_quantization_quality


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KVQualityTests(unittest.TestCase):
    def test_quality_metrics_are_reasonable(self):
        x = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        result = evaluate_int8_kv_quality(x)
        self.assertLess(result.mean_abs_error, 0.02)
        self.assertGreater(result.cosine_similarity, 0.999)
        self.assertGreater(result.compression_ratio, 1.0)
        self.assertIn("Cosine", result.to_text())

    def test_quality_supports_multiple_dtypes(self):
        x = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        for dtype_name in ("int8", "uint8", "fp16", "bf16", "fp32"):
            with self.subTest(dtype_name=dtype_name):
                result = evaluate_kv_quantization_quality(x, quant_dtype=dtype_name)
                self.assertEqual(result.quant_dtype, dtype_name)
                self.assertGreater(result.quantized_bytes, 0)

    def test_attention_quality_compares_sdpa_output(self):
        q = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        k = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        v = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        result = evaluate_attention_kv_quality(q, k, v, quant_dtype="int8")
        self.assertEqual(result.attention_shape, (1, 2, 1, 8))
        self.assertLess(result.output_mean_abs_error, 0.03)
        self.assertGreater(result.output_cosine_similarity, 0.999)
        self.assertIn("Output cosine", result.to_text())


if __name__ == "__main__":
    unittest.main()
