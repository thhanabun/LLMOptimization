import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.kv_quality import evaluate_int8_kv_quality


@unittest.skipIf(torch is None, "PyTorch is not installed")
class KVQualityTests(unittest.TestCase):
    def test_quality_metrics_are_reasonable(self):
        x = torch.randn(1, 2, 4, 8, dtype=torch.float16)
        result = evaluate_int8_kv_quality(x)
        self.assertLess(result.mean_abs_error, 0.02)
        self.assertGreater(result.cosine_similarity, 0.999)
        self.assertGreater(result.compression_ratio, 1.0)
        self.assertIn("Cosine", result.to_text())


if __name__ == "__main__":
    unittest.main()
