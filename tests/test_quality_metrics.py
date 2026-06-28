import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.quality_metrics import compare_logits, compare_token_sequences


@unittest.skipIf(torch is None, "PyTorch is not installed")
class QualityMetricsTests(unittest.TestCase):
    def test_compare_logits_passes_for_small_drift(self):
        torch.manual_seed(0)
        baseline = torch.randn(2, 3, 11)
        candidate = baseline + torch.randn_like(baseline) * 0.001
        result = compare_logits(baseline, candidate, top_k=3)
        self.assertTrue(result.passed)
        self.assertGreater(result.cosine_similarity, 0.999)
        self.assertEqual(result.top1_agreement, 1.0)

    def test_compare_logits_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            compare_logits(torch.randn(1, 2), torch.randn(1, 3))

    def test_compare_logits_rejects_invalid_top_k(self):
        with self.assertRaises(ValueError):
            compare_logits(torch.randn(1, 2, 3), torch.randn(1, 2, 3), top_k=0)

    def test_compare_token_sequences_reports_agreement(self):
        result = compare_token_sequences([1, 2, 3, 4], [1, 2, 5, 4])
        self.assertFalse(result.exact_match)
        self.assertEqual(result.token_agreement, 0.75)


if __name__ == "__main__":
    unittest.main()
