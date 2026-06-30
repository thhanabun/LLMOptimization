import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.backends.cutile import certify_cutile_decode_attention, cutile_fused_decode_attention, detect_cutile_runtime


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CuTileBackendTests(unittest.TestCase):
    def test_detect_cutile_runtime_is_explainable(self):
        info = detect_cutile_runtime(device="cpu")
        self.assertIn(info.architecture, {"cpu", "unknown", "ampere", "hopper", "blackwell", "volta-turing", "pre-ampere"})
        self.assertTrue(info.reasons)

    def test_cutile_fused_decode_falls_back_to_torch(self):
        q = torch.randn(1, 4, 1, 8)
        k_pages = torch.randn(1, 2, 2, 4, 8)
        v_pages = torch.randn_like(k_pages)
        page_table = torch.tensor([[0, 1]])
        lengths = torch.tensor([8])
        result = cutile_fused_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=4)
        self.assertEqual(tuple(result.output.shape), (1, 4, 1, 8))
        self.assertIn(result.backend_used, {"torch-fallback", "cutile-experimental"})
        if result.backend_used == "torch-fallback":
            self.assertIn("fallback", result.fallback_reason)

    def test_cutile_certification_reports_not_production_on_fallback(self):
        q = torch.randn(1, 4, 1, 8)
        k_pages = torch.randn(1, 2, 2, 4, 8)
        v_pages = torch.randn_like(k_pages)
        page_table = torch.tensor([[0, 1]])
        lengths = torch.tensor([8])
        result = certify_cutile_decode_attention(q, k_pages, v_pages, page_table, lengths, page_size=4)
        self.assertFalse(result.passed)
        self.assertTrue(result.quality.passed)


if __name__ == "__main__":
    unittest.main()
