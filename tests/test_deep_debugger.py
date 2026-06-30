import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.deep_debugger import build_deep_debug_report, write_deep_debug_html


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DeepDebuggerTests(unittest.TestCase):
    def test_deep_debug_html_contains_drift_and_quality(self):
        baseline = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        candidate = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        candidate.load_state_dict(baseline.state_dict())
        with torch.no_grad():
            candidate[0].bias.add_(0.001)
        report = build_deep_debug_report(baseline, candidate, torch.randn(2, 4))
        html = report.to_html()
        self.assertIn("Layer drift", html)
        self.assertIsNotNone(report.quality)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_deep_debug_html(report, pathlib.Path(tmp) / "debug.html")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
