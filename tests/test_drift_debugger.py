import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.drift_debugger import compare_layer_drift


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DriftDebuggerTests(unittest.TestCase):
    def test_compare_layer_drift(self):
        baseline = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).eval()
        candidate = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU()).eval()
        candidate.load_state_dict(baseline.state_dict())
        with torch.no_grad():
            candidate[0].bias.add_(0.01)
        report = compare_layer_drift(baseline, candidate, torch.randn(2, 4))
        self.assertGreater(len(report.records), 0)
        self.assertIsNotNone(report.worst)
        self.assertIn("Worst drift", report.to_text())


if __name__ == "__main__":
    unittest.main()
