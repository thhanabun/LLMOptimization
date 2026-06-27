import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.torch_debugger import TorchTrace, trace_forward


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TorchDebuggerTests(unittest.TestCase):
    def test_trace_forward_collects_layer_stats(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        output, trace = trace_forward(model, torch.randn(3, 4))
        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(len(trace.records), 3)
        self.assertGreater(trace.records[0].input_bytes, 0)
        self.assertIsNotNone(trace.records[0].output_stats)
        self.assertIn("Hot layers", trace.to_text(show_shapes=True, show_stats=True))
        self.assertIn("records", trace.to_json())

    def test_gradient_monitor_collects_stats(self):
        model = torch.nn.Linear(4, 2)
        trace = TorchTrace(model)
        trace.attach_gradient_monitor()
        with trace:
            loss = model(torch.randn(3, 4)).square().mean()
            loss.backward()
        self.assertGreater(len(trace.gradients), 0)
        self.assertGreaterEqual(trace.gradients[0].max_abs, 0)


if __name__ == "__main__":
    unittest.main()

