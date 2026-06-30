import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.html_report import (
    trace_interactive_to_html,
    trace_timeline_to_html,
    trace_to_html,
    write_interactive_html,
    write_timeline_html,
    write_trace_html,
)
from llm_memlab.torch_debugger import trace_forward


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HtmlReportTests(unittest.TestCase):
    def test_trace_to_html_contains_layer_table(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        _, trace = trace_forward(model, torch.randn(2, 4))
        html = trace_to_html(trace)
        self.assertIn("Layer table", html)
        self.assertIn("Linear", html)

    def test_write_trace_html(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4))
        _, trace = trace_forward(model, torch.randn(2, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "trace.html"
            write_trace_html(trace, path)
            self.assertTrue(path.exists())
            self.assertIn("llm-memlab trace", path.read_text(encoding="utf-8"))

    def test_timeline_html_contains_rows(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        _, trace = trace_forward(model, torch.randn(2, 4))
        html = trace_timeline_to_html(trace)
        self.assertIn("Sequential layer runtime timeline", html)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_timeline_html(trace, pathlib.Path(tmp) / "timeline.html")
            self.assertTrue(path.exists())

    def test_interactive_html_contains_filter_controls(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
        _, trace = trace_forward(model, torch.randn(2, 4))
        html = trace_interactive_to_html(trace)
        self.assertIn("id='filter'", html)
        self.assertIn("data-name", html)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_interactive_html(trace, pathlib.Path(tmp) / "interactive.html")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
