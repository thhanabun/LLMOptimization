import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.benchmark import BenchmarkConfig, benchmark_callable, compare_benchmarks


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkTests(unittest.TestCase):
    def test_benchmark_callable(self):
        x = torch.randn(2, 4)
        layer = torch.nn.Linear(4, 3)
        result = benchmark_callable("linear", lambda: layer(x), BenchmarkConfig(warmup=1, repeats=2))
        self.assertEqual(len(result.elapsed_ms), 2)
        self.assertEqual(result.output_shape, "(2, 3)")
        self.assertIn("Mean", result.to_text())

    def test_compare_benchmarks(self):
        x = torch.randn(2, 4)
        layer = torch.nn.Linear(4, 3)
        result = benchmark_callable("linear", lambda: layer(x), BenchmarkConfig(warmup=0, repeats=1))
        table = compare_benchmarks([result])
        self.assertIn("linear", table)


if __name__ == "__main__":
    unittest.main()
