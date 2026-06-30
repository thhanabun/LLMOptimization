import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.benchmark import BenchmarkResult
from llm_memlab.compare_report import CompareReport, compare_report_to_html, scoreboard_to_html, write_compare_html, write_scoreboard_html


class CompareReportTests(unittest.TestCase):
    def test_compare_report_html_contains_sections(self):
        report = CompareReport(
            title="demo",
            benchmarks=[BenchmarkResult(name="baseline", elapsed_ms=[1.0], output_shape="(1,)")],
        )
        html = compare_report_to_html(report)
        self.assertIn("Benchmark", html)
        self.assertIn("baseline", html)

    def test_write_compare_html(self):
        report = CompareReport(title="demo")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_compare_html(report, pathlib.Path(tmp) / "compare.html")
            self.assertTrue(path.exists())
            self.assertIn("demo", path.read_text(encoding="utf-8"))

    def test_scoreboard_html(self):
        rows = [{"model": "tiny", "status": "ok", "baseline_ms": 2.0, "optimized_ms": 1.0, "speedup": 2.0, "patched": 3}]
        html = scoreboard_to_html(rows)
        self.assertIn("tiny", html)
        self.assertIn("2.00x", html)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_scoreboard_html(rows, pathlib.Path(tmp) / "scoreboard.html")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
