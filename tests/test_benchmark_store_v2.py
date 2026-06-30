import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.benchmark import BenchmarkResult
from llm_memlab.benchmark_store import (
    BenchmarkGateConfig,
    assert_no_regressions,
    benchmark_gate,
    collect_run_metadata,
    compare_benchmark_records,
    read_benchmark_csv,
    record_from_benchmark,
    write_benchmark_csv,
    write_benchmark_json,
)


class BenchmarkStoreV2Tests(unittest.TestCase):
    def test_metadata_csv_roundtrip_and_regression(self):
        metadata = collect_run_metadata(dtype="fp16", sequence_length=128)
        base = record_from_benchmark(BenchmarkResult("decode", [10.0]), kind="decode", metadata=metadata)
        candidate = record_from_benchmark(BenchmarkResult("decode", [11.0]), kind="decode", metadata=metadata)
        comparison = compare_benchmark_records(base, candidate, max_slowdown_pct=10.0)
        self.assertTrue(comparison.passed)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_benchmark_csv([base], pathlib.Path(tmp) / "bench.csv")
            loaded = read_benchmark_csv(path)
            self.assertEqual(loaded[0].metadata["dtype"], "fp16")
        assert_no_regressions([comparison])
        failed = compare_benchmark_records(base, candidate, max_slowdown_pct=5.0)
        with self.assertRaises(AssertionError):
            assert_no_regressions([failed])

    def test_benchmark_compare_cli_reports_regression(self):
        metadata = collect_run_metadata(dtype="fp16", sequence_length=32)
        base = record_from_benchmark(BenchmarkResult("decode", [10.0]), kind="decode", metadata=metadata)
        candidate = record_from_benchmark(BenchmarkResult("decode", [12.0]), kind="decode", metadata=metadata)
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(__file__).resolve().parents[1]
            base_path = write_benchmark_json([base], pathlib.Path(tmp) / "baseline.json")
            cand_path = write_benchmark_json([candidate], pathlib.Path(tmp) / "candidate.json")
            env = dict(__import__("os").environ)
            env["PYTHONPATH"] = str(root / "src")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llm_memlab",
                    "benchmark-compare",
                    "--baseline",
                    str(base_path),
                    "--candidate",
                    str(cand_path),
                    "--max-slowdown-pct",
                    "10",
                    "--fail-on-regression",
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("FAIL", result.stdout)

    def test_benchmark_gate_fails_on_quality_regression(self):
        base = record_from_benchmark(BenchmarkResult("decode", [10.0]), kind="decode")
        candidate = record_from_benchmark(BenchmarkResult("decode", [9.0]), kind="decode")
        candidate.extra["quality_passed"] = False
        result = benchmark_gate([base], [candidate], BenchmarkGateConfig(require_quality_passed=True))
        self.assertFalse(result.passed)
        with self.assertRaises(AssertionError):
            assert_no_regressions(result)


if __name__ == "__main__":
    unittest.main()
