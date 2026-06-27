import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.memory_policy import choose_memory_policy
from llm_memlab.optimization_report import OptimizationReport, infer_findings
from llm_memlab.benchmark import BenchmarkResult


class PolicyAndOptimizationReportTests(unittest.TestCase):
    def test_choose_memory_policy_recommends_quantized_cache(self):
        class Info:
            kv_cache_bytes_fp16 = 6 * 1024**3

        policy = choose_memory_policy(max_vram="8GB", model_info=Info(), sequence_length=8192)
        self.assertTrue(policy.use_quantized_cache)
        self.assertTrue(policy.use_paged_cache)
        self.assertIn("KV dtype", policy.to_text())

    def test_optimization_report_infers_findings(self):
        report = OptimizationReport(
            title="demo",
            benchmarks=[
                BenchmarkResult("baseline", elapsed_ms=[2.0]),
                BenchmarkResult("optimized", elapsed_ms=[1.0]),
            ],
        )
        findings = infer_findings(report)
        self.assertTrue(any(item.area == "speed" for item in findings))
        self.assertIn("2.00x", report.to_text())


if __name__ == "__main__":
    unittest.main()
