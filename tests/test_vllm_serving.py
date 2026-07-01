import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.backend_registry import default_backend_registry
from llm_memlab.backends.vllm import detect_vllm_runtime
from llm_memlab.benchmark_dashboard import BenchmarkDashboard
from llm_memlab.benchmark_store import read_benchmark_files
from llm_memlab.serving_benchmark import ServingBenchmarkResult, ServingRun


class VLLMServingTests(unittest.TestCase):
    def test_default_registry_exposes_vllm_serving_backend(self):
        registry = default_backend_registry()
        info = registry.get("vllm-serving")

        self.assertEqual(info.kind, "serving")
        self.assertIn("vllm", info.name)
        self.assertTrue(info.reason)

    def test_vllm_detector_reports_fallback_reason(self):
        info = detect_vllm_runtime()

        self.assertIsInstance(info.available, bool)
        self.assertTrue(info.fallback_reason)
        if not info.available:
            self.assertIn("fallback", info.fallback_reason.lower())

    def test_serving_result_writes_dashboard_records(self):
        result = ServingBenchmarkResult(
            model="unit-model",
            prompt="hello",
            runs=(
                ServingRun(
                    "hf-generate",
                    "hf",
                    True,
                    elapsed_ms=10.0,
                    first_token_ms=10.0,
                    tokens_per_second=100.0,
                    new_tokens=1,
                    token_match=True,
                ),
                ServingRun(
                    "vllm",
                    "vllm-serving",
                    False,
                    fallback_reason="vLLM is not installed; policy fallback: use HF",
                ),
            ),
            metadata={"backend": "serving-benchmark", "gpu": "unit-test"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            json_path = result.write_json(tmp_path / "serving.json")
            csv_path = result.write_csv(tmp_path / "serving.csv")
            html_path = result.write_html(tmp_path / "serving.html")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "llm_memlab.serving_benchmark.v1")
            self.assertEqual(len(payload["records"]), 2)
            self.assertEqual(len(read_benchmark_files([json_path, csv_path])), 4)
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Serving metrics", html)
            self.assertIn("First token", html)
            self.assertIn("vllm-serving", html)

    def test_dashboard_renders_serving_specific_columns(self):
        result = ServingBenchmarkResult(
            model="unit-model",
            prompt="hello",
            runs=(ServingRun("hf-generate", "hf", True, elapsed_ms=1.0, first_token_ms=1.0, tokens_per_second=1.0),),
        )
        html = BenchmarkDashboard(tuple(result.to_records())).to_html()

        self.assertIn("Serving metrics", html)
        self.assertIn("Tok/s", html)
        self.assertIn("Prefix cache", html)


if __name__ == "__main__":
    unittest.main()
