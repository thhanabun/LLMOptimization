import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.benchmark import BenchmarkConfig
from llm_memlab.benchmark_suite import benchmark_inference_suite, compare_inference_suites


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BenchmarkSuiteTests(unittest.TestCase):
    def test_inference_suite_prefill_and_generate(self):
        class TinyLM(torch.nn.Module):
            def forward(self, input_ids):
                return {"logits": torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)}

            def generate(self, input_ids, max_new_tokens=2, do_sample=False):
                return torch.cat([input_ids, torch.ones(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)], dim=-1)

        model = TinyLM()
        encoded = {"input_ids": torch.ones(1, 3, dtype=torch.long)}
        result = benchmark_inference_suite(model, encoded, model_name="tiny", max_new_tokens=2, config=BenchmarkConfig(warmup=0, repeats=1))
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.new_tokens, 2)
        self.assertGreater(result.prefill_tokens_per_second, 0)
        self.assertIn("tiny", compare_inference_suites([result]))


if __name__ == "__main__":
    unittest.main()
