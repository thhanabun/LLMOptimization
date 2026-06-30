import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.benchmark_store import (
    BenchmarkRecord,
    benchmark_history,
    benchmark_history_gate,
    write_benchmark_json,
)
from llm_memlab.debugger_v2 import DebuggerV2Report, debugger_v2_to_html
from llm_memlab.hf_adapter_matrix import hf_adapter_matrix_to_text, production_hf_adapter_matrix
from llm_memlab.hf_cache_policy import HFCachePolicy, select_hf_cache_policy
from llm_memlab.hf_cache_profiles import QuantizedCacheCertificationProfile, select_quantized_cache_profile
from llm_memlab.memory_profiler import _attribute_cache_memory


class ProductionHardeningV2Tests(unittest.TestCase):
    def test_adapter_matrix_covers_major_families(self):
        entries = production_hf_adapter_matrix(transformers_version="test")
        families = {entry.family for entry in entries}
        self.assertTrue({"llama", "qwen", "qwen3", "mistral"}.issubset(families))
        self.assertIn("Qwen3MemoryAdapter", hf_adapter_matrix_to_text(entries))

    def test_quantized_profile_drives_policy_fallback_and_allow(self):
        fallback = select_hf_cache_policy(
            family="llama",
            prompt_tokens=8,
            device="cpu",
            policy=HFCachePolicy(requested_cache="quantized", quant_dtype="int8"),
        )
        self.assertEqual(fallback.cache, "paged")
        self.assertFalse(fallback.quantized_allowed)

        certified = QuantizedCacheCertificationProfile(
            family="llama",
            quant_dtype="int8",
            safe_prompt_tokens=16,
            production=True,
        )
        allowed = select_hf_cache_policy(
            family="llama",
            prompt_tokens=8,
            device="cpu",
            policy=HFCachePolicy(requested_cache="quantized", quant_dtype="int8", quantized_profiles=(certified,)),
        )
        self.assertEqual(allowed.cache, "quantized")
        self.assertTrue(allowed.quantized_allowed)
        self.assertTrue(select_quantized_cache_profile(family="llama", quant_dtype="int8", profiles=(certified,)).production)

    def test_benchmark_history_uses_median_baseline_for_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_a = pathlib.Path(tmp) / "base_a.json"
            base_b = pathlib.Path(tmp) / "base_b.json"
            cand = pathlib.Path(tmp) / "cand.json"
            write_benchmark_json([BenchmarkRecord("decode", "kernel", 10.0, 10.0, 10.0)], base_a)
            write_benchmark_json([BenchmarkRecord("decode", "kernel", 12.0, 12.0, 12.0)], base_b)
            write_benchmark_json([BenchmarkRecord("decode", "kernel", 11.0, 11.0, 11.0)], cand)
            history = benchmark_history([base_a, base_b])
            self.assertEqual(history.source_count, 2)
            result = benchmark_history_gate([base_a, base_b], [cand])
            self.assertTrue(result.passed, result.to_text())

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_memory_attribution_reads_legacy_past_key_values(self):
        key = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
        value = torch.zeros(1, 2, 4, 8, dtype=torch.float16)
        attrs = _attribute_cache_memory(((key, value),))
        self.assertEqual(attrs[0].name, "layer_0.kv_cache")
        self.assertEqual(attrs[0].bytes, key.numel() * key.element_size() + value.numel() * value.element_size())

    def test_debugger_v2_html_combines_sections(self):
        html = debugger_v2_to_html(metadata={"gpu": "test"})
        self.assertIn("Cache certification", html)
        self.assertIn("Memory", html)
        self.assertIn("Layers / drift", html)
        report = DebuggerV2Report(metadata={"ok": True})
        self.assertIn("llm_memlab.debugger_v2.v1", report.to_html())


if __name__ == "__main__":
    unittest.main()
