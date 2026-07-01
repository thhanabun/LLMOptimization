import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.benchmark_dashboard import benchmark_dashboard_from_files, write_benchmark_dashboard_html
from llm_memlab.benchmark_store import BenchmarkRecord, write_benchmark_csv, write_benchmark_json
from llm_memlab.certification_matrix import ModelCertificationTarget, certify_model_matrix
from llm_memlab.cli import main
from llm_memlab.hardware import HardwareProfile
from llm_memlab.hf_adapter import LlamaMemoryAdapter, MemoryFirstHFConfig
from llm_memlab.hf_cache_profiles import QuantizedCacheCertificationProfile, write_quantized_cache_profiles
from llm_memlab.inspector import inspect_model
from llm_memlab.kernel_certification import KernelCertificationCase, KernelCertificationReport, KernelCertificationResult
from llm_memlab.kernel_promotion import KernelPromotionRequirements, decide_kernel_promotion
from llm_memlab.local_model_harness import scan_local_model_fixtures


class TinyConfig:
    model_type = "llama"
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 1
    hidden_size = 8
    head_dim = 4


class SignatureLimitedGenerateModel:
    config = TinyConfig()

    def generate(self, input_ids, max_new_tokens=1, use_cache=True, past_key_values=None, attention_mask=None):
        del max_new_tokens, use_cache, past_key_values, attention_mask
        return input_ids


class RuntimeCacheRejectingGenerateModel:
    config = TinyConfig()

    def generate(self, input_ids, max_new_tokens=1, use_cache=True, past_key_values=None, attention_mask=None, cache_position=None):
        del max_new_tokens, use_cache, attention_mask, cache_position
        if past_key_values is not None:
            raise RuntimeError("device mismatch while using external cache object")
        return input_ids


class Gemma4LikeConfig:
    model_type = "gemma4"

    class text_config:
        model_type = "gemma4_text"
        hidden_size = 2560
        intermediate_size = 10240
        num_hidden_layers = 42
        num_attention_heads = 8
        num_key_value_heads = 2
        head_dim = 256
        vocab_size = 262144
        max_position_embeddings = 131072


class ConfigOnlyModel:
    config = Gemma4LikeConfig()

    def parameters(self):
        return iter(())


class ProductionMatrixDashboardTests(unittest.TestCase):
    def test_missing_real_model_matrix_writes_conservative_profile(self):
        target = ModelCertificationTarget("local-llama", "llama", "definitely-missing-local-model", local_files_only=True)
        with tempfile.TemporaryDirectory() as tmp:
            profiles_path = pathlib.Path(tmp) / "profiles.json"
            report_path = pathlib.Path(tmp) / "matrix.json"
            report = certify_model_matrix([target], prompts=("hello",), allow_remote=False)

            self.assertTrue(report.passed)
            self.assertEqual(report.outcomes[0].status, "skipped")
            self.assertEqual(report.profiles[0].certified_backend, "paged")
            report.write_json(report_path)
            report.write_profiles(profiles_path)

            payload = json.loads(profiles_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["profiles"][0]["family"], "llama")
            self.assertFalse(payload["profiles"][0]["production"])

    def test_strict_certification_gate_fails_when_models_are_skipped(self):
        target = ModelCertificationTarget("local-llama", "llama", "definitely-missing-local-model", local_files_only=True)
        report = certify_model_matrix([target], prompts=("hello",), allow_remote=False)
        gate = report.evaluate_gate(require_real_models=True, min_certified_models=1, strict=True)

        self.assertFalse(gate.passed)
        self.assertIn("skipped", "; ".join(gate.reasons))
        self.assertEqual(
            main(["certify-model-matrix", "--models", "llama=definitely-missing-local-model", "--strict", "--profiles-out", ""]),
            1,
        )

    def test_local_model_harness_scans_fixture_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "TinyLlama-1.1B-Chat-v1.0").mkdir()
            (root / "gemma-4-E4B-it").mkdir()
            report = scan_local_model_fixtures(root)
            available = [item for item in report.matches if item.available]

            self.assertEqual(available[0].fixture.family, "llama")
            self.assertGreaterEqual(report.available_count, 2)
            self.assertIn("gemma4", {item.fixture.name for item in available})
            self.assertEqual(main(["local-model-harness", "--root", str(root), "--json-out", str(root / "fixtures.json")]), 0)

    def test_inspector_reads_nested_gemma4_text_config(self):
        info = inspect_model(ConfigOnlyModel())

        self.assertEqual(info.model_type, "gemma4_text")
        self.assertEqual(info.hidden_size, 2560)
        self.assertEqual(info.num_layers, 42)
        self.assertEqual(info.num_key_value_heads, 2)
        self.assertTrue(any("nested language config" in note for note in info.notes))

    def test_profile_cli_export_merge_explain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            first = tmp_path / "first.json"
            merged = tmp_path / "merged.json"
            profile = QuantizedCacheCertificationProfile(
                family="llama",
                model="tiny",
                gpu_arch="cpu",
                certified_backend="quantized",
                quant_dtype="int8",
                safe_prompt_tokens=64,
                production=True,
            )
            write_quantized_cache_profiles([profile], first)

            self.assertEqual(main(["profile", "merge", "--inputs", str(first), "--out", str(merged)]), 0)
            self.assertEqual(
                main(
                    [
                        "profile",
                        "explain",
                        "--family",
                        "llama",
                        "--model",
                        "tiny",
                        "--prompt-tokens",
                        "16",
                        "--profile",
                        str(merged),
                    ]
                ),
                0,
            )

    def test_benchmark_dashboard_reads_json_and_csv_history(self):
        records = [
            BenchmarkRecord(
                "decode",
                "kernel",
                1.25,
                1.0,
                1.5,
                peak_cuda_bytes=1024,
                extra={"quality_passed": True, "mean_abs": 0.001, "model": "tiny"},
                metadata={"gpu": "unit-test-gpu", "backend": "triton", "commit": "abc123"},
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            json_path = write_benchmark_json(records, tmp_path / "history.json")
            csv_path = write_benchmark_csv(records, tmp_path / "history.csv")
            html_path = tmp_path / "dashboard.html"

            dashboard = benchmark_dashboard_from_files([json_path, csv_path])
            self.assertEqual(len(dashboard.records), 2)
            write_benchmark_dashboard_html([json_path, csv_path], html_path, title="test dashboard")

            html = html_path.read_text(encoding="utf-8")
            self.assertIn("latency", html.lower())
            self.assertIn("Latency trend", html)
            self.assertIn("Quality drift trend", html)
            self.assertIn("Memory peak trend", html)
            self.assertIn("unit-test-gpu", html)
            self.assertIn("triton", html)

    def test_kernel_promotion_rejects_small_smoke_matrix_by_default(self):
        report = KernelCertificationReport(
            (
                KernelCertificationResult(
                    KernelCertificationCase(compute_dtype="fp16", quant_dtype="int8", sequence_length=4096),
                    passed=True,
                    fused_ms=1.0,
                    reference_ms=2.0,
                ),
                KernelCertificationResult(
                    KernelCertificationCase(compute_dtype="bf16", quant_dtype="uint8", sequence_length=4096),
                    passed=True,
                    fused_ms=1.0,
                    reference_ms=2.0,
                ),
            )
        )
        decision = decide_kernel_promotion(
            report,
            backend="triton",
            hardware=HardwareProfile(device="cuda", cuda_available=True, cuda_capability=(8, 0)),
        )
        self.assertFalse(decision.promoted)
        self.assertIn("coverage", "; ".join(decision.reasons))

    def test_kernel_promotion_allows_explicit_small_requirement_for_unit_tests(self):
        report = KernelCertificationReport(
            (
                KernelCertificationResult(
                    KernelCertificationCase(
                        batch=1,
                        q_heads=2,
                        kv_heads=1,
                        head_dim=16,
                        sequence_length=32,
                        page_size=16,
                        compute_dtype="fp16",
                        quant_dtype="int8",
                    ),
                    passed=True,
                    fused_ms=1.0,
                    reference_ms=2.0,
                ),
                KernelCertificationResult(
                    KernelCertificationCase(
                        batch=1,
                        q_heads=2,
                        kv_heads=1,
                        head_dim=16,
                        sequence_length=32,
                        page_size=16,
                        compute_dtype="bf16",
                        quant_dtype="uint8",
                    ),
                    passed=True,
                    fused_ms=1.0,
                    reference_ms=2.0,
                ),
            )
        )
        requirements = KernelPromotionRequirements(
            required_batches=(1,),
            required_q_heads=(2,),
            required_kv_heads=(1,),
            required_head_dims=(16,),
            required_sequence_lengths=(32,),
            required_page_sizes=(16,),
            min_cases=2,
            require_gqa=False,
            require_mqa=True,
            require_long_context=False,
        )
        decision = decide_kernel_promotion(
            report,
            backend="triton",
            hardware=HardwareProfile(device="cuda", cuda_available=True, cuda_capability=(8, 0)),
            requirements=requirements,
        )
        self.assertTrue(decision.promoted, decision.to_text())

        cutile = decide_kernel_promotion(
            report,
            backend="cutile",
            hardware=HardwareProfile(device="cuda", cuda_available=True, cuda_capability=(8, 0)),
            requirements=requirements,
        )
        self.assertFalse(cutile.promoted)
        self.assertIn("Hopper/Blackwell", "; ".join(cutile.reasons))

    def test_hf_adapter_filters_unsupported_generate_kwargs(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        model = SignatureLimitedGenerateModel()
        input_ids = torch.tensor([[1, 2, 3]])
        adapter = LlamaMemoryAdapter(model, MemoryFirstHFConfig(cache="paged"))
        kwargs = adapter.prepare_generate_kwargs(input_ids, max_new_tokens=1, sliding_window=8)
        plan = adapter.integration_plan(input_ids)

        self.assertNotIn("cache_position", kwargs)
        self.assertNotIn("sliding_window", kwargs)
        self.assertIn("attention_mask", kwargs)
        self.assertFalse(plan.cache_position)
        self.assertTrue(plan.attention_mask)

    def test_hf_adapter_falls_back_on_runtime_cache_errors(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        model = RuntimeCacheRejectingGenerateModel()
        input_ids = torch.tensor([[1, 2, 3]])
        result = LlamaMemoryAdapter(model, MemoryFirstHFConfig(cache="paged")).generate(input_ids, max_new_tokens=1)

        self.assertEqual(result.cache_impl, "llama:original-fallback")
        self.assertIsNotNone(result.fallback_reason)


if __name__ == "__main__":
    unittest.main()
