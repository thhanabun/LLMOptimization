import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.hf_cache_certification import assert_hf_cache_certified, certify_hf_cache_model, hf_cache_certification_to_html
from llm_memlab.hf_cache_policy import HFCachePolicy, select_hf_cache_policy


class TinyTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt"):
        del return_tensors
        values = [max(1, min(31, ord(ch) % 32)) for ch in prompt] or [1]
        return {"input_ids": torch.tensor([values], dtype=torch.long), "attention_mask": torch.ones(1, len(values), dtype=torch.long)}


class TinyConfig:
    model_type = "qwen3"
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    hidden_size = 16
    head_dim = 8
    vocab_size = 32


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TinyCertLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig()
        self.embed = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.proj = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=True, attention_mask=None, **kwargs):
        del past_key_values, use_cache, attention_mask, kwargs
        logits = self.proj(self.embed(input_ids))
        return {"logits": logits}

    def generate(self, input_ids, max_new_tokens=1, past_key_values=None, use_cache=True, attention_mask=None, **kwargs):
        del past_key_values, use_cache, attention_mask, kwargs
        return torch.cat([input_ids, torch.zeros(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)], dim=-1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HFCacheCertificationTests(unittest.TestCase):
    def test_certify_hf_cache_model_passes_direct_cache(self):
        report = certify_hf_cache_model(
            TinyCertLM(),
            TinyTokenizer(),
            model_name="tiny-qwen3",
            prompts=["hi"],
            token_counts=[1],
            caches=["paged"],
            quant_dtypes=["int8"],
            allow_experimental_direct_cache=True,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.results), 1)
        self.assertTrue(report.results[0].direct_cache)
        self.assertEqual(report.results[0].cache_impl, "qwen3:paged")
        assert_hf_cache_certified(report)
        self.assertEqual(report.to_records()[0].kind, "hf-cache-certification")
        self.assertIn("<table>", hf_cache_certification_to_html(report))
        with tempfile.TemporaryDirectory() as tmp:
            html = report.write_html(pathlib.Path(tmp) / "cert.html")
            self.assertTrue(html.read_text(encoding="utf-8").startswith("<!doctype html>"))

    def test_certification_gate_fails_when_direct_cache_is_disabled(self):
        report = certify_hf_cache_model(
            TinyCertLM(),
            TinyTokenizer(),
            model_name="tiny-qwen3",
            prompts=["hi"],
            token_counts=[1],
            caches=["paged"],
            quant_dtypes=["int8"],
            allow_experimental_direct_cache=False,
        )
        self.assertFalse(report.passed)
        with self.assertRaises(AssertionError):
            assert_hf_cache_certified(report)

    def test_experimental_case_reports_without_failing_gate(self):
        report = certify_hf_cache_model(
            TinyCertLM(),
            TinyTokenizer(),
            model_name="tiny-qwen3",
            prompts=["hello"],
            token_counts=[1],
            caches=["paged"],
            experimental_caches=["quantized"],
            quant_dtypes=["int8"],
            allow_experimental_direct_cache=True,
        )
        self.assertFalse(report.results[1].production)
        assert_hf_cache_certified(report)

    def test_cache_policy_falls_back_qwen3_quantized_prefill(self):
        decision = select_hf_cache_policy(
            family="qwen3",
            prompt_tokens=8,
            device="cpu",
            policy=HFCachePolicy(requested_cache="quantized"),
        )
        self.assertEqual(decision.cache, "paged")
        self.assertFalse(decision.quantized_allowed)
        self.assertIn("falling back", "; ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
