import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import torch

from llm_memlab.hf_adapter import (
    LlamaMemoryAdapter,
    MemoryFirstHFConfig,
    MemoryFirstTransformersCache,
    QwenMemoryAdapter,
    detect_hf_adapter_info,
    install_memory_first_generate,
    make_transformers_cache_from_model,
    select_memory_adapter,
)
from llm_memlab.kernel_policy import KernelPolicy, select_kernel_policy
from llm_memlab.kv_cache import KVCacheConfig
from llm_memlab.memory_profiler import profile_decode_memory, write_memory_profile_html, write_memory_profile_json
from llm_memlab.quality_suite import QualityThresholds, assert_quality_regression, run_quality_regression


class TinyLM(torch.nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 8)
        self.proj = torch.nn.Linear(8, vocab)
        self.config = type(
            "Config", (), {"model_type": "llama", "num_hidden_layers": 1, "num_key_value_heads": 2, "hidden_size": 8, "head_dim": 4}
        )()

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        logits = self.proj(self.embed(input_ids))
        k = torch.zeros(input_ids.shape[0], 2, input_ids.shape[1], 4, dtype=torch.float32)
        v = torch.zeros_like(k)
        return {"logits": logits, "past_key_values": ((k, v),)}

    def generate(self, input_ids, max_new_tokens=1, past_key_values=None, use_cache=True, **kwargs):
        self.seen_cache = past_key_values
        return torch.cat([input_ids, torch.zeros(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)], dim=-1)


class RuntimeV3Tests(unittest.TestCase):
    def test_kernel_policy_explains_selection(self):
        selection = select_kernel_policy(q_heads=4, kv_heads=2, head_dim=32, sequence_length=128, paged=True, policy=KernelPolicy())
        self.assertIn(selection.backend, {"torch", "triton"})
        self.assertTrue(selection.stable)
        experimental = select_kernel_policy(
            q_heads=4, kv_heads=2, head_dim=32, sequence_length=8192, paged=True, policy=KernelPolicy(allow_experimental=True)
        )
        if experimental.backend != "torch":
            self.assertEqual(experimental.backend, "triton-experimental")
            self.assertFalse(experimental.stable)
        cutile = select_kernel_policy(
            q_heads=4,
            kv_heads=2,
            head_dim=32,
            sequence_length=128,
            paged=True,
            policy=KernelPolicy(backend="cutile-experimental", allow_experimental=True),
        )
        self.assertIn(cutile.backend, {"torch", "triton", "triton-experimental", "cutile-experimental"})
        self.assertTrue(any("CuTile" in reason or "Triton" in reason or "CUDA" in reason for reason in cutile.reasons))

    def test_memory_first_transformers_cache_update(self):
        cfg = KVCacheConfig(num_layers=1, batch_size=1, num_heads=2, head_dim=4, max_seq_len=8, dtype=torch.float32)
        cache = MemoryFirstTransformersCache(cfg, cache="quantized", quant_dtype="int8")
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        out_k, out_v = cache.update(k, v, 0)
        self.assertEqual(tuple(out_k.shape), (1, 2, 1, 4))
        self.assertEqual(cache.get_seq_length(), 1)

    def test_hf_generate_injection_smoke(self):
        model = TinyLM()
        install_memory_first_generate(model, MemoryFirstHFConfig(max_new_tokens=2))
        out = model.generate(torch.tensor([[1, 2]]), max_new_tokens=2)
        self.assertEqual(tuple(out.shape), (1, 4))
        self.assertIsNotNone(model.seen_cache)
        self.assertEqual(detect_hf_adapter_info(model).family, "llama")
        self.assertIsInstance(select_memory_adapter(model), LlamaMemoryAdapter)
        qwen = TinyLM()
        qwen.config.model_type = "qwen2"
        self.assertIsInstance(select_memory_adapter(qwen), QwenMemoryAdapter)
        cache = make_transformers_cache_from_model(model, MemoryFirstHFConfig(), batch_size=1, max_seq_len=8)
        self.assertEqual(cache.config.num_layers, 1)

    def test_memory_profiler_html(self):
        model = TinyLM().eval()
        profile = profile_decode_memory(model, torch.tensor([[1, 2]]), max_new_tokens=2)
        self.assertEqual(len(profile.samples), 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_memory_profile_html(profile, pathlib.Path(tmp) / "memory.html")
            json_path = write_memory_profile_json(profile, pathlib.Path(tmp) / "memory.json")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<!doctype html>"))
            self.assertIn("schema_version", json_path.read_text(encoding="utf-8"))

    def test_quality_regression_suite(self):
        baseline = TinyLM().eval()
        candidate = TinyLM().eval()
        candidate.load_state_dict(baseline.state_dict())
        result = run_quality_regression(baseline, candidate, torch.tensor([[1, 2, 3]]), max_new_tokens=2, thresholds=QualityThresholds())
        self.assertTrue(result.passed)
        self.assertIn("schema_version", result.to_dict())
        assert_quality_regression(result)


if __name__ == "__main__":
    unittest.main()
