import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.hf_adapter import (
    LlamaMemoryAdapter,
    MemoryFirstHFConfig,
    MemoryFirstTransformersCache,
    MistralMemoryAdapter,
    Qwen3MemoryAdapter,
    QwenMemoryAdapter,
    install_memory_first_generate,
    memory_first_generate_hf,
    select_memory_adapter,
)
from llm_memlab.kv_cache import KVCacheConfig


class TinyFamilyConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 2
        self.hidden_size = 16
        self.head_dim = 8
        self.vocab_size = 32
        self.sliding_window = 64 if model_type == "mistral" else None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TinyFamilyLM(torch.nn.Module):
    def __init__(self, model_type: str, *, reject_external_cache: bool = False):
        super().__init__()
        self.config = TinyFamilyConfig(model_type)
        self.reject_external_cache = reject_external_cache
        self.embed = torch.nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.proj = torch.nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.last_generate_kwargs = None
        self.last_forward_kwargs = None

    def forward(
        self, input_ids, past_key_values=None, use_cache=True, attention_mask=None, cache_position=None, sliding_window=None, **kwargs
    ):
        self.last_forward_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "sliding_window": sliding_window,
        }
        logits = self.proj(self.embed(input_ids))
        length = input_ids.shape[-1] if past_key_values is None else past_key_values[0][0].shape[-2] + input_ids.shape[-1]
        key = torch.zeros(input_ids.shape[0], self.config.num_key_value_heads, length, self.config.head_dim)
        value = torch.zeros_like(key)
        return {"logits": logits, "past_key_values": tuple((key, value) for _ in range(self.config.num_hidden_layers))}

    def generate(
        self,
        input_ids,
        max_new_tokens=1,
        past_key_values=None,
        use_cache=True,
        attention_mask=None,
        cache_position=None,
        sliding_window=None,
        **kwargs,
    ):
        self.last_generate_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "sliding_window": sliding_window,
            "use_cache": use_cache,
        }
        if self.reject_external_cache and past_key_values is not None:
            raise TypeError("external cache object is not accepted")
        return torch.cat([input_ids, torch.zeros(input_ids.shape[0], max_new_tokens, dtype=input_ids.dtype)], dim=-1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class HFIntegrationMatrixTests(unittest.TestCase):
    def test_family_adapter_selection(self):
        self.assertIsInstance(select_memory_adapter(TinyFamilyLM("llama")), LlamaMemoryAdapter)
        self.assertIsInstance(select_memory_adapter(TinyFamilyLM("qwen2")), QwenMemoryAdapter)
        self.assertIsInstance(select_memory_adapter(TinyFamilyLM("qwen3")), Qwen3MemoryAdapter)
        self.assertIsInstance(select_memory_adapter(TinyFamilyLM("mistral")), MistralMemoryAdapter)

    def test_generate_injects_family_cache_api_kwargs(self):
        input_ids = torch.tensor([[1, 2, 3]])
        cases = [
            ("llama", LlamaMemoryAdapter),
            ("qwen2", QwenMemoryAdapter),
            ("mistral", MistralMemoryAdapter),
        ]
        for family, adapter_type in cases:
            with self.subTest(family=family):
                model = TinyFamilyLM(family)
                result = memory_first_generate_hf(
                    model, input_ids, MemoryFirstHFConfig(max_new_tokens=2, cache="quantized"), max_new_tokens=2
                )
                self.assertEqual(tuple(result.sequences.shape), (1, 5))
                self.assertIsInstance(select_memory_adapter(model), adapter_type)
                kwargs = model.last_generate_kwargs
                self.assertIsNotNone(kwargs["past_key_values"])
                self.assertIsNotNone(kwargs["attention_mask"])
                self.assertIsNotNone(kwargs["cache_position"])
                if family == "mistral":
                    self.assertEqual(kwargs["sliding_window"], 64)

    def test_qwen3_default_uses_quality_gated_fallback(self):
        model = TinyFamilyLM("qwen3")
        result = memory_first_generate_hf(model, torch.tensor([[1]]), MemoryFirstHFConfig(max_new_tokens=1), max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 2))
        self.assertIsNone(model.last_generate_kwargs["past_key_values"])
        self.assertIsNone(model.last_generate_kwargs["cache_position"])
        self.assertEqual(result.cache_impl, "qwen3:quality-gated-fallback")
        self.assertFalse(result.direct_cache)
        self.assertIn("not quality-certified", result.fallback_reason)

    def test_qwen3_experimental_single_token_does_not_send_top_level_cache_position(self):
        model = TinyFamilyLM("qwen3")
        config = MemoryFirstHFConfig(max_new_tokens=1, allow_experimental_direct_cache=True)
        result = memory_first_generate_hf(model, torch.tensor([[1]]), config, max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 2))
        self.assertIsNotNone(model.last_generate_kwargs["past_key_values"])
        self.assertIsNone(model.last_generate_kwargs["cache_position"])
        self.assertEqual(result.cache_impl, "qwen3:paged-policy-fallback")
        self.assertEqual(result.requested_cache_impl, "qwen3:quantized:int8")
        self.assertTrue(result.direct_cache)

    def test_qwen3_explicit_quantized_experimental_allows_direct_cache(self):
        model = TinyFamilyLM("qwen3")
        config = MemoryFirstHFConfig(
            max_new_tokens=1,
            allow_experimental_direct_cache=True,
            allow_experimental_quantized_cache=True,
        )
        result = memory_first_generate_hf(model, torch.tensor([[1]]), config, max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 2))
        self.assertIsNotNone(model.last_generate_kwargs["past_key_values"])
        self.assertIsNone(model.last_generate_kwargs["cache_position"])
        self.assertEqual(result.cache_impl, "qwen3:quantized")
        self.assertTrue(result.direct_cache)

    def test_qwen3_multi_token_prefill_uses_explained_fallback(self):
        model = TinyFamilyLM("qwen3")
        result = memory_first_generate_hf(model, torch.tensor([[1, 2, 3]]), MemoryFirstHFConfig(max_new_tokens=1), max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 4))
        self.assertIsNone(model.last_generate_kwargs["past_key_values"])
        self.assertIsNone(model.last_generate_kwargs["cache_position"])
        self.assertEqual(result.cache_impl, "qwen3:quality-gated-fallback")
        self.assertFalse(result.direct_cache)
        self.assertIn("not quality-certified", result.fallback_reason)

    def test_qwen3_quantized_multi_token_falls_back_to_paged_direct_cache(self):
        model = TinyFamilyLM("qwen3")
        config = MemoryFirstHFConfig(max_new_tokens=1, cache="quantized", allow_experimental_direct_cache=True)
        result = memory_first_generate_hf(model, torch.tensor([[1, 2, 3]]), config, max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 4))
        self.assertEqual(result.cache_impl, "qwen3:paged-policy-fallback")
        self.assertEqual(result.requested_cache_impl, "qwen3:quantized:int8")
        self.assertTrue(result.direct_cache)
        self.assertIn("quantized direct cache", result.fallback_reason)

    def test_generate_falls_back_to_original_when_model_rejects_cache(self):
        model = TinyFamilyLM("llama", reject_external_cache=True)
        result = memory_first_generate_hf(
            model, torch.tensor([[1, 2]]), MemoryFirstHFConfig(max_new_tokens=2, cache="quantized"), max_new_tokens=2
        )
        self.assertEqual(tuple(result.sequences.shape), (1, 4))
        self.assertEqual(result.steps, 2)
        self.assertIsNone(result.cache)
        self.assertEqual(result.cache_impl, "llama:original-fallback")

    def test_install_memory_first_generate_preserves_original_fallback(self):
        model = TinyFamilyLM("qwen2", reject_external_cache=True)
        install_memory_first_generate(model, MemoryFirstHFConfig(max_new_tokens=1))
        out = model.generate(torch.tensor([[1, 2]]), max_new_tokens=1)
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertIsNone(model.last_generate_kwargs["past_key_values"])

    def test_value_error_injection_falls_back_to_original_generate(self):
        class RejectingQwen3(TinyFamilyLM):
            def generate(
                self,
                input_ids,
                max_new_tokens=1,
                past_key_values=None,
                use_cache=True,
                attention_mask=None,
                cache_position=None,
                sliding_window=None,
                **kwargs,
            ):
                if past_key_values is not None:
                    raise ValueError("The following `model_kwargs` are not used by the model: ['past_key_values']")
                return super().generate(
                    input_ids, max_new_tokens=max_new_tokens, past_key_values=None, use_cache=use_cache, attention_mask=attention_mask
                )

        model = RejectingQwen3("qwen3")
        config = MemoryFirstHFConfig(
            max_new_tokens=1,
            allow_experimental_direct_cache=True,
            allow_experimental_quantized_cache=True,
        )
        result = memory_first_generate_hf(model, torch.tensor([[1]]), config, max_new_tokens=1)
        self.assertEqual(tuple(result.sequences.shape), (1, 2))
        self.assertEqual(result.cache_impl, "qwen3:original-fallback")
        self.assertFalse(result.direct_cache)
        self.assertIn("model_kwargs", result.fallback_reason)

    def test_transformers_cache_tracks_lengths_per_layer(self):
        cfg = KVCacheConfig(num_layers=2, batch_size=1, num_heads=2, head_dim=4, max_seq_len=8, dtype=torch.float32)
        cache = MemoryFirstTransformersCache(cfg, cache="paged")
        key0 = torch.ones(1, 2, 3, 4)
        value0 = key0 + 1
        key1 = key0 + 2
        value1 = key0 + 3
        out0 = cache.update(key0, value0, 0, None)
        out1 = cache.update(key1, value1, 1, None)
        self.assertEqual(cache.get_seq_length(0), 3)
        self.assertEqual(cache.get_seq_length(1), 3)
        self.assertEqual(cache.get_mask_sizes(2, 1), (5, 0))
        self.assertEqual(tuple(out0[0].shape), (1, 2, 3, 4))
        self.assertEqual(tuple(out1[0].shape), (1, 2, 3, 4))
        self.assertTrue(torch.allclose(out1[0], key1))

    @unittest.skipUnless(os.environ.get("LLM_MEMLAB_HF_SMOKE_MODEL"), "Set LLM_MEMLAB_HF_SMOKE_MODEL to a cached local HF model path/name")
    def test_cached_local_transformers_model_smoke(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:  # pragma: no cover
            self.skipTest("transformers is not installed")
        model_id = os.environ["LLM_MEMLAB_HF_SMOKE_MODEL"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
        encoded = tokenizer("hello", return_tensors="pt")
        result = memory_first_generate_hf(
            model,
            encoded["input_ids"],
            MemoryFirstHFConfig(max_new_tokens=1),
            max_new_tokens=1,
            attention_mask=encoded.get("attention_mask"),
        )
        self.assertGreaterEqual(result.sequences.shape[-1], encoded["input_ids"].shape[-1])


if __name__ == "__main__":
    unittest.main()
