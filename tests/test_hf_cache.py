import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.hf_cache import plan_hf_cache
from llm_memlab.memory_policy import MemoryPolicy


class HFCacheTests(unittest.TestCase):
    def test_policy_to_generation_kwargs(self):
        policy = MemoryPolicy(
            max_vram_bytes=8,
            kv_dtype="int8",
            cache_impl="paged",
            use_quantized_cache=True,
            use_paged_cache=True,
            page_size=16,
            use_chunked_lm_head=True,
            recommended_chunk_size=512,
            attention_backend="sdpa",
            notes=(),
        )
        plan = plan_hf_cache(policy)
        self.assertEqual(plan.generation_kwargs(), {"use_cache": True, "cache_implementation": "dynamic"})
        self.assertIn("quantized", plan.to_text())

    def test_model_without_generate_disables_cache(self):
        policy = MemoryPolicy(1, "fp16", "static", False, False, 16, False, 512, "sdpa", ())
        plan = plan_hf_cache(policy, model=object())
        self.assertFalse(plan.use_cache)


if __name__ == "__main__":
    unittest.main()
