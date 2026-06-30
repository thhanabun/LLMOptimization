import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.backend_registry import BackendPlugin, BackendRegistry, default_backend_registry
from llm_memlab.env_certification import certify_environment
from llm_memlab.hardware import HardwareProfile, detect_hardware_profile
from llm_memlab.hf_adapter import (
    DeepSeekMemoryAdapter,
    FalconMemoryAdapter,
    GemmaMemoryAdapter,
    GPTNeoXMemoryAdapter,
    LlamaMemoryAdapter,
    MemoryFirstHFConfig,
    MixtralMemoryAdapter,
    PhiMemoryAdapter,
    Qwen3MemoryAdapter,
    adapter_satisfies_contract,
    list_memory_adapters,
    select_memory_adapter,
)
from llm_memlab.hf_cache_policy import HFCachePolicy, select_hf_cache_policy
from llm_memlab.hf_cache_profiles import (
    QuantizedCacheCertificationProfile,
    load_quantized_cache_profiles,
    write_quantized_cache_profiles,
)


class TinyConfig:
    def __init__(self, model_type):
        self.model_type = model_type
        self.num_hidden_layers = 1
        self.num_attention_heads = 2
        self.num_key_value_heads = 1
        self.hidden_size = 8
        self.head_dim = 4


class TinyModel:
    def __init__(self, model_type):
        self.config = TinyConfig(model_type)


class GeneralizationTests(unittest.TestCase):
    def test_adapter_registry_selects_major_model_families(self):
        expected = {
            "llama": LlamaMemoryAdapter,
            "qwen3": Qwen3MemoryAdapter,
            "mixtral": MixtralMemoryAdapter,
            "gemma2": GemmaMemoryAdapter,
            "phi3": PhiMemoryAdapter,
            "deepseek": DeepSeekMemoryAdapter,
            "gpt_neox": GPTNeoXMemoryAdapter,
            "falcon": FalconMemoryAdapter,
        }
        prefixes = {prefix for prefix, _ in list_memory_adapters()}
        for family, adapter_cls in expected.items():
            self.assertIn(family, prefixes)
            adapter = select_memory_adapter(TinyModel(family), MemoryFirstHFConfig(cache="paged"))
            self.assertIsInstance(adapter, adapter_cls)
            self.assertTrue(adapter_satisfies_contract(adapter))

    def test_profile_registry_json_drives_policy(self):
        profile = QuantizedCacheCertificationProfile(
            family="gemma",
            model="tiny",
            model_architecture="gemma2",
            gpu_arch="cpu",
            certified_backend="quantized",
            quant_dtype="int8",
            safe_prompt_tokens=32,
            production=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "profiles.json"
            write_quantized_cache_profiles([profile], path)
            registry = load_quantized_cache_profiles([path])
            self.assertEqual(registry.select(family="gemma", model="tiny-gemma", quant_dtype="int8").certified_backend, "quantized")
            decision = select_hf_cache_policy(
                family="gemma",
                prompt_tokens=8,
                device="cpu",
                policy=HFCachePolicy(
                    requested_cache="quantized",
                    model="tiny-gemma",
                    quant_dtype="int8",
                    quantized_profile_paths=(str(path),),
                    hardware_profile=HardwareProfile(device="cpu"),
                ),
            )
            self.assertEqual(decision.cache, "quantized", decision.to_text())

    def test_gpu_arch_mismatch_falls_back(self):
        certified = QuantizedCacheCertificationProfile(
            family="llama",
            gpu_arch="hopper",
            quant_dtype="int8",
            safe_prompt_tokens=32,
            production=True,
        )
        decision = select_hf_cache_policy(
            family="llama",
            prompt_tokens=8,
            policy=HFCachePolicy(
                requested_cache="quantized",
                quantized_profiles=(certified,),
                hardware_profile=HardwareProfile(device="cpu"),
            ),
        )
        self.assertEqual(decision.cache, "paged")
        self.assertIn("gpu arch", "; ".join(decision.reasons))

    def test_backend_registry_accepts_plugin_backends(self):
        registry = BackendRegistry()
        registry.register_plugin(BackendPlugin("toy", lambda: (True, "ok"), priority=99, kind="test"))
        best = registry.best("toy")
        self.assertTrue(best.available)
        self.assertEqual(best.kind, "test")
        names = {item.name for item in default_backend_registry().list()}
        self.assertIn("flash-attn", names)
        self.assertIn("bitsandbytes", names)

    def test_hardware_profile_and_certify_env_smoke(self):
        hardware = detect_hardware_profile("cpu")
        self.assertIsNotNone(hardware.schema_version)
        report = certify_environment(run_hf=False, run_kernel=False)
        self.assertTrue(report.passed)
        payload = report.to_dict()
        self.assertIn("hardware", payload)
        self.assertIn("adapter_matrix", payload)


if __name__ == "__main__":
    unittest.main()
