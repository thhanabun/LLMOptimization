import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.backend_registry import BackendRegistry, default_backend_registry


class BackendRegistryTests(unittest.TestCase):
    def test_custom_registry_selects_highest_available_priority(self):
        registry = BackendRegistry()
        registry.register("slow", lambda: (True, "ok"), priority=1)
        registry.register("fast", lambda: (True, "ok"), priority=10)
        self.assertEqual(registry.best("slow", "fast").name, "fast")
        self.assertIn("fast", registry.to_text())

    def test_default_registry_has_torch(self):
        names = [item.name for item in default_backend_registry().list()]
        self.assertIn("torch", names)


if __name__ == "__main__":
    unittest.main()
