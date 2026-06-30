import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.backends import default_backend_registry
from llm_memlab.backends.cutile import detect_cutile_runtime
from llm_memlab.kernel_policy import KernelPolicy, default_kernel_policy


class APIPolishTests(unittest.TestCase):
    def test_backends_module_and_kernel_policy(self):
        names = [item.name for item in default_backend_registry().list()]
        self.assertIn("torch", names)
        self.assertIn("cutile", names)
        self.assertIsNotNone(detect_cutile_runtime())
        policy = default_kernel_policy()
        policy.validate()
        with self.assertRaises(ValueError):
            KernelPolicy(backend="bad").validate()


if __name__ == "__main__":
    unittest.main()
