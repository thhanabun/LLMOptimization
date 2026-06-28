import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import llm_memlab.oom_runner as oom_runner
from llm_memlab.oom_runner import OOMStrategy, is_oom_error, run_with_oom_fallback


class OOMRunnerTests(unittest.TestCase):
    def test_fallback_recovers_from_oom(self):
        calls = []

        def fn(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return kwargs["value"]

        result = run_with_oom_fallback(fn, [OOMStrategy("first", {"value": 1}), OOMStrategy("second", {"value": 2})])
        self.assertEqual(result.value, 2)
        self.assertEqual(result.strategy.name, "second")
        self.assertEqual(result.attempts, ("first: OOM", "second"))

    def test_fallback_clears_cuda_cache_after_oom(self):
        calls = []
        original = oom_runner.clear_cuda_cache
        oom_runner.clear_cuda_cache = lambda: calls.append("cleared") or True
        try:
            run_with_oom_fallback(
                lambda **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")) if kwargs.get("fail") else "ok",
                [OOMStrategy("first", {"fail": True}), OOMStrategy("second", {"fail": False})],
            )
        finally:
            oom_runner.clear_cuda_cache = original
        self.assertEqual(calls, ["cleared"])

    def test_non_oom_error_is_not_swallowed(self):
        with self.assertRaisesRegex(RuntimeError, "shape"):
            run_with_oom_fallback(lambda: (_ for _ in ()).throw(RuntimeError("shape mismatch")), [OOMStrategy("x")])
        self.assertTrue(is_oom_error(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")))


if __name__ == "__main__":
    unittest.main()
