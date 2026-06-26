import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab import GraphSpec, MemoryPlanner, OperationSpec, TensorSpec


class PlannerTests(unittest.TestCase):
    def test_graph_lifetimes_and_plan(self):
        graph = GraphSpec(inputs=("x",), outputs=("z",))
        graph.add_tensor(TensorSpec.from_shape("x", (2, 4), dtype="fp16"))
        graph.add_tensor(TensorSpec.from_shape("y", (2, 4), dtype="fp16"))
        graph.add_tensor(TensorSpec.from_shape("z", (2, 4), dtype="fp16"))
        graph.add_op(OperationSpec.make("a", "identity", ("x",), ("y",)))
        graph.add_op(OperationSpec.make("b", "identity", ("y",), ("z",)))

        lifetimes = graph.tensor_lifetimes()
        plan = MemoryPlanner(lifetimes).plan()

        self.assertEqual(len(lifetimes), 3)
        self.assertGreater(plan.eager_peak_bytes, 0)
        self.assertGreaterEqual(plan.total_allocated_bytes, plan.eager_peak_bytes)


if __name__ == "__main__":
    unittest.main()
