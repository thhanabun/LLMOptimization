import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.benchmark import BenchmarkResult
from llm_memlab.benchmark_store import read_benchmark_json, record_from_benchmark, write_benchmark_csv, write_benchmark_json


class BenchmarkStoreTests(unittest.TestCase):
    def test_record_roundtrip_json_and_csv(self):
        result = BenchmarkResult("forward", [1.0, 3.0], peak_cuda_bytes=123, output_shape="(1,)", extra={"tokens": 4})
        record = record_from_benchmark(result, kind="prefill")
        self.assertEqual(record.mean_ms, 2.0)
        self.assertEqual(record.extra["tokens"], 4)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            json_path = write_benchmark_json([record], root / "bench.json")
            csv_path = write_benchmark_csv([record], root / "bench.csv")
            loaded = read_benchmark_json(json_path)
            self.assertEqual(loaded[0].name, "forward")
            self.assertEqual(loaded[0].kind, "prefill")
            self.assertIn("extra", csv_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))[0]["mean_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
