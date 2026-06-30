import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from llm_memlab.kernel_certification import (
    CERTIFICATION_SCHEMA_VERSION,
    KernelCertificationCase,
    certify_quantized_attention,
    default_kernel_certification_cases,
)


class KernelCertificationTests(unittest.TestCase):
    def test_quick_cases_cover_int8_and_uint8(self):
        cases = default_kernel_certification_cases(quick=True)
        self.assertEqual({case.quant_dtype for case in cases}, {"int8", "uint8"})
        for case in cases:
            case.validate()
            self.assertIn("cert/", case.name)

    def test_certification_report_writes_benchmark_records(self):
        report = certify_quantized_attention(
            cases=[KernelCertificationCase(q_heads=2, kv_heads=1, head_dim=16, sequence_length=16, quant_dtype="int8")],
            repeats=1,
            warmup=0,
            seed=7,
        )
        self.assertTrue(report.results)
        self.assertEqual(report.schema_version, CERTIFICATION_SCHEMA_VERSION)
        self.assertTrue(report.passed or any(not item.skipped for item in report.results))
        records = report.to_records()
        self.assertEqual(records[0].kind, "kernel-certification")
        self.assertIn("certification_schema_version", records[0].extra)
        with tempfile.TemporaryDirectory() as tmp:
            json_path = report.write_json(pathlib.Path(tmp) / "cert.json")
            csv_path = report.write_csv(pathlib.Path(tmp) / "cert.csv")
            self.assertIn("kernel-certification", json_path.read_text(encoding="utf-8"))
            self.assertIn("schema_version", csv_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_quick_certification_passes_quality(self):
        report = certify_quantized_attention(quick=True, repeats=1, warmup=0, seed=11)
        self.assertFalse(report.skipped)
        self.assertTrue(report.passed, report.to_text())


if __name__ == "__main__":
    unittest.main()
