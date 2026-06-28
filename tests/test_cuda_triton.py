import os
import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover
    triton = None

from llm_memlab.kernels import (
    quantized_kv_attention,
    scaled_dot_product_attention,
    triton_dequantize_int8_per_token,
    triton_dequantize_uint8_per_token,
    triton_fused_int8_kv_attention,
    triton_fused_uint8_kv_attention,
    triton_quantize_int8_per_token,
    triton_quantize_uint8_per_token,
)

CUDA_TRITON = torch is not None and triton is not None and torch.cuda.is_available()


@unittest.skipUnless(CUDA_TRITON, "CUDA and Triton are required")
class CUDATritonKernelTests(unittest.TestCase):
    def test_triton_int8_quantize_dequantize_cuda(self):
        x = torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.float16)
        q, scale = triton_quantize_int8_per_token(x)
        y = triton_dequantize_int8_per_token(q, scale, dtype=x.dtype)
        torch.cuda.synchronize()
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(y.shape, x.shape)
        self.assertLess((x - y).abs().mean().item(), 0.02)

    def test_triton_uint8_quantize_dequantize_cuda(self):
        x = torch.randn(2, 4, 16, 64, device="cuda", dtype=torch.float16)
        q, scale, zero_point = triton_quantize_uint8_per_token(x)
        y = triton_dequantize_uint8_per_token(q, scale, zero_point, dtype=x.dtype)
        torch.cuda.synchronize()
        self.assertEqual(q.dtype, torch.uint8)
        self.assertEqual(y.shape, x.shape)
        self.assertLess((x - y).abs().mean().item(), 0.02)

    def test_fused_int8_decode_attention_matches_dequant_sdpa(self):
        q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        qk, ks = triton_quantize_int8_per_token(k)
        qv, vs = triton_quantize_int8_per_token(v)
        fused = triton_fused_int8_kv_attention(q, qk, ks, qv, vs)
        k_dequant = triton_dequantize_int8_per_token(qk, ks, dtype=k.dtype)
        v_dequant = triton_dequantize_int8_per_token(qv, vs, dtype=v.dtype)
        ref = scaled_dot_product_attention(q, k_dequant, v_dequant)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(fused, ref, atol=2e-2, rtol=2e-2))

    def test_fused_uint8_decode_attention_matches_dequant_sdpa(self):
        q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        qk, ks, kz = triton_quantize_uint8_per_token(k)
        qv, vs, vz = triton_quantize_uint8_per_token(v)
        fused = triton_fused_uint8_kv_attention(q, qk, ks, kz, qv, vs, vz)
        k_dequant = triton_dequantize_uint8_per_token(qk, ks, kz, dtype=k.dtype)
        v_dequant = triton_dequantize_uint8_per_token(qv, vs, vz, dtype=v.dtype)
        ref = scaled_dot_product_attention(q, k_dequant, v_dequant)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(fused, ref, atol=2e-2, rtol=2e-2))

    def test_quantized_kv_attention_triton_backend_cuda(self):
        q = torch.randn(1, 2, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        out = quantized_kv_attention(q, k, v, quant_dtype="int8", backend="triton")
        torch.cuda.synchronize()
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_cli_kernel_demo_cuda_smoke(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-m", "llm_memlab", "kernel-demo", "--device", "cuda", "--repeats", "1"],
            cwd=pathlib.Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("device=cuda", result.stdout)


if __name__ == "__main__":
    unittest.main()
