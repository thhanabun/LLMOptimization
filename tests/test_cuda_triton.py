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

from llm_memlab.decode_benchmarks import benchmark_fused_decode_attention
from llm_memlab.kernels import (
    quantized_kv_attention,
    scaled_dot_product_attention,
    triton_dequantize_int8_per_token,
    triton_dequantize_uint8_per_token,
    triton_fused_int8_kv_attention,
    triton_fused_int8_paged_kv_attention,
    triton_fused_uint8_paged_kv_attention,
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

    def test_fused_int8_decode_attention_supports_gqa(self):
        q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        qk, ks = triton_quantize_int8_per_token(k)
        qv, vs = triton_quantize_int8_per_token(v)
        fused = triton_fused_int8_kv_attention(q, qk, ks, qv, vs)
        k_dequant = triton_dequantize_int8_per_token(qk, ks, dtype=k.dtype).repeat_interleave(2, dim=1)
        v_dequant = triton_dequantize_int8_per_token(qv, vs, dtype=v.dtype).repeat_interleave(2, dim=1)
        ref = scaled_dot_product_attention(q, k_dequant, v_dequant)
        torch.cuda.synchronize()
        self.assertEqual(fused.shape, q.shape)
        self.assertTrue(torch.allclose(fused, ref, atol=3e-2, rtol=3e-2))

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

    def test_fused_int8_paged_decode_attention_matches_dense_fused(self):
        q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        page_size = 4
        num_pages = 4
        page_table = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
        qk, ks = triton_quantize_int8_per_token(k)
        qv, vs = triton_quantize_int8_per_token(v)
        k_pages = torch.empty(1, 2, num_pages, page_size, 32, device="cuda", dtype=torch.int8)
        v_pages = torch.empty_like(k_pages)
        k_scales = torch.empty(1, 2, num_pages, page_size, 1, device="cuda", dtype=torch.float32)
        v_scales = torch.empty_like(k_scales)
        for logical_page, physical_page in enumerate(page_table.cpu().tolist()):
            start = logical_page * page_size
            end = start + page_size
            k_pages[:, :, physical_page] = qk[:, :, start:end]
            v_pages[:, :, physical_page] = qv[:, :, start:end]
            k_scales[:, :, physical_page] = ks[:, :, start:end]
            v_scales[:, :, physical_page] = vs[:, :, start:end]

        paged = triton_fused_int8_paged_kv_attention(
            q,
            k_pages,
            k_scales,
            v_pages,
            v_scales,
            page_table,
            length=16,
            page_size=page_size,
        )
        dense = triton_fused_int8_kv_attention(q, qk, ks, qv, vs)
        torch.cuda.synchronize()
        self.assertEqual(paged.shape, q.shape)
        self.assertTrue(torch.allclose(paged, dense, atol=3e-2, rtol=3e-2))

    def test_fused_uint8_paged_decode_attention_supports_batch_varlen(self):
        q = torch.randn(2, 4, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(2, 2, 16, 32, device="cuda", dtype=torch.float16)
        page_size = 4
        num_pages = 4
        page_table = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]], device="cuda", dtype=torch.int64)
        lengths = torch.tensor([16, 12], device="cuda", dtype=torch.int32)
        qk, ks, kz = triton_quantize_uint8_per_token(k)
        qv, vs, vz = triton_quantize_uint8_per_token(v)
        k_pages = torch.empty(2, 2, num_pages, page_size, 32, device="cuda", dtype=torch.uint8)
        v_pages = torch.empty_like(k_pages)
        k_scales = torch.empty(2, 2, num_pages, page_size, 1, device="cuda", dtype=torch.float32)
        v_scales = torch.empty_like(k_scales)
        k_zps = torch.empty(2, 2, num_pages, page_size, 1, device="cuda", dtype=torch.uint8)
        v_zps = torch.empty_like(k_zps)
        for b in range(2):
            for logical_page, physical_page in enumerate(page_table[b].cpu().tolist()):
                start = logical_page * page_size
                end = start + page_size
                k_pages[b, :, physical_page] = qk[b, :, start:end]
                v_pages[b, :, physical_page] = qv[b, :, start:end]
                k_scales[b, :, physical_page] = ks[b, :, start:end]
                v_scales[b, :, physical_page] = vs[b, :, start:end]
                k_zps[b, :, physical_page] = kz[b, :, start:end]
                v_zps[b, :, physical_page] = vz[b, :, start:end]

        paged = triton_fused_uint8_paged_kv_attention(
            q, k_pages, k_scales, k_zps, v_pages, v_scales, v_zps, page_table, lengths=lengths, page_size=page_size
        )
        refs = []
        for b, length in enumerate(lengths.cpu().tolist()):
            kd = triton_dequantize_uint8_per_token(
                qk[b : b + 1, :, :length], ks[b : b + 1, :, :length], kz[b : b + 1, :, :length], dtype=k.dtype
            ).repeat_interleave(2, dim=1)
            vd = triton_dequantize_uint8_per_token(
                qv[b : b + 1, :, :length], vs[b : b + 1, :, :length], vz[b : b + 1, :, :length], dtype=v.dtype
            ).repeat_interleave(2, dim=1)
            refs.append(scaled_dot_product_attention(q[b : b + 1], kd, vd))
        ref = torch.cat(refs, dim=0)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(paged, ref, atol=3e-2, rtol=3e-2))

    def test_fused_int8_paged_decode_streaming_fallback_matches_dense(self):
        q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        page_size = 4
        page_table = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int64)
        qk, ks = triton_quantize_int8_per_token(k)
        qv, vs = triton_quantize_int8_per_token(v)
        k_pages = qk.view(1, 2, 4, page_size, 32).contiguous()
        v_pages = qv.view(1, 2, 4, page_size, 32).contiguous()
        k_scales = ks.view(1, 2, 4, page_size, 1).contiguous()
        v_scales = vs.view(1, 2, 4, page_size, 1).contiguous()
        streaming = triton_fused_int8_paged_kv_attention(
            q, k_pages, k_scales, v_pages, v_scales, page_table, length=16, page_size=page_size, block_tokens=8
        )
        dense = triton_fused_int8_kv_attention(q, qk, ks, qv, vs)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(streaming, dense, atol=3e-2, rtol=3e-2))

    def test_quantized_kv_attention_triton_backend_cuda(self):
        q = torch.randn(1, 4, 1, 32, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        v = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.float16)
        out = quantized_kv_attention(q, k, v, quant_dtype="int8", backend="triton")
        torch.cuda.synchronize()
        self.assertEqual(out.shape, q.shape)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_fused_decode_benchmark_cuda_smoke(self):
        result = benchmark_fused_decode_attention(q_heads=4, kv_heads=2, tokens=16, head_dim=32, repeats=1)
        torch.cuda.synchronize()
        self.assertGreater(result.fused.mean_ms, 0.0)
        self.assertGreater(result.dequant_sdpa.mean_ms, 0.0)
        self.assertLess(result.quality.mean_abs_error, 0.05)

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
