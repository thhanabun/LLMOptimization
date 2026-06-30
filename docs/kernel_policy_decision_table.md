# Kernel Policy Decision Table

`select_kernel_policy()` is intentionally conservative.

| Condition | Selection | Stability |
| --- | --- | --- |
| CUDA unavailable | `torch`, `dequant+sdpa` | stable |
| Triton unavailable | `torch`, `dequant+sdpa` | stable |
| `backend="torch"` | `torch`, `dequant+sdpa` | stable |
| invalid GQA shape (`q_heads % kv_heads != 0`) | `triton`, `quant-dequant+sdpa` | stable |
| unsupported quant dtype for fused decode | `triton`, `quant-dequant+sdpa` | stable |
| dense decode under configured limits | `triton`, `fused-decode` | stable |
| paged requested without experimental opt-in | `triton`, `quant-dequant+sdpa` | stable |
| paged requested with `allow_experimental=True` | `triton-experimental`, `paged-fused-decode` | experimental |
| long dense context over configured limit | fallback unless explicit paged experimental | stable fallback |

Production guidance: keep the default policy for application code, then explicitly allow experimental kernels only for certified shapes.
