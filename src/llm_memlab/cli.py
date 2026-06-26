from __future__ import annotations

import argparse
import time

from .estimates import TransformerConfig, estimate_transformer_memory, preset_config
from .ir import GraphSpec, OperationSpec, TensorSpec
from .planner import MemoryPlanner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-memlab", description="Memory-first LLM analysis toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser("estimate", help="Estimate LLM memory use.")
    estimate_parser.add_argument("--preset", default="7b-like", help="tiny, 1b-like, 3b-like, 7b-like, 13b-like")
    estimate_parser.add_argument("--layers", type=int)
    estimate_parser.add_argument("--hidden", type=int)
    estimate_parser.add_argument("--intermediate", type=int)
    estimate_parser.add_argument("--heads", type=int)
    estimate_parser.add_argument("--kv-heads", type=int)
    estimate_parser.add_argument("--vocab", type=int)
    estimate_parser.add_argument("--seq", type=int, default=2048)
    estimate_parser.add_argument("--batch", type=int, default=1)
    estimate_parser.add_argument("--dtype", default="bf16")
    estimate_parser.add_argument("--training", choices=["inference", "lora", "full"], default="inference")
    estimate_parser.add_argument("--optimizer", default="adamw")
    estimate_parser.add_argument("--checkpointing", choices=["none", "selective", "full"], default="none")
    estimate_parser.add_argument("--lora-rank", type=int, default=0)
    estimate_parser.add_argument("--no-flash-attention", action="store_true")
    estimate_parser.add_argument("--untied-embeddings", action="store_true")
    estimate_parser.set_defaults(func=_estimate)

    plan_parser = subparsers.add_parser("plan-demo", help="Show tensor-lifetime planning on a toy transformer block.")
    plan_parser.add_argument("--seq", type=int, default=1024)
    plan_parser.add_argument("--hidden", type=int, default=4096)
    plan_parser.add_argument("--intermediate", type=int, default=11008)
    plan_parser.add_argument("--dtype", default="bf16")
    plan_parser.set_defaults(func=_plan_demo)

    trace_parser = subparsers.add_parser("trace-demo", help="Trace a tiny PyTorch model if torch is installed.")
    trace_parser.set_defaults(func=_trace_demo)

    kernel_parser = subparsers.add_parser("kernel-demo", help="Run correctness checks and microbenchmarks for optimized kernels.")
    kernel_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    kernel_parser.add_argument("--compile", action="store_true", help="Use torch.compile where available.")
    kernel_parser.add_argument("--repeats", type=int, default=20)
    kernel_parser.set_defaults(func=_kernel_demo)

    args = parser.parse_args(argv)
    return args.func(args)


def _estimate(args: argparse.Namespace) -> int:
    if all(value is not None for value in (args.layers, args.hidden, args.intermediate, args.heads, args.vocab)):
        cfg = TransformerConfig(
            num_layers=args.layers,
            hidden_size=args.hidden,
            intermediate_size=args.intermediate,
            num_attention_heads=args.heads,
            vocab_size=args.vocab,
            sequence_length=args.seq,
            batch_size=args.batch,
            dtype=args.dtype,
        )
    else:
        cfg = preset_config(args.preset, sequence_length=args.seq, batch_size=args.batch, dtype=args.dtype)
    cfg = TransformerConfig(
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        sequence_length=cfg.sequence_length,
        batch_size=cfg.batch_size,
        dtype=cfg.dtype,
        training=args.training,
        optimizer=args.optimizer,
        activation_checkpointing=args.checkpointing,
        use_flash_attention=not args.no_flash_attention,
        tie_embeddings=not args.untied_embeddings,
        num_key_value_heads=args.kv_heads or cfg.num_key_value_heads,
        lora_rank=args.lora_rank,
    )
    print(estimate_transformer_memory(cfg).to_text())
    return 0


def _plan_demo(args: argparse.Namespace) -> int:
    graph = _toy_block_graph(args.seq, args.hidden, args.intermediate, args.dtype)
    plan = MemoryPlanner(graph.tensor_lifetimes()).plan()
    print(plan.to_text())
    return 0


def _trace_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run trace-demo: pip install torch")
        return 2

    from .torch_debugger import TorchTrace

    model = torch.nn.Sequential(
        torch.nn.Linear(16, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 16),
    )
    x = torch.randn(8, 16)
    with TorchTrace(model) as trace:
        _ = model(x)
    print(trace.to_text())
    return 0


def _kernel_demo(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it to run kernel-demo: pip install torch")
        return 2

    from .kernels import KernelConfig, chunked_cross_entropy, kernel
    from .report import make_table

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available.")
        return 2
    dtype = torch.float16 if device == "cuda" else torch.float32
    cfg = KernelConfig(compile=args.compile)

    rms = kernel("rms_norm", cfg)
    sdpa = kernel("scaled_dot_product_attention", cfg)
    swiglu_kernel = kernel("swiglu", cfg)

    x = torch.randn(2, 128, 512, device=device, dtype=dtype)
    weight = torch.randn(512, device=device, dtype=dtype)
    gate = torch.randn(1536, 512, device=device, dtype=dtype)
    up = torch.randn(1536, 512, device=device, dtype=dtype)
    down = torch.randn(512, 1536, device=device, dtype=dtype)
    q = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    k = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    v = torch.randn(2, 8, 128, 64, device=device, dtype=dtype)
    logits = torch.randn(2, 128, 2048, device=device, dtype=dtype)
    targets = torch.randint(0, 2048, (2, 128), device=device)

    checks = []
    checks.append(("rms_norm", tuple(rms(x, weight).shape)))
    checks.append(("swiglu", tuple(swiglu_kernel(x, gate, up, down).shape)))
    checks.append(("sdpa", tuple(sdpa(q, k, v, is_causal=True).shape)))
    checks.append(("chunked_cross_entropy", tuple(chunked_cross_entropy(logits, targets, chunk_size=64).shape)))

    rows = []
    for name, fn in [
        ("rms_norm", lambda: rms(x, weight)),
        ("swiglu", lambda: swiglu_kernel(x, gate, up, down)),
        ("sdpa", lambda: sdpa(q, k, v, is_causal=True)),
        ("chunked_cross_entropy", lambda: chunked_cross_entropy(logits, targets, chunk_size=64)),
    ]:
        rows.append((name, f"{_bench(torch, fn, args.repeats):.3f} ms"))

    print(f"device={device} dtype={dtype} compile={args.compile}")
    print(make_table(("Kernel", "Output shape"), checks))
    print("")
    print(make_table(("Kernel", "Avg time"), rows))
    return 0


def _bench(torch, fn, repeats: int) -> float:
    repeats = max(1, repeats)
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / repeats


def _toy_block_graph(seq: int, hidden: int, intermediate: int, dtype: str) -> GraphSpec:
    graph = GraphSpec(inputs=("x",), outputs=("out",))
    for tensor in [
        TensorSpec.from_shape("x", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("qkv", (1, seq, hidden * 3), dtype=dtype, role="activation"),
        TensorSpec.from_shape("attn", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("mlp_up", (1, seq, intermediate * 2), dtype=dtype, role="activation"),
        TensorSpec.from_shape("mlp_down", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("out", (1, seq, hidden), dtype=dtype, role="activation"),
        TensorSpec.from_shape("w_qkv", (hidden, hidden * 3), dtype=dtype, role="parameter"),
        TensorSpec.from_shape("w_mlp_up", (hidden, intermediate * 2), dtype=dtype, role="parameter"),
        TensorSpec.from_shape("w_mlp_down", (intermediate, hidden), dtype=dtype, role="parameter"),
    ]:
        graph.add_tensor(tensor)
    graph.add_op(OperationSpec.make("qkv_proj", "linear", ("x", "w_qkv"), ("qkv",)))
    graph.add_op(OperationSpec.make("attention", "flash_attention", ("qkv",), ("attn",)))
    graph.add_op(OperationSpec.make("mlp_up", "swiglu_up", ("attn", "w_mlp_up"), ("mlp_up",)))
    graph.add_op(OperationSpec.make("mlp_down", "linear", ("mlp_up", "w_mlp_down"), ("mlp_down",)))
    graph.add_op(OperationSpec.make("residual", "add", ("x", "mlp_down"), ("out",)))
    return graph
