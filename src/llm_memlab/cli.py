from __future__ import annotations

import argparse

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
