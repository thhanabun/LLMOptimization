from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from llm_memlab import BenchmarkRecord, write_benchmark_csv, write_benchmark_dashboard_html, write_benchmark_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Portable llm-memlab smoke workflow for Colab, Kaggle, and cloud GPUs.")
    parser.add_argument("--model", default=os.environ.get("LLM_MEMLAB_MODEL"), help="Optional local HF model path/name.")
    parser.add_argument("--model-root", default=os.environ.get("LLM_MEMLAB_MODEL_ROOT", "./models"))
    parser.add_argument("--prompt", default=os.environ.get("LLM_MEMLAB_PROMPT", "hello"))
    parser.add_argument("--tokens", type=int, default=int(os.environ.get("LLM_MEMLAB_TOKENS", "1")))
    parser.add_argument("--device", default=os.environ.get("LLM_MEMLAB_DEVICE", "auto"))
    parser.add_argument("--dtype", default=os.environ.get("LLM_MEMLAB_DTYPE", "auto"))
    parser.add_argument("--cache", choices=["paged", "quantized"], default=os.environ.get("LLM_MEMLAB_CACHE", "paged"))
    parser.add_argument("--out-dir", default=os.environ.get("LLM_MEMLAB_OUT_DIR", "llm_memlab_outputs"))
    parser.add_argument("--local-files-only", action="store_true", default=os.environ.get("LLM_MEMLAB_LOCAL_ONLY", "1") != "0")
    parser.add_argument("--skip-model-bench", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _resolve_model(args.model, args.model_root)

    _run(["backend-demo"])
    _run(["estimate", "--preset", "7b-like", "--seq", "2048", "--batch", "1", "--training", "inference"])

    records = _write_example_dashboard(out_dir)
    if model is None or args.skip_model_bench:
        reason = "no model path was provided or discovered" if model is None else "--skip-model-bench was set"
        print(f"Skipping HF model benchmarks: {reason}. Example dashboard written from synthetic records.")
    else:
        common = ["--model", str(model), "--prompt", args.prompt, "--tokens", str(args.tokens), "--device", args.device, "--dtype", args.dtype]
        if args.local_files_only:
            common.append("--local-files-only")
        benchmark_inputs: list[Path] = []
        memory_rc = _run(
            [
                "memory-first-hf-bench",
                *common,
                "--cache",
                args.cache,
                "--json-out",
                str(out_dir / "memory_first_hf.json"),
                "--csv-out",
                str(out_dir / "memory_first_hf.csv"),
            ],
            check=False,
        )
        if memory_rc == 0:
            benchmark_inputs.append(out_dir / "memory_first_hf.json")
        else:
            benchmark_inputs.append(_write_failure_dashboard(out_dir, "memory-first-hf-bench", memory_rc))
            print(f"memory-first-hf-bench failed with exit code {memory_rc}; continuing with remaining smoke steps.")
        serving_rc = _run(
            [
                "serving-bench",
                *common,
                "--cache",
                args.cache,
                "--json-out",
                str(out_dir / "serving_bench.json"),
                "--csv-out",
                str(out_dir / "serving_bench.csv"),
                "--html-out",
                str(out_dir / "serving_dashboard.html"),
            ],
            check=False,
        )
        if serving_rc == 0:
            benchmark_inputs.append(out_dir / "serving_bench.json")
        else:
            benchmark_inputs.append(_write_failure_dashboard(out_dir, "serving-bench", serving_rc))
            print(f"serving-bench failed with exit code {serving_rc}; keeping fallback dashboard.")
        if benchmark_inputs:
            _run(["benchmark-dashboard", "--inputs", *(str(path) for path in benchmark_inputs), "--out", str(out_dir / "cloud_dashboard.html")])
            records = benchmark_inputs

    print(f"Artifacts written under {out_dir.resolve()}")
    print("Dashboard inputs:", ", ".join(str(path) for path in records))
    return 0


def _run(args: list[str], *, check: bool = True) -> int:
    command = [sys.executable, "-m", "llm_memlab", *args]
    print("+", " ".join(command))
    completed = subprocess.run(command, check=False)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _resolve_model(model: str | None, model_root: str) -> Path | None:
    if model:
        return Path(model)
    root = Path(model_root)
    candidates = (
        "TinyLlama-1.1B-Chat-v1.0",
        "Qwen3-1.7B",
        "Qwen2.5-0.5B-Instruct",
        "gemma-2-2b-it",
    )
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _write_example_dashboard(out_dir: Path) -> list[Path]:
    records = [
        BenchmarkRecord(
            "cloud-smoke/backend-demo",
            "smoke",
            0.0,
            0.0,
            0.0,
            extra={"model": "none", "quality_passed": True, "first_token_ms": None, "tokens_per_second": None},
            metadata={"backend": "torch-or-fallback", "gpu": "runtime-dependent", "commit": "local"},
        )
    ]
    json_path = write_benchmark_json(records, out_dir / "cloud_smoke.json")
    csv_path = write_benchmark_csv(records, out_dir / "cloud_smoke.csv")
    write_benchmark_dashboard_html([json_path, csv_path], out_dir / "cloud_dashboard.html", title="llm-memlab cloud smoke dashboard")
    return [json_path, csv_path]


def _write_failure_dashboard(out_dir: Path, name: str, returncode: int) -> Path:
    record = BenchmarkRecord(
        f"cloud-smoke/{name}",
        "smoke-failure",
        0.0,
        0.0,
        0.0,
        extra={"model": "runtime-model", "quality_passed": False, "fallback_reason": f"command failed with exit code {returncode}"},
        metadata={"backend": "hf-or-serving", "gpu": "runtime-dependent", "commit": "local"},
    )
    path = out_dir / f"{name.replace('-', '_')}_failure.json"
    write_benchmark_json([record], path)
    write_benchmark_dashboard_html([path], out_dir / "cloud_dashboard.html", title="llm-memlab cloud smoke dashboard")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
