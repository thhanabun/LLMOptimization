from llm_memlab import BenchmarkRecord, write_benchmark_csv, write_benchmark_json
from llm_memlab.benchmark_dashboard import write_benchmark_dashboard_html


if __name__ == "__main__":
    records = [
        BenchmarkRecord(
            "decode",
            "kernel",
            1.25,
            1.10,
            1.40,
            peak_cuda_bytes=128 * 1024 * 1024,
            extra={
                "model": "tiny-llama",
                "quality_passed": True,
                "mean_abs": 0.001,
                "first_token_ms": 12.5,
                "tokens_per_second": 42.0,
            },
            metadata={"gpu": "example-gpu", "backend": "triton-experimental", "commit": "local"},
        )
    ]
    write_benchmark_json(records, "example_benchmark.json")
    write_benchmark_csv(records, "example_benchmark.csv")
    path = write_benchmark_dashboard_html(["example_benchmark.json", "example_benchmark.csv"], "example_dashboard.html")
    print(f"Dashboard written to {path}")
