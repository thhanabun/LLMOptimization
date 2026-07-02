import os
from pathlib import Path

from llm_memlab.production import benchmark_serving_paths


def model_root() -> Path:
    configured = os.environ.get("LLM_MEMLAB_MODEL_ROOT")
    if configured:
        return Path(configured)
    return Path("./models")


if __name__ == "__main__":
    model = model_root() / "TinyLlama-1.1B-Chat-v1.0"
    result = benchmark_serving_paths(
        str(model),
        prompt="hello",
        max_new_tokens=1,
        device="cpu",
        dtype="fp32",
        local_files_only=True,
        include_vllm=True,
    )
    result.write_json("serving_bench.json")
    result.write_csv("serving_bench.csv")
    result.write_html("serving_dashboard.html")
    print(result.to_text())
