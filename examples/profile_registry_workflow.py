import os
from pathlib import Path

from llm_memlab.production import (
    ModelCertificationTarget,
    certify_model_matrix,
    select_hf_cache_policy,
)
from llm_memlab.hf_cache_policy import HFCachePolicy


def model_root() -> Path:
    configured = os.environ.get("LLM_MEMLAB_MODEL_ROOT")
    if configured:
        return Path(configured)
    return Path(r"D:\hf_models" if os.name == "nt" else "./models")


if __name__ == "__main__":
    target = ModelCertificationTarget(
        "local-llama",
        "llama",
        str(model_root() / "TinyLlama-1.1B-Chat-v1.0"),
        local_files_only=True,
    )
    report = certify_model_matrix([target], prompts=("hello",), allow_remote=False)
    report.write_json("example_matrix.json")
    report.write_profiles("example_profiles.json")
    print(report.to_text())

    decision = select_hf_cache_policy(
        family="llama",
        prompt_tokens=32,
        policy=HFCachePolicy(requested_cache="quantized", quantized_profile_paths=("example_profiles.json",)),
    )
    print("")
    print(decision.to_text())
