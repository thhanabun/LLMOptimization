"""Run a small HF inference suite from an env-configured model path.

Examples:
  LLM_MEMLAB_MODEL=/models/TinyLlama-1.1B-Chat-v1.0 python examples/hf_suite_example.py
  python examples/hf_suite_example.py /models/TinyLlama-1.1B-Chat-v1.0
"""

import os
import sys

from transformers import AutoTokenizer

from llm_memlab import BenchmarkConfig, benchmark_inference_suite, load_hf_model


model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("LLM_MEMLAB_MODEL")
if not model_name:
    raise SystemExit("Set LLM_MEMLAB_MODEL or pass a local model path as argv[1].")

tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = load_hf_model(model_name, local_files_only=True)
encoded = tokenizer(os.environ.get("LLM_MEMLAB_PROMPT", "Hello"), return_tensors="pt")
tokens = int(os.environ.get("LLM_MEMLAB_TOKENS", "8"))
result = benchmark_inference_suite(model, encoded, model_name=model_name, max_new_tokens=tokens, config=BenchmarkConfig(warmup=1, repeats=1))
print(result.to_text())
