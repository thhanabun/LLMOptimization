# Requires torch + transformers and a locally cached model.
# python examples/hf_suite_example.py Qwen/Qwen2.5-0.5B

import sys
from transformers import AutoTokenizer
from llm_memlab import BenchmarkConfig, benchmark_inference_suite, load_hf_model

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = load_hf_model(model_name, local_files_only=True)
encoded = tokenizer("Hello", return_tensors="pt")
result = benchmark_inference_suite(model, encoded, model_name=model_name, max_new_tokens=8, config=BenchmarkConfig(warmup=1, repeats=1))
print(result.to_text())
