# vLLM vs llm-memlab

vLLM and llm-memlab are complementary tools with different production jobs.

vLLM is a serving engine. Use it when the goal is high-throughput online or offline inference, request scheduling, paged attention, prefix caching, and OpenAI-compatible serving.

llm-memlab is a memory-first lab, certifier, and debugger. Use it to inspect model architecture, validate cache/kernel quality, compare optimized paths against Hugging Face, capture memory timelines, and decide whether a backend is safe for your model/hardware.

## Recommended Production Flow

1. Use plain Hugging Face generate as the correctness baseline.
2. Use `llm-memlab memory-first-hf` or `serving-bench` to certify token match, quality drift, and peak memory.
3. Use vLLM as a serving baseline when it is available on the target host.
4. Keep llm-memlab in CI to catch regressions before changing cache policy or serving backend.

## CLI Baseline

```powershell
$env:PYTHONPATH = "src"
$env:LLM_MEMLAB_MODEL_ROOT = "D:\hf_models"
python -m llm_memlab serving-bench --model "$env:LLM_MEMLAB_MODEL_ROOT\TinyLlama-1.1B-Chat-v1.0" --local-files-only --tokens 1 --device cpu --dtype fp32 --json-out serving_bench.json --csv-out serving_bench.csv --html-out serving_dashboard.html
```

The `vllm-serving` backend is conservative by default:

- If vLLM is not installed, it reports a fallback reason.
- If CUDA is unavailable, it falls back to HF paths.
- On native Windows, it reports a compatibility fallback and recommends Linux, WSL, or Docker for vLLM serving.

## Backend Names

- `hf`: baseline Hugging Face generate.
- `llm-memlab`: memory-first Hugging Face adapter path.
- `vllm-serving`: optional vLLM serving/offline engine baseline.

## What The Dashboard Shows

The serving dashboard uses the same benchmark database schema as the rest of llm-memlab and adds:

- first-token latency
- tokens/sec
- peak CUDA memory
- prefix-cache flag when vLLM exposes it
- backend selected
- token-match quality
- fallback reason

## Limitations

`vllm-serving` is a baseline adapter, not a replacement for vLLM's own production server metrics. For fleet serving, keep vLLM's native metrics enabled and use llm-memlab artifacts as CI/debugger evidence.
