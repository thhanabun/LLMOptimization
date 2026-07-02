# How To Compare With vLLM

vLLM is a serving engine. llm-memlab is a debugger, certifier, and memory-first lab. Compare them when you need a serving baseline, not because they solve the same full problem.

## 1. Detect Whether vLLM Is Usable

```bash
llm-memlab backend-demo
```

Look for `vllm-serving`.

- `Available=True`: the optional vLLM path can run in this environment.
- `Available=False`: read the fallback reason. Native Windows commonly falls back; Linux, WSL, Docker, or cloud GPU hosts are better serving targets.

## 2. Run The Three-Path Benchmark

```bash
llm-memlab serving-bench \
  --model "$LLM_MEMLAB_MODEL" \
  --local-files-only \
  --prompt hello \
  --tokens 1 \
  --device auto \
  --dtype auto \
  --cache paged \
  --json-out serving_bench.json \
  --csv-out serving_bench.csv \
  --html-out serving_dashboard.html
```

The three paths are:

- `hf-generate`: Hugging Face baseline.
- `memory-first-hf`: llm-memlab conservative HF adapter path.
- `vllm`: optional `vllm-serving` baseline when installed and compatible.

## 3. Read The Metrics

The dashboard records:

- first-token latency
- tokens/sec
- peak CUDA memory
- backend selected
- prefix-cache flag when available
- token-match quality
- fallback reason

## 4. Keep The Comparison Honest

Use identical:

- model revision/path
- prompt
- max new tokens
- dtype
- device/GPU
- local cache/model files

Do not compare Windows HF fallback numbers against Linux vLLM serving numbers as if they were one environment. Store environment metadata with every benchmark run.

## 5. CI Gate Example

```bash
llm-memlab benchmark-dashboard \
  --inputs serving_bench.json serving_bench.csv \
  --out serving_history_dashboard.html \
  --title "llm-memlab serving comparison"
```

For production, keep vLLM's native metrics enabled too. Use llm-memlab as certification evidence and vLLM as the serving engine.
