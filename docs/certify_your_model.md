# How To Certify Your Model

Certification answers one production question: which cache/backend path is safe for this model on this machine?

## 1. Start With Runtime Detection

```bash
llm-memlab backend-demo
```

Read the fallback reasons. A missing backend is not a failure by itself; it tells the policy which path to avoid.

## 2. Point To Local Models

Use environment variables so scripts work across Windows, Colab, Kaggle, and Linux GPU hosts.

```bash
export LLM_MEMLAB_MODEL_ROOT=/models
export LLM_MEMLAB_MODEL=/models/TinyLlama-1.1B-Chat-v1.0
```

PowerShell:

```powershell
$env:LLM_MEMLAB_MODEL_ROOT = "D:\hf_models"
$env:LLM_MEMLAB_MODEL = "D:\hf_models\TinyLlama-1.1B-Chat-v1.0"
```

## 3. Scan Available Fixtures

```bash
llm-memlab local-model-harness --root "$LLM_MEMLAB_MODEL_ROOT" --json-out local_model_fixtures.json
```

This does not download models. It tells you which known small model-family fixtures are present.

## 4. Run A Conservative Cache Certification

```bash
llm-memlab hf-cache-certify \
  --model "$LLM_MEMLAB_MODEL" \
  --local-files-only \
  --prompts "hello|Explain KV cache briefly." \
  --tokens 1 \
  --caches paged \
  --device auto \
  --dtype auto \
  --json-out hf_cache_cert.json \
  --csv-out hf_cache_cert.csv \
  --html-out hf_cache_cert.html \
  --fail-on-regression
```

Start with `paged`. Add `quantized` only after paged passes:

```bash
llm-memlab hf-cache-certify \
  --model "$LLM_MEMLAB_MODEL" \
  --local-files-only \
  --tokens 1 \
  --caches paged,quantized \
  --experimental-caches quantized \
  --quant-dtypes int8,uint8 \
  --device auto \
  --dtype auto \
  --html-out hf_cache_cert_quantized.html
```

## 5. Build A Model Matrix

```bash
llm-memlab certify-model-matrix \
  --models llama="$LLM_MEMLAB_MODEL" \
  --prompt hello \
  --device auto \
  --dtype auto \
  --json-out model_matrix.json \
  --profiles-out profiles.json \
  --min-certified-models 1
```

Use `profiles.json` as policy evidence in CI and future benchmark runs.

## 6. Gate Benchmarks And Quality

```bash
llm-memlab memory-first-hf-bench \
  --model "$LLM_MEMLAB_MODEL" \
  --local-files-only \
  --tokens 1 \
  --device auto \
  --dtype auto \
  --cache paged \
  --json-out memory_first_hf.json \
  --csv-out memory_first_hf.csv \
  --fail-on-regression
```

## Production Rule

Promote only the paths that pass on your real model, real GPU, real dtype, real sequence length, and real serving environment. Leave everything else as fallback or experimental.
