# Cloud Notebook Examples

These examples are intentionally conservative. Notebook platforms change often, so treat this page as a portable starter recipe and validate the runtime with `backend-demo` before trusting GPU paths.

## Google Colab

```python
!git clone https://github.com/thhanabun/LLMOptimization.git
%cd LLMOptimization
!python -m pip install -e .[torch,transformers]
```

Check the runtime:

```python
!python -m llm_memlab backend-demo
!python -m llm_memlab estimate --preset 7b-like --seq 2048 --batch 1 --training inference
```

Run a small memory-first HF smoke if you have a local or downloaded model:

```python
MODEL = "/content/hf_models/TinyLlama-1.1B-Chat-v1.0"
!python -m llm_memlab serving-bench --model "$MODEL" --local-files-only --prompt hello --tokens 1 --device auto --dtype auto --cache paged --json-out serving_bench.json --csv-out serving_bench.csv --html-out serving_dashboard.html
```

Notes:

- Use `device auto` first; Colab GPU availability depends on the selected runtime.
- Keep vLLM as an optional baseline. If vLLM is unavailable, `vllm-serving` should report a fallback reason.
- For free-tier runtimes, prefer tiny models and one-token smoke tests before full certification.

## Kaggle Notebooks

```python
!git clone https://github.com/thhanabun/LLMOptimization.git
%cd LLMOptimization
!python -m pip install -e .[torch,transformers]
```

Kaggle datasets are commonly mounted under `/kaggle/input`. Use an environment variable instead of hardcoding model paths:

```python
import os
os.environ["LLM_MEMLAB_MODEL_ROOT"] = "/kaggle/input/hf-models"
```

```python
!python -m llm_memlab local-model-harness --root "$LLM_MEMLAB_MODEL_ROOT" --json-out local_model_fixtures.json
!python -m llm_memlab backend-demo
```

If a compatible model folder exists:

```python
MODEL = "/kaggle/input/hf-models/TinyLlama-1.1B-Chat-v1.0"
!python -m llm_memlab memory-first-hf-bench --model "$MODEL" --local-files-only --tokens 1 --device auto --dtype auto --cache paged --json-out tiny_bench.json --csv-out tiny_bench.csv
```

Notes:

- Kaggle internet access and GPU availability depend on notebook settings.
- Write generated reports to `/kaggle/working` if you want to download them.
- Prefer `local-files-only` when using dataset-mounted models.

## RunPod, Paperspace, Lambda, Vast, Or Local Linux GPU

```bash
git clone https://github.com/thhanabun/LLMOptimization.git
cd LLMOptimization
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch,transformers,triton]"
python -m llm_memlab backend-demo
```

Optional vLLM baseline:

```bash
python -m pip install vllm
python -m llm_memlab backend-demo
python -m llm_memlab serving-bench --model /models/TinyLlama-1.1B-Chat-v1.0 --local-files-only --tokens 1 --device cuda --dtype fp16 --cache paged --json-out serving_bench.json --csv-out serving_bench.csv --html-out serving_dashboard.html
```

Notes:

- Validate vLLM on the same container/driver/CUDA stack you plan to use.
- Keep `serving-bench` output as CI evidence before promoting a serving backend.
- Use `certify-env` or `certify-model-matrix` for model/hardware-specific profiles.

## GitHub Codespaces

Codespaces is useful for docs, CPU tests, and API work. Do not assume GPU availability.

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests examples
python -m unittest discover -s tests
python -m llm_memlab backend-demo
```

## Portable Rules

- Start with `backend-demo`.
- Use `local-model-harness` to avoid hardcoded model paths.
- Keep generated JSON/CSV/HTML artifacts out of source commits unless they are intentional baselines.
- Treat experimental kernels and serving backends as opt-in until certified on the target hardware.
