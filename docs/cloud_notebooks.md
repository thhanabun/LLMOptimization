# Cloud Notebook Examples

These examples are intentionally conservative. Notebook platforms change often, so treat this page as a portable starter recipe and validate the runtime with `backend-demo` before trusting GPU paths.

## Ready-To-Run Notebooks

- [Colab quickstart notebook](../notebooks/colab_quickstart.ipynb)
- [Kaggle quickstart notebook](../notebooks/kaggle_quickstart.ipynb)

Both notebooks call `examples/cloud_smoke.py`, which runs:

- `backend-demo`
- `estimate`
- optional `memory-first-hf-bench`
- optional `serving-bench`
- HTML dashboard export

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

Download a model from Hugging Face into the runtime and point `LLM_MEMLAB_MODEL` at the local folder. The default model is intentionally tiny so the cell is fast and cheap; replace `HF_MODEL_ID` with a larger model when the runtime can handle it.

```python
import os
from huggingface_hub import snapshot_download

os.environ.setdefault("LLM_MEMLAB_MODEL_ROOT", "/content/hf_models")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "hf-internal-testing/tiny-random-LlamaForCausalLM")
model_path = snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=os.path.join(os.environ["LLM_MEMLAB_MODEL_ROOT"], HF_MODEL_ID.replace("/", "__")),
    token=os.environ.get("HF_TOKEN"),
)
os.environ["LLM_MEMLAB_MODEL"] = model_path
```

Run the portable smoke script:

```python
!python examples/cloud_smoke.py --tokens 1
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

Kaggle datasets are commonly mounted under `/kaggle/input`, but Hugging Face downloads should go to `/kaggle/working` because `/kaggle/input` is read-only:

```python
import os
os.environ["LLM_MEMLAB_MODEL_ROOT"] = "/kaggle/working/hf_models"
```

Download a model from Hugging Face. For gated/private models, create a Kaggle secret named `HF_TOKEN` and uncomment the secret block:

```python
# from kaggle_secrets import UserSecretsClient
# os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

from huggingface_hub import snapshot_download

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "hf-internal-testing/tiny-random-LlamaForCausalLM")
model_path = snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=os.path.join(os.environ["LLM_MEMLAB_MODEL_ROOT"], HF_MODEL_ID.replace("/", "__")),
    token=os.environ.get("HF_TOKEN"),
)
os.environ["LLM_MEMLAB_MODEL"] = model_path
```

Then run the harness and smoke workflow:

```python
!python -m llm_memlab local-model-harness --root "$LLM_MEMLAB_MODEL_ROOT" --json-out /kaggle/working/local_model_fixtures.json
!python -m llm_memlab backend-demo
!python examples/cloud_smoke.py --tokens 1
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
