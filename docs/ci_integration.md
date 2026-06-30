# How To Integrate In CI

Install the package with development tools:

```bash
python -m pip install -e .[dev]
```

Run the stable checks:

```bash
python -m ruff check src/llm_memlab/production.py src/llm_memlab/experimental.py src/llm_memlab/kernel_certification.py tests/test_hf_integration_matrix.py tests/test_kernel_certification.py
python -m ruff format --check src/llm_memlab/production.py src/llm_memlab/experimental.py src/llm_memlab/kernel_certification.py tests/test_hf_integration_matrix.py tests/test_kernel_certification.py
python -m mypy src/llm_memlab/production.py src/llm_memlab/experimental.py src/llm_memlab/kernel_certification.py
python -m unittest discover -s tests
```

Run optional profile-specific checks:

```bash
python -m unittest discover -s tests -p test_cuda_triton.py
python -m unittest discover -s tests -p test_hf_integration_matrix.py
python -m unittest discover -s tests -p test_kernel_certification.py
```

Run real-model cache certification when a local model is available:

```bash
python -m llm_memlab hf-cache-certify \
  --model /models/qwen3 \
  --local-files-only \
  --prompts "hello|Explain KV cache briefly." \
  --caches paged \
  --experimental-caches quantized \
  --json-out hf_cache_cert.json \
  --csv-out hf_cache_cert.csv \
  --html-out hf_cache_cert.html \
  --fail-on-regression
```

Production cache paths fail CI on drift. Experimental cache paths are recorded in JSON/CSV/HTML but do not fail the production gate.

Example benchmark gate:

```python
from llm_memlab.production import BenchmarkGateConfig, BenchmarkRecord, assert_no_regressions, benchmark_gate

baseline = [BenchmarkRecord("decode", "decode", 10.0, 9.5, 11.0, extra={"quality_passed": True, "mean_abs": 0.0})]
candidate = [BenchmarkRecord("decode", "decode", 9.0, 8.8, 9.5, extra={"quality_passed": True, "mean_abs": 0.0})]
result = benchmark_gate(baseline, candidate, BenchmarkGateConfig(max_slowdown_pct=10.0, max_quality_mean_abs=0.01))
assert_no_regressions(result)
```

Wheel smoke test:

```bash
python -m build
python -m venv .venv-wheel
. .venv-wheel/bin/activate
python -m pip install dist/*.whl
python -c "from llm_memlab.production import KernelPolicy; from llm_memlab.experimental import experimental_kernel_policy; print(KernelPolicy(), experimental_kernel_policy())"
```
