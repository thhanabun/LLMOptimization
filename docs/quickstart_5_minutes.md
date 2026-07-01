# Use llm-memlab In 5 Minutes

Install the wheel or editable package:

```bash
python -m pip install llm-memlab
# or from a checkout:
python -m pip install -e .[dev]
```

Check the stable runtime surface:

```bash
python -c "from llm_memlab.production import PRODUCTION_API_VERSION, STABILITY_POLICY; print(PRODUCTION_API_VERSION); print(STABILITY_POLICY)"
llm-memlab backend-demo
```

Export the built-in cache profiles and ask the policy why it chooses a backend:

```bash
llm-memlab profile export --out profiles.json
llm-memlab profile explain --family llama --model TinyLlama --prompt-tokens 32 --profile profiles.json
```

Run a local-only real-model certification matrix. If the models are not present locally, the command still writes conservative fallback profiles instead of fetching from the network:

```bash
llm-memlab certify-model-matrix --local-root ./models --profiles-out profiles.json --json-out matrix.json
```

Build a benchmark dashboard from JSON/CSV history:

```bash
python examples/benchmark_dashboard_example.py
llm-memlab benchmark-dashboard --inputs example_benchmark.json example_benchmark.csv --out example_dashboard.html
```

On a CUDA machine, run the kernel promotion gate. It keeps Triton/CuTile experimental unless the certification matrix is strong enough:

```bash
llm-memlab kernel-promote --backend triton --quick
```

