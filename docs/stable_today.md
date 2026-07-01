# What Is Stable Today

Use `llm_memlab.production` as the stable API surface for the current `0.x` line. The production path is conservative: it prefers a slower Torch fallback over a fast path that has not been certified for the current model, dtype, shape, and GPU.

## Stable For Production Use

- Memory estimation for inference, LoRA, and full fine-tuning.
- Tensor lifetime planning and buffer reuse reports.
- PyTorch module tracing, layer stats, attention stats, drift reports, and HTML debug reports.
- Benchmark JSON/CSV storage, compare gates, benchmark dashboard, and quality regression checks.
- Hardware profile detection and backend registry.
- HF adapter selection and cache policy explanation with conservative fallback.
- Profile registry workflows: `profile export`, `profile merge`, `profile explain`.
- Real-model certification matrix when run with strict gates:

```bash
llm-memlab certify-model-matrix \
  --local-root ./models \
  --require-real-models \
  --min-certified-models 2 \
  --strict \
  --json-out matrix.json \
  --profiles-out profiles.json
```

## Stable CI Pattern

```bash
python -m unittest discover -s tests
llm-memlab local-model-harness --root ./models --json-out local_models.json
llm-memlab certify-model-matrix --local-root ./models --require-real-models --min-certified-models 1 --json-out matrix.json --profiles-out profiles.json
llm-memlab benchmark-compare --baseline baseline.json --candidate candidate.json --fail-on-regression
llm-memlab benchmark-dashboard --inputs baseline.json candidate.json --out dashboard.html
```

## Production Rule Of Thumb

An optimized path is production only after it has:

- correctness certification,
- quality drift metrics,
- benchmark history,
- hardware metadata,
- fallback behavior,
- and a profile or promotion record explaining why it was selected.

