# Release Checklist

Use this checklist before tagging a release.

## Required For v0.1.x

```powershell
$env:PYTHONPATH = "src"
python -m ruff check src tests examples
python -m unittest discover -s tests
python -m mypy src\llm_memlab\backends\vllm.py src\llm_memlab\serving_benchmark.py src\llm_memlab\backend_registry.py src\llm_memlab\benchmark_dashboard.py
python -m build
```

## Wheel Smoke

```powershell
python -m venv .wheel-smoke
.\.wheel-smoke\Scripts\python.exe -m pip install --no-deps dist\llm_memlab-0.1.1-py3-none-any.whl
.\.wheel-smoke\Scripts\python.exe -c "import llm_memlab; print(llm_memlab.__version__)"
.\.wheel-smoke\Scripts\llm-memlab.exe backend-demo
```

Remove `.wheel-smoke`, `build`, `dist`, and `src\llm_memlab.egg-info` after smoke testing.

## Optional Hardware Checks

```powershell
python -m llm_memlab backend-demo
python -m llm_memlab kernel-demo --device cuda --repeats 3
python -m llm_memlab serving-bench --model D:\hf_models\TinyLlama-1.1B-Chat-v1.0 --local-files-only --tokens 1 --device cpu --dtype fp32 --cache paged --json-out serving_bench.json --csv-out serving_bench.csv --html-out serving_dashboard.html
```

## Release Steps

```powershell
git status --short
git tag -a v0.1.1 -m "llm-memlab v0.1.1"
git push origin main
git push origin v0.1.1
```

Create a GitHub release from the matching release notes file.
