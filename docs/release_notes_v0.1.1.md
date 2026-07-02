# llm-memlab v0.1.1 Release Notes

`llm-memlab` v0.1.1 is a notebook, docs, and portability patch release.

## Highlights

- Added ready-to-run Colab and Kaggle notebooks.
- Added `examples/cloud_smoke.py`, a portable smoke workflow for local, Colab, Kaggle, and cloud GPU environments.
- Added docs for model certification, vLLM comparison, and the stable API surface.
- Improved quickstart and cloud docs around `LLM_MEMLAB_MODEL` and `LLM_MEMLAB_MODEL_ROOT`.
- Bumped package metadata to `0.1.1`.

## Validation

- `ruff check src tests examples`
- `python -m unittest discover -s tests`
- selected `mypy` checks
- `python -m build`
- wheel install smoke

## Scope

This is a polish release. It does not promote experimental Triton/CuTile/vLLM execution paths to production; those still require target-environment certification.
