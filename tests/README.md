# Test Profiles

The project keeps tests in `unittest` form and splits CI by filename profile:

- `cpu`: `python -m unittest discover -s tests`
- `cuda`: `python -m unittest discover -s tests -p test_cuda_triton.py`
- `hf`: `python -m unittest discover -s tests -p test_hf_integration_matrix.py`
- `slow`: `python -m unittest discover -s tests -p test_kernel_certification.py`

CUDA tests must use `skipUnless(torch.cuda.is_available())`. HF tests must avoid network by default and only run real cached model smoke tests when `LLM_MEMLAB_HF_SMOKE_MODEL` points to a local model.
