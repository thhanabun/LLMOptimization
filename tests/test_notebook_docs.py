import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


ROOT = pathlib.Path(__file__).resolve().parents[1]


class NotebookDocsTests(unittest.TestCase):
    def test_cloud_notebooks_are_valid_and_call_smoke_script(self):
        for path in (ROOT / "notebooks").glob("*_quickstart.ipynb"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["nbformat"], 4)
            sources = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
            self.assertIn("examples/cloud_smoke.py", sources)
            self.assertIn("LLM_MEMLAB_MODEL_ROOT", sources)
            self.assertIn("cloud_dashboard.html", sources)

    def test_docs_cover_certification_vllm_and_api_reference(self):
        docs = {
            "certify": ROOT / "docs" / "certify_your_model.md",
            "vllm": ROOT / "docs" / "compare_with_vllm.md",
            "api": ROOT / "docs" / "api_reference.md",
            "cloud": ROOT / "docs" / "cloud_notebooks.md",
        }
        for path in docs.values():
            self.assertTrue(path.exists(), path)
        self.assertIn("hf-cache-certify", docs["certify"].read_text(encoding="utf-8"))
        self.assertIn("serving-bench", docs["vllm"].read_text(encoding="utf-8"))
        self.assertIn("llm_memlab.production", docs["api"].read_text(encoding="utf-8"))
        self.assertIn("notebooks/colab_quickstart.ipynb", docs["cloud"].read_text(encoding="utf-8"))

    def test_examples_prefer_env_model_paths(self):
        cloud_smoke = (ROOT / "examples" / "cloud_smoke.py").read_text(encoding="utf-8")
        hf_suite = (ROOT / "examples" / "hf_suite_example.py").read_text(encoding="utf-8")
        self.assertIn("LLM_MEMLAB_MODEL", cloud_smoke)
        self.assertIn("LLM_MEMLAB_MODEL_ROOT", cloud_smoke)
        self.assertIn("LLM_MEMLAB_MODEL", hf_suite)


if __name__ == "__main__":
    unittest.main()
