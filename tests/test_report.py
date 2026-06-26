import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from llm_memlab.report import make_table


class ReportTests(unittest.TestCase):
    def test_table_contains_headers_and_rows(self):
        table = make_table(("A", "B"), [(1, "two")])
        self.assertIn("A", table)
        self.assertIn("two", table)


if __name__ == "__main__":
    unittest.main()
