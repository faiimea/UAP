import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from update_metric import update_metric


class UpdateMetricTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec = Path(self.tmp.name) / "spec.json"
        self.log = Path(self.tmp.name) / "run.log"
        self.original = {"metrics": {"accuracy": {"recorded": 0.8,
            "source": "Table 1 / accuracy / baseline", "tolerance": 0.01,
            "pattern": r"accuracy=(\d+\.\d+)", "after": None}}, "keep": True}
        self.spec.write_text(json.dumps(self.original))
        self.log.write_text("accuracy=0.1\nFINAL\naccuracy=0.91\n")

    def update(self, **kwargs):
        return update_metric(self.spec, "accuracy", **kwargs)

    def test_only_supplied_fields_change(self):
        self.assertEqual(self.update(recorded=0.91)["status"], "OK")
        expected = self.original
        expected["metrics"]["accuracy"]["recorded"] = 0.91
        self.assertEqual(json.loads(self.spec.read_text()), expected)

    def test_valid_reader(self):
        result = self.update(pattern=r"accuracy=(?P<value>\d+\.\d+)",
                             after="FINAL", run_log_path=self.log)
        self.assertEqual(result["status"], "OK")

    def test_partial_refuses_pair(self):
        result = self.update(recorded=0.91, pattern=r"missing=(\d+)",
                             after="FINAL", run_log_path=self.log)
        self.assertEqual(result["status"], "PARTIAL")
        self.assertEqual(set(result["fields_not_changed"]), {"pattern", "after"})
        saved = json.loads(self.spec.read_text())["metrics"]["accuracy"]
        self.assertEqual(saved["pattern"], self.original["metrics"]["accuracy"]["pattern"])
        self.assertIsNone(saved["after"])

    def test_invalid_readers(self):
        for kwargs in ({"pattern": "["}, {"after": "ABSENT"},
                       {"pattern": "accuracy"}, {"pattern": r"(1e999)"}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.update(**kwargs, run_log_path=self.log)["status"], "ERROR")

    def test_log_required(self):
        self.assertEqual(self.update(pattern=r"(\d+)")["status"], "ERROR")

    def test_invalid_paper_fields(self):
        for kwargs in ({"recorded": True}, {"recorded": float("nan")},
                       {"tolerance": -1}, {"source": ""}):
            self.assertEqual(self.update(**kwargs)["status"], "ERROR")

    def test_unknown_metric(self):
        self.assertEqual(update_metric(self.spec, "missing", recorded=1)["status"], "ERROR")

    def test_write_failure_reports_nothing_saved(self):
        with patch("update_metric.os.replace", side_effect=OSError("write failed")):
            result = self.update(recorded=0.91)
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["fields_saved"], {})
        self.assertEqual(json.loads(self.spec.read_text()), self.original)


if __name__ == "__main__":
    unittest.main()
