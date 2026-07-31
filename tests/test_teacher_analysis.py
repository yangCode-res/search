import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "analyze_teacher_labels",
    Path(__file__).parents[1] / "scripts" / "analyze_teacher_labels.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TeacherAnalysisTest(unittest.TestCase):
    def test_deduplicates_and_counts_gold_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            records = [
                {"query_id": "q1", "paper_id": "p1", "teacher_label": "SELECT"},
                {"query_id": "q1", "paper_id": "p1", "teacher_label": "SELECT"},
                {
                    "query_id": "q1",
                    "paper_id": "p2",
                    "teacher_label": "REJECT",
                    "gold_match": True,
                },
            ]
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
            )
            report = MODULE.analyze(path)

        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["gold_reject_rate"], 1.0)
