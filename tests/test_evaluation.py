import unittest

from pnsearch.evaluation import evaluate_ids


class EvaluationTest(unittest.TestCase):
    def test_f1_and_recall(self):
        result = evaluate_ids(["1", "x", "2"], ["1", "2", "3"])
        self.assertAlmostEqual(result.precision, 2 / 3)
        self.assertAlmostEqual(result.recall, 2 / 3)
        self.assertAlmostEqual(result.f1, 2 / 3)
        self.assertAlmostEqual(result.recall_at_20, 2 / 3)

