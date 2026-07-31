import unittest

from pnsearch.boundary import normalize_evidence_boundary
from pnsearch.schema import DecisionLabel


class BoundaryNormalizationTest(unittest.TestCase):
    def test_missing_abstract_evidence_becomes_borderline(self):
        label = normalize_evidence_boundary(
            DecisionLabel.REJECT,
            "The abstract does not mention superpixels or image patches.",
        )
        self.assertEqual(label, DecisionLabel.BORDERLINE)

    def test_explicit_task_mismatch_stays_reject(self):
        label = normalize_evidence_boundary(
            DecisionLabel.REJECT,
            "The abstract does not mention image retrieval; retrieval refers to phase recovery, not image retrieval.",
        )
        self.assertEqual(label, DecisionLabel.REJECT)
