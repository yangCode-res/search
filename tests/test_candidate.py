import unittest

from pnsearch.candidate import merge_candidates
from pnsearch.schema import Paper


class CandidatePoolTest(unittest.TestCase):
    def test_deduplicates_by_normalized_title(self):
        existing = {}
        first = Paper(paper_id="1", title="A Paper: About Search", retrieved_by=["q1"])
        second = Paper(paper_id="2", title="A paper about search", abstract="abstract", retrieved_by=["q2"])
        new, ratio = merge_candidates(existing, [first, second])
        self.assertEqual(len(new), 1)
        self.assertEqual(len(existing), 1)
        self.assertEqual(ratio, 0.5)
        self.assertEqual(existing["1"].abstract, "abstract")
        self.assertEqual(existing["1"].retrieved_by, ["q1", "q2"])

