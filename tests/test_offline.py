import sqlite3
import tempfile
import unittest
from pathlib import Path

from pnsearch.offline import PasaOfflineIndex, fts_query, initialize_index, insert_papers


class OfflineIndexTest(unittest.TestCase):
    def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "papers.sqlite"
            connection = sqlite3.connect(path)
            initialize_index(connection)
            insert_papers(
                connection,
                [
                    (
                        "2501.00001",
                        "An LLM Agent for Academic Paper Search",
                        "The agent iteratively searches papers and follows citations.",
                        2025,
                        "ACL",
                    ),
                    (
                        "2501.00002",
                        "Image Classification with Residual Networks",
                        "A convolutional network for image recognition.",
                        2025,
                        "CVPR",
                    ),
                ],
            )
            connection.close()
            with PasaOfflineIndex(path) as index:
                hits = index.search("LLM agent academic paper search", limit=2)
                self.assertTrue(hits)
                self.assertEqual(hits[0].paper.paper_id, "2501.00001")
                self.assertEqual(index.get_by_id("2501.00002").venue, "CVPR")
                broad_hits = index.search(
                    "LLM agent academic paper search", limit=2, strategy="broad"
                )
                self.assertEqual(broad_hits[0].paper.paper_id, "2501.00001")

    def test_query_removes_generic_words(self):
        expression = fts_query("Which papers are about search agents?")
        self.assertIn('"search"', expression)
        self.assertIn('"agents"', expression)
        self.assertNotIn('"papers"', expression)
