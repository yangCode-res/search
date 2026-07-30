import unittest

from pnsearch.clients.academic import CompositeAcademicClient
from pnsearch.config import Settings
from pnsearch.models.analyzer import HeuristicQueryAnalyzer
from pnsearch.models.reasoner import HeuristicReasoner
from pnsearch.models.reranker import HeuristicListwiseReranker
from pnsearch.pipeline import PNSearchPipeline
from pnsearch.schema import Paper


class FakeSearchClient:
    async def execute(self, actions):
        papers = [
            Paper(
                paper_id=f"p-{actions[0].query[:6]}",
                title="LLM Agent Academic Paper Search",
                abstract="An LLM agent autonomously performs academic paper search and iterative query search. " * 5,
                year=2025,
                source="fake",
            ),
            Paper(
                paper_id=f"n-{actions[0].query[:6]}",
                title="Fixed Corpus Question Answering",
                abstract="Question answering over a fixed document corpus. " * 5,
                year=2025,
                source="fake",
            ),
        ]
        return papers, len(actions), []


class PipelineTest(unittest.IsolatedAsyncioTestCase):
    async def test_end_to_end(self):
        settings = Settings(
            mode="heuristic",
            max_rounds=1,
            max_api_calls=3,
            min_select_score=0.4,
            min_borderline_score=0.2,
        )
        pipeline = PNSearchPipeline(
            settings,
            analyzer=HeuristicQueryAnalyzer(),
            reasoner=HeuristicReasoner(("semantic_scholar",)),
            reranker=HeuristicListwiseReranker(0.4, 0.2),
            search_client=FakeSearchClient(),
        )
        result = await pipeline.search(
            "LLM agent academic paper search，排除 fixed corpus question answering"
        )
        self.assertGreaterEqual(result.api_calls, 1)
        self.assertTrue(result.selected)
        self.assertEqual(result.stop_reason, "max_rounds_reached")
        self.assertIn("highly_relevant", result.to_dict())
