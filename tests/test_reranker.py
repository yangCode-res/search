import unittest

from pnsearch.models.reranker import HeuristicListwiseReranker, LLMListwiseReranker
from pnsearch.schema import Criterion, DecisionLabel, Paper, QuerySpec
from pnsearch.training.formatting import reranker_messages


class RerankerTest(unittest.IsolatedAsyncioTestCase):
    async def test_accepts_semantic_scores_from_mimo(self):
        class FakeClient:
            async def chat_json(self, **kwargs):
                return {
                    "results": [
                        {
                            "paper_id": "p1",
                            "label": "SELECT",
                            "relevance_score": "CLEAR",
                            "evidence_sufficiency": "SUFFICIENT",
                        }
                    ]
                }

        judgments = await LLMListwiseReranker(FakeClient(), "mimo").rank(
            QuerySpec(
                original_query="query",
                research_intent="query",
                inclusion_criteria=[],
                exclusion_criteria=[],
            ),
            [Paper(paper_id="p1", title="Paper", abstract="Abstract")],
        )
        self.assertEqual(judgments[0].relevance_score, 0.9)
        self.assertEqual(judgments[0].evidence_sufficiency, 0.9)

    async def test_separates_relevant_and_irrelevant(self):
        spec = QuerySpec(
            original_query="LLM agent academic paper search",
            research_intent="LLM agent academic paper search",
            inclusion_criteria=[
                Criterion("I1", "academic paper search", True),
                Criterion("I2", "LLM agent", True),
            ],
            exclusion_criteria=[Criterion("E1", "fixed corpus question answering")],
        )
        papers = [
            Paper(
                paper_id="good",
                title="An LLM Agent for Academic Paper Search",
                abstract="We propose an LLM agent that autonomously searches academic papers and iterates queries. " * 4,
            ),
            Paper(
                paper_id="bad",
                title="Question Answering over a Fixed Corpus",
                abstract="A fixed corpus question answering benchmark. " * 4,
            ),
        ]
        judgments = await HeuristicListwiseReranker(0.45, 0.25).rank(spec, papers)
        labels = {item.paper_id: item.label for item in judgments}
        self.assertEqual(labels["good"], DecisionLabel.SELECT)
        self.assertEqual(labels["bad"], DecisionLabel.REJECT)

    async def test_training_format_compacts_long_abstracts(self):
        messages = reranker_messages(
            {
                "query": "academic search agents",
                "candidates": [
                    {
                        "paper_id": "p1",
                        "title": "Paper",
                        "abstract": "a" * 5000,
                        "label": "SELECT",
                    }
                ],
            }
        )
        self.assertLess(len(messages[1]["content"]), 2500)
