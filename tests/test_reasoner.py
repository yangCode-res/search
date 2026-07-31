import unittest

from pnsearch.models.reasoner import HeuristicReasoner
from pnsearch.schema import Criterion, QuerySpec, SearchState


class ReasonerTest(unittest.IsolatedAsyncioTestCase):
    async def test_deduplicates_queries_within_the_same_round(self):
        spec = QuerySpec(
            original_query="academic paper search agents",
            research_intent="academic paper search agents",
            inclusion_criteria=[
                Criterion("I1", "academic paper search agents", required=True)
            ],
            exclusion_criteria=[],
        )
        actions = await HeuristicReasoner(("pasa_offline",)).plan(
            SearchState(query_spec=spec), 1, 3
        )
        self.assertEqual(len(actions), 1)
