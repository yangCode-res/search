import unittest

from pnsearch.models.analyzer import HeuristicQueryAnalyzer


class QueryAnalyzerTest(unittest.IsolatedAsyncioTestCase):
    async def test_extracts_year_and_exclusion(self):
        spec = await HeuristicQueryAnalyzer().analyze(
            "寻找2023年之后使用LLM Agent进行学术搜索的方法，关注查询分解，排除固定论文集问答"
        )
        self.assertEqual(spec.metadata.year_min, 2023)
        self.assertTrue(spec.inclusion_criteria)
        self.assertEqual(spec.exclusion_criteria[0].text, "固定论文集问答")

