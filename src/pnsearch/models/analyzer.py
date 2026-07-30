from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.schema import Criterion, MetadataConstraints, QuerySpec


class QueryAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, query: str) -> QuerySpec:
        raise NotImplementedError


class HeuristicQueryAnalyzer(QueryAnalyzer):
    async def analyze(self, query: str) -> QuerySpec:
        years = [int(value) for value in re.findall(r"(?:19|20)\d{2}", query)]
        exclusions = _extract_exclusions(query)
        positive_text = re.split(r"(?:排除|不要|不包括|excluding?|without)", query, maxsplit=1, flags=re.I)[0]
        clauses = [
            clause.strip(" ，,。;；")
            for clause in re.split(r"[，,；;。]|(?:并且|同时|要求|重点关注|关注)", positive_text)
            if len(clause.strip()) >= 3
        ]
        if not clauses:
            clauses = [query.strip()]
        inclusion = [
            Criterion(id=f"I{index}", text=clause, required=index <= 2)
            for index, clause in enumerate(clauses[:6], start=1)
        ]
        exclusion = [
            Criterion(id=f"E{index}", text=clause, required=True)
            for index, clause in enumerate(exclusions[:5], start=1)
        ]
        metadata = MetadataConstraints(
            year_min=min(years) if years else None,
            year_max=max(years) if len(years) > 1 else None,
        )
        return QuerySpec(
            original_query=query,
            research_intent=clauses[0],
            inclusion_criteria=inclusion,
            exclusion_criteria=exclusion,
            metadata=metadata,
        )


class LLMQueryAnalyzer(QueryAnalyzer):
    def __init__(self, client: OpenAICompatibleClient, model: str):
        self.client = client
        self.model = model

    async def analyze(self, query: str) -> QuerySpec:
        data = await self.client.chat_json(
            model=self.model,
            system=(
                "你是复杂学术查询分析器。只输出 JSON。区分必须满足的纳入条件、排序偏好、"
                "明确排除条件和可规则执行的元数据约束。不要把宽泛主题拆成大量同义条件。"
            ),
            user=(
                f"查询：{query}\n\n"
                "输出结构：{research_intent:string, inclusion_criteria:[{id,criterion,required}], "
                "exclusion_criteria:[{id,criterion}], metadata_constraints:{year_min,year_max,venues,domains,paper_types}}"
            ),
            max_tokens=1600,
        )
        metadata = data.get("metadata_constraints") or {}
        return QuerySpec(
            original_query=query,
            research_intent=data.get("research_intent") or query,
            inclusion_criteria=[
                Criterion(
                    id=item.get("id") or f"I{index}",
                    text=item.get("criterion") or "",
                    required=bool(item.get("required", True)),
                )
                for index, item in enumerate(data.get("inclusion_criteria") or [], start=1)
                if item.get("criterion")
            ],
            exclusion_criteria=[
                Criterion(id=item.get("id") or f"E{index}", text=item.get("criterion") or "")
                for index, item in enumerate(data.get("exclusion_criteria") or [], start=1)
                if item.get("criterion")
            ],
            metadata=MetadataConstraints(
                year_min=metadata.get("year_min"),
                year_max=metadata.get("year_max"),
                venues=metadata.get("venues") or [],
                domains=metadata.get("domains") or [],
                paper_types=metadata.get("paper_types") or [],
            ),
        )


def _extract_exclusions(query: str) -> list[str]:
    match = re.search(r"(?:排除|不要|不包括|excluding?|without)\s*(.+)$", query, flags=re.I)
    if not match:
        return []
    return [item.strip(" ，,。;；") for item in re.split(r"[，,；;。]", match.group(1)) if item.strip()]

