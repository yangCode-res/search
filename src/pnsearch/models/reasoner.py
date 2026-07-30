from __future__ import annotations

import json
from abc import ABC, abstractmethod

from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.schema import ActionType, Feedback, SearchAction, SearchState
from pnsearch.text import compact, top_terms


class SearchReasoner(ABC):
    @abstractmethod
    async def plan(self, state: SearchState, round_index: int, remaining_calls: int) -> list[SearchAction]:
        raise NotImplementedError


class HeuristicReasoner(SearchReasoner):
    def __init__(self, sources: tuple[str, ...] = ("semantic_scholar", "openalex")):
        self.sources = sources

    async def plan(self, state: SearchState, round_index: int, remaining_calls: int) -> list[SearchAction]:
        if remaining_calls <= 0:
            return [SearchAction(type=ActionType.STOP, purpose="API budget exhausted")]
        spec = state.query_spec
        feedback = state.history[-1].feedback if state.history else Feedback()
        queries: list[tuple[str, str, str]] = []
        if round_index == 1:
            queries.append((spec.research_intent, "direct intent search", "initial"))
            required = [item.text for item in spec.inclusion_criteria if item.required]
            if required:
                queries.append((" ".join(required), "required criteria search", "initial"))
            preferred = [item.text for item in spec.inclusion_criteria if not item.required]
            if preferred:
                queries.append((f"{spec.research_intent} {' '.join(preferred)}", "preference expansion", "initial"))
        else:
            if feedback.positive_expansions:
                terms = " ".join(feedback.positive_expansions[:5])
                queries.append((f"{spec.research_intent} {terms}", "positive concept expansion", "positive"))
            if feedback.negative_patterns:
                excluded = [str(item.get("type", "")) for item in feedback.negative_patterns[:4]]
                queries.append((spec.research_intent, "negative-feedback query repair", "negative:" + ",".join(excluded)))
            if feedback.borderline_needs:
                paper = feedback.borderline_needs[0]
                queries.append((f"{spec.research_intent} {paper.get('missing_evidence', '')}", "borderline verification", "borderline"))
        if not queries:
            queries.append((spec.original_query, "fallback query", "fallback"))
        seen = {action.query.casefold() for record in state.history for action in record.actions}
        actions: list[SearchAction] = []
        for index, (query, purpose, feedback_source) in enumerate(queries):
            normalized = " ".join(query.split())
            if not normalized or normalized.casefold() in seen:
                continue
            source = self.sources[index % len(self.sources)]
            actions.append(
                SearchAction(
                    type=ActionType.KEYWORD_SEARCH,
                    query=normalized,
                    source=source,
                    purpose=purpose,
                    max_results=30,
                    feedback_source=feedback_source,
                    exclude_concepts=[item.text for item in spec.exclusion_criteria],
                )
            )
            if len(actions) >= remaining_calls:
                break
        return actions or [SearchAction(type=ActionType.STOP, purpose="no novel query available")]


class LLMReasoner(SearchReasoner):
    def __init__(self, client: OpenAICompatibleClient, model: str, sources: tuple[str, ...]):
        self.client = client
        self.model = model
        self.sources = sources

    async def plan(self, state: SearchState, round_index: int, remaining_calls: int) -> list[SearchAction]:
        context = _state_summary(state)
        data = await self.client.chat_json(
            model=self.model,
            system=(
                "你是高召回学术搜索 Reasoner。只输出 JSON。首轮生成互补查询以提高召回；后续同时利用正选概念、"
                "负选错误模式和边界证据缺口。每个动作必须有目的，不得重复历史查询，并严格遵守 API 预算。"
            ),
            user=(
                f"当前轮次：{round_index}\n剩余 API 调用：{remaining_calls}\n"
                f"允许数据源：{list(self.sources)}\n状态：{json.dumps(context, ensure_ascii=False)}\n"
                "输出：{actions:[{type,query,source,purpose,max_results,seed_paper_id,exclude_concepts,feedback_source}],stop:boolean}"
            ),
            max_tokens=2200,
        )
        if data.get("stop"):
            return [SearchAction(type=ActionType.STOP, purpose="reasoner requested stop")]
        actions: list[SearchAction] = []
        for item in data.get("actions") or []:
            try:
                action_type = ActionType(item.get("type", "KEYWORD_SEARCH"))
            except ValueError:
                action_type = ActionType.KEYWORD_SEARCH
            source = item.get("source") or self.sources[0]
            if source not in self.sources:
                source = self.sources[0]
            actions.append(
                SearchAction(
                    type=action_type,
                    query=item.get("query") or "",
                    source=source,
                    purpose=item.get("purpose") or "",
                    max_results=min(int(item.get("max_results") or 20), 100),
                    seed_paper_id=item.get("seed_paper_id"),
                    exclude_concepts=item.get("exclude_concepts") or [],
                    feedback_source=item.get("feedback_source") or "llm",
                )
            )
            if len(actions) >= remaining_calls:
                break
        return actions or [SearchAction(type=ActionType.STOP, purpose="empty plan")]


def _state_summary(state: SearchState) -> dict[str, object]:
    selected = [
        {
            "title": state.papers[paper_id].title,
            "score": judgment.relevance_score,
            "reason_codes": judgment.reason_codes,
        }
        for paper_id, judgment in state.judgments.items()
        if judgment.label.value == "SELECT" and paper_id in state.papers
    ][:15]
    return {
        "query": state.query_spec.original_query,
        "intent": state.query_spec.research_intent,
        "inclusion": [item.text for item in state.query_spec.inclusion_criteria],
        "exclusion": [item.text for item in state.query_spec.exclusion_criteria],
        "selected": selected,
        "last_feedback": state.history[-1].feedback if state.history else {},
        "searched_queries": [action.query for record in state.history for action in record.actions],
    }
