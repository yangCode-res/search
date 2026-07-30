from __future__ import annotations

from pnsearch.candidate import apply_hard_filters, coarse_rank, merge_candidates
from pnsearch.clients.academic import CompositeAcademicClient, OpenAlexClient, SemanticScholarClient
from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.config import Settings
from pnsearch.feedback import analyze_feedback
from pnsearch.models.analyzer import HeuristicQueryAnalyzer, LLMQueryAnalyzer, QueryAnalyzer
from pnsearch.models.reasoner import HeuristicReasoner, LLMReasoner, SearchReasoner
from pnsearch.models.reranker import (
    HeuristicListwiseReranker,
    LLMListwiseReranker,
    ListwiseReranker,
)
from pnsearch.schema import (
    ActionType,
    DecisionLabel,
    RoundRecord,
    SearchResult,
    SearchState,
)
from pnsearch.stopping import should_stop
from pnsearch.text import tokenize


class PNSearchPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        analyzer: QueryAnalyzer | None = None,
        reasoner: SearchReasoner | None = None,
        reranker: ListwiseReranker | None = None,
        search_client: CompositeAcademicClient | None = None,
    ):
        self.settings = settings
        llm = OpenAICompatibleClient(
            settings.llm_base_url,
            settings.llm_api_key,
            timeout=max(settings.request_timeout, 120.0),
        )
        heuristic_ranker = HeuristicListwiseReranker(
            settings.min_select_score,
            settings.min_borderline_score,
        )
        if settings.mode == "llm":
            self.analyzer = analyzer or LLMQueryAnalyzer(llm, settings.reasoner_model)
            self.reasoner = reasoner or LLMReasoner(
                llm, settings.reasoner_model, settings.search_sources
            )
            self.reranker = reranker or LLMListwiseReranker(
                llm,
                settings.reranker_model,
                settings.batch_size,
                fallback=heuristic_ranker,
            )
        else:
            self.analyzer = analyzer or HeuristicQueryAnalyzer()
            self.reasoner = reasoner or HeuristicReasoner(settings.search_sources)
            self.reranker = reranker or heuristic_ranker
        self.search_client = search_client or CompositeAcademicClient(
            [
                SemanticScholarClient(
                    settings.semantic_scholar_api_key,
                    settings.user_agent,
                    settings.request_timeout,
                ),
                OpenAlexClient(settings.user_agent, settings.request_timeout),
            ]
        )

    async def search(self, query: str) -> SearchResult:
        spec = await self.analyzer.analyze(query)
        state = SearchState(query_spec=spec)
        previous_selected: set[str] = set()

        for round_index in range(1, self.settings.max_rounds + 1):
            remaining_calls = self.settings.max_api_calls - state.api_calls
            actions = await self.reasoner.plan(state, round_index, remaining_calls)
            if not actions or all(action.type == ActionType.STOP for action in actions):
                state.stop_reason = actions[0].purpose if actions else "empty_plan"
                break
            actions = [action for action in actions if action.type != ActionType.STOP][:remaining_calls]
            raw_papers, api_calls, errors = await self.search_client.execute(actions)
            state.api_calls += api_calls
            unique_new, duplicate_ratio = merge_candidates(state.papers, raw_papers)
            filtered = apply_hard_filters(unique_new, spec.metadata)
            candidates = coarse_rank(query, filtered, self.settings.candidate_limit)
            judgments = await self.reranker.rank(spec, candidates)
            for judgment in judgments:
                state.judgments[judgment.paper_id] = judgment
            current_selected = {
                paper_id
                for paper_id, judgment in state.judgments.items()
                if judgment.label == DecisionLabel.SELECT
            }
            new_select_count = len(current_selected - previous_selected)
            previous_selected = current_selected
            reject_count = sum(item.label == DecisionLabel.REJECT for item in judgments)
            reject_ratio = reject_count / len(judgments) if judgments else 1.0
            feedback = analyze_feedback(
                state.papers,
                judgments,
                query_terms=set(tokenize(query)),
            )
            if errors:
                feedback.negative_patterns.append(
                    {"type": "SEARCH_API_ERROR", "count": len(errors), "details": errors[:3]}
                )
            state.history.append(
                RoundRecord(
                    round_index=round_index,
                    actions=actions,
                    retrieved_count=len(raw_papers),
                    unique_new_count=len(unique_new),
                    new_select_count=new_select_count,
                    reject_ratio=reject_ratio,
                    duplicate_ratio=duplicate_ratio,
                    api_calls=api_calls,
                    feedback=feedback,
                )
            )
            stop, reason = should_stop(state, self.settings)
            if stop:
                state.stop_reason = reason
                break

        if not state.stop_reason:
            state.stop_reason = "completed"
        ranked = sorted(
            state.judgments.values(), key=lambda item: item.relevance_score, reverse=True
        )
        selected = [
            (state.papers[item.paper_id], item)
            for item in ranked
            if item.label == DecisionLabel.SELECT and item.paper_id in state.papers
        ][: self.settings.final_limit]
        borderline = [
            (state.papers[item.paper_id], item)
            for item in ranked
            if item.label == DecisionLabel.BORDERLINE and item.paper_id in state.papers
        ][: self.settings.final_limit]
        rejected_count = sum(item.label == DecisionLabel.REJECT for item in ranked)
        return SearchResult(
            query_spec=spec,
            selected=selected,
            borderline=borderline,
            rejected_count=rejected_count,
            rounds=state.history,
            api_calls=state.api_calls,
            stop_reason=state.stop_reason,
        )
