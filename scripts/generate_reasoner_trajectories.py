#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pnsearch.candidate import coarse_rank, merge_candidates
from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.config import Settings
from pnsearch.datasets import iter_json_records
from pnsearch.evaluation import normalize_title
from pnsearch.feedback import analyze_feedback
from pnsearch.models.analyzer import HeuristicQueryAnalyzer, LLMQueryAnalyzer
from pnsearch.models.reasoner import HeuristicReasoner, LLMReasoner
from pnsearch.models.reranker import HeuristicListwiseReranker, LLMListwiseReranker
from pnsearch.offline import PasaOfflineSearchClient
from pnsearch.schema import ActionType, DecisionLabel, RoundRecord, SearchAction, SearchState
from pnsearch.text import tokenize


SUPPORTED_OFFLINE_ACTIONS = {
    ActionType.KEYWORD_SEARCH,
    ActionType.SEMANTIC_SEARCH,
    ActionType.QUERY_REWRITE,
    ActionType.SIMILAR_PAPER,
    ActionType.AUTHOR_EXPANSION,
}


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env(args.config)
    llm = OpenAICompatibleClient(
        settings.llm_base_url, settings.llm_api_key, timeout=args.timeout
    )
    heuristic_reranker = HeuristicListwiseReranker(
        settings.min_select_score, settings.min_borderline_score
    )
    if args.teacher == "llm":
        analyzer = LLMQueryAnalyzer(llm, settings.reasoner_model)
        reasoner = LLMReasoner(llm, settings.reasoner_model, ("pasa_offline",))
        reranker = LLMListwiseReranker(
            llm,
            settings.reranker_model,
            batch_size=args.reranker_batch_size,
            fallback=heuristic_reranker,
        )
    else:
        analyzer = HeuristicQueryAnalyzer()
        reasoner = HeuristicReasoner(("pasa_offline",))
        reranker = heuristic_reranker

    search_client = PasaOfflineSearchClient(args.index)
    completed: set[str] = set()
    if args.resume and args.output.exists():
        completed = {str(item["query_id"]) for item in iter_json_records(args.output)}
    for path in (args.output, args.preferences_output, args.output.with_suffix(".errors.jsonl")):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    stats = {
        "existing_queries": len(completed),
        "queries": 0,
        "rounds": 0,
        "positive_reward_rounds": 0,
        "errors": 0,
    }
    attempted = 0
    try:
        with ExitStack() as stack:
            output_handle = stack.enter_context(args.output.open(mode, encoding="utf-8"))
            preferences_handle = (
                stack.enter_context(args.preferences_output.open(mode, encoding="utf-8"))
                if args.preferences_output
                else None
            )
            errors_handle = stack.enter_context(
                args.output.with_suffix(".errors.jsonl").open(mode, encoding="utf-8")
            )
            for record in iter_json_records(args.queries):
                query_id = str(record["query_id"])
                if query_id in completed:
                    continue
                if args.limit is not None and attempted >= args.limit:
                    break
                attempted += 1
                try:
                    query_examples, query_preferences = await generate_query_trajectory(
                        record,
                        analyzer=analyzer,
                        reasoner=reasoner,
                        reranker=reranker,
                        search_client=search_client,
                        max_rounds=args.max_rounds,
                        max_calls=args.max_calls,
                        candidate_limit=args.candidate_limit,
                        results_per_action=args.results_per_action,
                    )
                    output_handle.write(
                        "".join(
                            json.dumps(item, ensure_ascii=False) + "\n"
                            for item in query_examples
                        )
                    )
                    output_handle.flush()
                    if preferences_handle:
                        preferences_handle.write(
                            "".join(
                                json.dumps(item, ensure_ascii=False) + "\n"
                                for item in query_preferences
                            )
                        )
                        preferences_handle.flush()
                    stats["queries"] += 1
                    stats["rounds"] += sum(not item["stop"] for item in query_examples)
                    stats["positive_reward_rounds"] += sum(
                        float(item.get("reward") or 0) > 0 for item in query_examples
                    )
                except Exception as exc:  # one failed teacher call must not lose the full run
                    error = {
                        "query_id": query_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                    errors_handle.flush()
                    stats["errors"] += 1
                if attempted % 10 == 0:
                    print(
                        json.dumps(
                            {**stats, "llm_usage": llm.usage_snapshot()}, ensure_ascii=False
                        ),
                        flush=True,
                    )
    finally:
        search_client.close()
    print(json.dumps({**stats, "llm_usage": llm.usage_snapshot()}, ensure_ascii=False, indent=2))


async def generate_query_trajectory(
    record: dict[str, Any],
    *,
    analyzer: Any,
    reasoner: Any,
    reranker: Any,
    search_client: PasaOfflineSearchClient,
    max_rounds: int,
    max_calls: int,
    candidate_limit: int,
    results_per_action: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = str(record["query"])
    query_id = str(record["query_id"])
    spec = await analyzer.analyze(query)
    state = SearchState(query_spec=spec)
    examples: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    previous_gold_selected: set[str] = set()
    gold_ids = {
        str(item.get("paper_id") or "").casefold()
        for item in record.get("positive_papers") or []
        if item.get("paper_id")
    }
    gold_titles = {
        normalize_title(str(item.get("title") or ""))
        for item in record.get("positive_papers") or []
        if item.get("title")
    }
    gold_total = max(1, len(gold_ids | gold_titles))

    for round_index in range(1, max_rounds + 1):
        remaining = max_calls - state.api_calls
        before = _training_state(query_id, query, state, remaining)
        actions = await reasoner.plan(state, round_index, remaining)
        if not actions or all(action.type == ActionType.STOP for action in actions):
            before.update({"target_actions": [], "stop": True, "reward": 0.0})
            examples.append(before)
            break
        actions = [_offline_safe_action(item, query) for item in actions[:remaining]]
        for action in actions:
            action.max_results = min(results_per_action, action.max_results)
        before["target_actions"] = [_action_dict(item) for item in actions]
        before["stop"] = False

        raw_papers, api_calls, request_errors = await search_client.execute(actions)
        state.api_calls += api_calls
        unique_new, duplicate_ratio = merge_candidates(state.papers, raw_papers)
        candidates = coarse_rank(query, unique_new, candidate_limit)
        judgments = await reranker.rank(spec, candidates)
        for judgment in judgments:
            state.judgments[judgment.paper_id] = judgment

        selected_ids = {
            paper_id
            for paper_id, judgment in state.judgments.items()
            if judgment.label == DecisionLabel.SELECT
        }
        selected_gold = {
            paper_id
            for paper_id in selected_ids
            if paper_id.casefold() in gold_ids
            or (
                paper_id in state.papers
                and normalize_title(state.papers[paper_id].title) in gold_titles
            )
        }
        delta_gold = len(selected_gold - previous_gold_selected)
        previous_gold_selected = selected_gold
        new_select = sum(item.label == DecisionLabel.SELECT for item in judgments)
        reject_count = sum(item.label == DecisionLabel.REJECT for item in judgments)
        reject_ratio = reject_count / len(judgments) if judgments else 1.0
        feedback = analyze_feedback(state.papers, judgments, query_terms=set(tokenize(query)))
        if request_errors:
            feedback.negative_patterns.append(
                {"type": "OFFLINE_ACTION_ERROR", "count": len(request_errors)}
            )
        reward = (
            2.0 * (delta_gold / gold_total)
            + 0.05 * min(new_select, 5)
            - 0.15 * duplicate_ratio
            - 0.03 * api_calls
            - 0.05 * max(0.0, reject_ratio - 0.8)
        )
        state.history.append(
            RoundRecord(
                round_index=round_index,
                actions=actions,
                retrieved_count=len(raw_papers),
                unique_new_count=len(unique_new),
                new_select_count=new_select,
                reject_ratio=reject_ratio,
                duplicate_ratio=duplicate_ratio,
                api_calls=api_calls,
                feedback=feedback,
            )
        )
        before.update(
            {
                "reward": round(reward, 6),
                "outcome": {
                    "retrieved": len(raw_papers),
                    "unique_new": len(unique_new),
                    "new_select": new_select,
                    "new_gold": delta_gold,
                    "gold_recall": round(len(selected_gold) / gold_total, 6),
                    "reject_ratio": round(reject_ratio, 6),
                    "duplicate_ratio": round(duplicate_ratio, 6),
                },
            }
        )
        examples.append(before)
        if actions and reward > 0:
            if round_index == 1:
                rejected_query = "machine learning research papers"
                rejected_purpose = "overly broad query that ignores the user's constraints"
            else:
                rejected_query = state.history[0].actions[0].query
                rejected_purpose = "repeat a previous query without using new feedback"
            preferences.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "search_history": before["search_history"],
                    "feedback": before["feedback"],
                    "remaining_budget": remaining,
                    "chosen": before["target_actions"],
                    "rejected": [
                        {
                            "type": "KEYWORD_SEARCH",
                            "query": rejected_query,
                            "source": "pasa_offline",
                            "purpose": rejected_purpose,
                            "max_results": actions[0].max_results,
                            "feedback_source": "negative_synthetic",
                        }
                    ],
                    "chosen_reward": round(reward, 6),
                }
            )
        if state.api_calls >= max_calls:
            break

    if not examples or not examples[-1]["stop"]:
        final_state = _training_state(query_id, query, state, max(0, max_calls - state.api_calls))
        final_state.update({"target_actions": [], "stop": True, "reward": 0.0})
        examples.append(final_state)
    return examples, preferences


def _training_state(
    query_id: str, query: str, state: SearchState, remaining_budget: int
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": query,
        "search_history": [_round_dict(item) for item in state.history],
        "feedback": asdict(state.history[-1].feedback) if state.history else {},
        "remaining_budget": remaining_budget,
    }


def _round_dict(record: RoundRecord) -> dict[str, Any]:
    return {
        "round_index": record.round_index,
        "actions": [_action_dict(item) for item in record.actions],
        "retrieved_count": record.retrieved_count,
        "unique_new_count": record.unique_new_count,
        "new_select_count": record.new_select_count,
        "reject_ratio": record.reject_ratio,
        "duplicate_ratio": record.duplicate_ratio,
        "api_calls": record.api_calls,
        "feedback": asdict(record.feedback),
    }


def _action_dict(action: SearchAction) -> dict[str, Any]:
    data = asdict(action)
    data["type"] = action.type.value
    return data


def _offline_safe_action(action: SearchAction, original_query: str) -> SearchAction:
    if action.type in SUPPORTED_OFFLINE_ACTIONS:
        return action
    return SearchAction(
        type=ActionType.KEYWORD_SEARCH,
        query=action.query or original_query,
        source="pasa_offline",
        purpose=f"offline rewrite of {action.type.value}: {action.purpose}",
        max_results=action.max_results,
        exclude_concepts=action.exclude_concepts,
        feedback_source=action.feedback_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-round Reasoner trajectories")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preferences-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--teacher", choices=["heuristic", "llm"], default="llm")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-calls", type=int, default=9)
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument("--results-per-action", type=int, default=30)
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
