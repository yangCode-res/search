#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.config import Settings
from pnsearch.datasets import iter_json_records, write_jsonl
from pnsearch.models.analyzer import HeuristicQueryAnalyzer
from pnsearch.models.reranker import LLMListwiseReranker
from pnsearch.schema import Criterion, Paper


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env(args.config)
    queries = {str(item["query_id"]): item for item in iter_json_records(args.queries)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_path in args.candidates:
        for item in iter_json_records(candidate_path):
            grouped[str(item["query_id"])].append(item)

    completed: set[str] = set()
    existing_count = 0
    if args.resume and args.output.exists():
        for item in iter_json_records(args.output):
            completed.add(str(item["query_id"]))
            existing_count += 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("", encoding="utf-8")

    client = OpenAICompatibleClient(settings.llm_base_url, settings.llm_api_key, args.timeout)
    reranker = LLMListwiseReranker(
        client,
        settings.reranker_model,
        batch_size=settings.batch_size,
        fallback=None,
    )
    analyzer = HeuristicQueryAnalyzer()
    pending = [
        (query_id, candidates)
        for query_id, candidates in grouped.items()
        if query_id in queries and query_id not in completed
    ]
    if args.start:
        pending = pending[args.start :]
    if args.limit is not None:
        pending = pending[: args.limit]

    stats = {
        "existing_papers": existing_count,
        "pending_queries": len(pending),
        "completed_queries": 0,
        "labeled_papers": 0,
        "errors": 0,
    }
    errors: list[dict[str, str]] = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async def bounded(query_id: str, candidates: list[dict[str, Any]]):
        async with semaphore:
            try:
                if args.max_candidates_per_query:
                    candidates = candidates[: args.max_candidates_per_query]
                return query_id, await label_query(
                    query_id, queries[query_id], candidates, analyzer, reranker
                ), None
            except Exception as exc:
                return query_id, [], f"{type(exc).__name__}: {exc}"

    with args.output.open("a", encoding="utf-8") as output_handle:
        for offset in range(0, len(pending), args.concurrency):
            chunk = pending[offset : offset + args.concurrency]
            results = await asyncio.gather(
                *(bounded(query_id, candidates) for query_id, candidates in chunk)
            )
            for query_id, labeled, error in results:
                if error:
                    errors.append({"query_id": query_id, "error": error})
                    stats["errors"] += 1
                    continue
                for item in labeled:
                    output_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_handle.flush()
                stats["completed_queries"] += 1
                stats["labeled_papers"] += len(labeled)
            print(
                json.dumps({**stats, "llm_usage": client.usage_snapshot()}, ensure_ascii=False),
                flush=True,
            )
    if errors:
        write_jsonl(args.output.with_suffix(".errors.jsonl"), errors)
    print(
        json.dumps({**stats, "llm_usage": client.usage_snapshot()}, ensure_ascii=False, indent=2)
    )


async def label_query(
    query_id: str,
    query: dict[str, Any],
    candidates: list[dict[str, Any]],
    analyzer: HeuristicQueryAnalyzer,
    reranker: LLMListwiseReranker,
) -> list[dict[str, Any]]:
    base_spec = await analyzer.analyze(query["query"])
    relevance = query.get("relevance_criteria") or []
    if relevance:
        base_spec.inclusion_criteria = [
            Criterion(
                id=f"I{position}",
                text=item.get("description") or item.get("name") or str(item),
                required=True,
            )
            for position, item in enumerate(relevance, start=1)
        ]
    papers = [
        Paper(
            paper_id=str(item["paper_id"]),
            title=item.get("title") or "",
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            venue=item.get("venue") or "",
            doi=item.get("doi") or "",
            source=item.get("source") or "",
        )
        for item in candidates
    ]
    judgments = {item.paper_id: item for item in await reranker.rank(base_spec, papers)}
    labeled = []
    for candidate in candidates:
        judgment = judgments.get(str(candidate["paper_id"]))
        if not judgment:
            continue
        labeled.append(
            {
                **candidate,
                "teacher_label": judgment.label.value,
                "teacher_score": judgment.relevance_score,
                "criteria_labels": {
                    "inclusion": judgment.inclusion_judgments,
                    "exclusion": judgment.exclusion_judgments,
                },
                "reason_codes": judgment.reason_codes,
                "teacher_rationale": judgment.rationale,
            }
        )
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-label mined paper candidates")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-candidates-per-query", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
