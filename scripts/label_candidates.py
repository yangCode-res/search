#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.config import Settings
from pnsearch.datasets import iter_json_records, write_jsonl
from pnsearch.models.analyzer import HeuristicQueryAnalyzer
from pnsearch.models.reranker import HeuristicListwiseReranker, LLMListwiseReranker
from pnsearch.schema import Criterion, MetadataConstraints, Paper, QuerySpec


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env(args.config)
    queries = {str(item["query_id"]): item for item in iter_json_records(args.queries)}
    grouped = defaultdict(list)
    for item in iter_json_records(args.candidates):
        grouped[str(item["query_id"])].append(item)
    client = OpenAICompatibleClient(settings.llm_base_url, settings.llm_api_key, 180.0)
    reranker = LLMListwiseReranker(
        client,
        settings.reranker_model,
        batch_size=settings.batch_size,
        fallback=None,
    )
    analyzer = HeuristicQueryAnalyzer()
    labeled = []
    for index, (query_id, candidates) in enumerate(grouped.items(), start=1):
        query = queries.get(query_id)
        if not query:
            continue
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
        if index % 10 == 0:
            print(f"labeled_queries={index} labeled_papers={len(labeled)}")
    write_jsonl(args.output, labeled)
    print(json.dumps({"labeled": len(labeled)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Teacher-label mined paper candidates")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

