#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pnsearch.clients.academic import CompositeAcademicClient, OpenAlexClient, SemanticScholarClient
from pnsearch.config import Settings
from pnsearch.datasets import iter_json_records, write_jsonl
from pnsearch.models.analyzer import HeuristicQueryAnalyzer
from pnsearch.models.reasoner import HeuristicReasoner
from pnsearch.schema import SearchState


async def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env(args.config)
    analyzer = HeuristicQueryAnalyzer()
    reasoner = HeuristicReasoner(settings.search_sources)
    client = CompositeAcademicClient(
        [
            SemanticScholarClient(
                settings.semantic_scholar_api_key, settings.user_agent, settings.request_timeout
            ),
            OpenAlexClient(settings.user_agent, settings.request_timeout),
        ]
    )
    output = []
    errors = []
    for index, record in enumerate(iter_json_records(args.queries), start=1):
        spec = await analyzer.analyze(record["query"])
        state = SearchState(query_spec=spec)
        actions = await reasoner.plan(state, 1, args.actions_per_query)
        for action in actions:
            action.max_results = args.results_per_action
        papers, _, request_errors = await client.execute(actions)
        seen = set()
        for paper in papers:
            key = (paper.doi.casefold() if paper.doi else paper.title.casefold().strip())
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "query_id": str(record["query_id"]),
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "source": paper.source,
                    "retrieved_by": paper.retrieved_by,
                }
            )
        errors.extend({"query_id": record["query_id"], "error": value} for value in request_errors)
        if index % 20 == 0:
            print(f"processed={index} candidates={len(output)} errors={len(errors)}")
    write_jsonl(args.output, output)
    if errors:
        write_jsonl(args.output.with_suffix(".errors.jsonl"), errors)
    print(json.dumps({"candidates": len(output), "errors": len(errors)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine broad candidate pools from academic APIs")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--actions-per-query", type=int, default=3)
    parser.add_argument("--results-per-action", type=int, default=30)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

