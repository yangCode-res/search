#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pnsearch.datasets import iter_json_records, write_jsonl
from pnsearch.evaluation import normalize_title
from pnsearch.offline import PasaOfflineIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine broad candidates from the PaSa FTS index")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-per-query", type=int, default=80)
    parser.add_argument("--strategy", choices=["broad", "tiered", "hybrid"], default="broad")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inject-gold", action="store_true")
    args = parser.parse_args()

    output = []
    stats = {
        "queries": 0,
        "candidates": 0,
        "gold_retrieved": 0,
        "gold_injected": 0,
        "gold_missing": 0,
    }
    with PasaOfflineIndex(args.index) as index:
        for query in iter_json_records(args.queries):
            if args.limit is not None and stats["queries"] >= args.limit:
                break
            query_id = str(query["query_id"])
            candidates = []
            seen_ids: set[str] = set()
            seen_titles: set[str] = set()
            gold_ids = {
                str(item.get("paper_id") or "").casefold()
                for item in query.get("positive_papers") or []
                if item.get("paper_id")
            }
            gold_titles = {
                normalize_title(str(item.get("title") or ""))
                for item in query.get("positive_papers") or []
                if item.get("title")
            }
            for hit in index.search(
                query["query"], args.results_per_query, strategy=args.strategy
            ):
                title_key = normalize_title(hit.paper.title)
                if hit.paper.paper_id in seen_ids or title_key in seen_titles:
                    continue
                seen_ids.add(hit.paper.paper_id)
                seen_titles.add(title_key)
                natural_gold = (
                    hit.paper.paper_id.casefold() in gold_ids or title_key in gold_titles
                )
                candidates.append((hit.paper, hit.score, hit.rank, natural_gold, False))
                stats["gold_retrieved"] += int(natural_gold)
            if args.inject_gold:
                for gold in query.get("positive_papers") or []:
                    gold_id = str(gold.get("paper_id") or "")
                    gold_title = normalize_title(str(gold.get("title") or ""))
                    if gold_id in seen_ids or (gold_title and gold_title in seen_titles):
                        continue
                    paper = index.get_by_id(gold_id) if gold_id else None
                    if paper is None and gold.get("title"):
                        paper = index.get_by_title(str(gold["title"]))
                    if paper is None:
                        stats["gold_missing"] += 1
                        continue
                    candidates.append((paper, 1_000_000.0, 0, True, True))
                    seen_ids.add(paper.paper_id)
                    seen_titles.add(normalize_title(paper.title))
                    stats["gold_injected"] += 1
            for paper, score, rank, gold_match, gold_injected in candidates:
                output.append(
                    {
                        "query_id": query_id,
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "abstract": paper.abstract,
                        "year": paper.year,
                        "venue": paper.venue,
                        "source": "pasa_offline",
                        "retrieval_score": score,
                        "retrieval_rank": rank,
                        "gold_match": gold_match,
                        "gold_injected": gold_injected,
                    }
                )
            stats["queries"] += 1
            stats["candidates"] += len(candidates)
            if stats["queries"] % 100 == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)
    write_jsonl(args.output, output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
