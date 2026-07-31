#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from pnsearch.boundary import normalize_evidence_boundary
from pnsearch.datasets import iter_json_records, write_jsonl
from pnsearch.evaluation import normalize_title
from pnsearch.schema import DecisionLabel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query-grouped listwise reranker examples")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--list-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    queries = {str(item["query_id"]): item for item in iter_json_records(args.queries)}
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    stats = defaultdict(int)
    for candidate_path in args.candidates:
        for candidate in iter_json_records(candidate_path):
            query_id = str(candidate["query_id"])
            paper_id = str(candidate.get("paper_id", ""))
            if paper_id in grouped[query_id]:
                stats["duplicate_candidates_removed"] += 1
            grouped[query_id][paper_id] = candidate

    examples = []
    for query_id, query in queries.items():
        candidates = list(grouped.get(query_id, {}).values())
        if not candidates:
            continue
        gold_ids = {
            str(item.get("paper_id", "")).casefold()
            for item in query.get("positive_papers") or []
            if item.get("paper_id")
        }
        gold_titles = {
            normalize_title(item.get("title", ""))
            for item in query.get("positive_papers") or []
            if item.get("title")
        }
        known_bad = {str(value).casefold() for value in query.get("known_bad") or []}
        positives, borderlines, negatives = [], [], []
        for candidate in candidates:
            paper_id = str(candidate.get("paper_id", "")).casefold()
            title_key = normalize_title(candidate.get("title", ""))
            teacher_label = str(candidate.get("teacher_label") or "").upper()
            try:
                teacher_label = normalize_evidence_boundary(
                    DecisionLabel(teacher_label), candidate.get("teacher_rationale")
                ).value
            except ValueError:
                pass
            if paper_id in gold_ids or (title_key and title_key in gold_titles) or teacher_label == "SELECT":
                candidate["label"] = "SELECT"
                positives.append(candidate)
            elif teacher_label == "BORDERLINE":
                candidate["label"] = "BORDERLINE"
                borderlines.append(candidate)
            elif paper_id in known_bad or teacher_label == "REJECT":
                candidate["label"] = "REJECT"
                negatives.append(candidate)
            else:
                # Unjudged papers are not safe negatives and are excluded from supervised training.
                stats["unlabeled_skipped"] += 1
        if not positives or not negatives:
            stats["queries_without_both_boundaries"] += 1
            continue
        rng.shuffle(positives)
        rng.shuffle(borderlines)
        rng.shuffle(negatives)
        while positives and negatives:
            batch = positives[: max(1, args.list_size // 3)]
            positives = positives[len(batch) :]
            take_borderline = min(len(borderlines), max(1, args.list_size // 4))
            batch.extend(borderlines[:take_borderline])
            borderlines = borderlines[take_borderline:]
            needed = args.list_size - len(batch)
            batch.extend(negatives[:needed])
            negatives = negatives[needed:]
            if len(batch) < 2:
                break
            rng.shuffle(batch)
            examples.append(
                {
                    "query_id": query_id,
                    "query": query["query"],
                    "criteria": {
                        "relevance": query.get("relevance_criteria") or [],
                        "metadata": query.get("metadata_constraints") or {},
                    },
                    "candidates": [
                        {
                            key: item.get(key)
                            for key in ("paper_id", "title", "abstract", "year", "venue", "label", "reason_codes")
                        }
                        for item in batch
                    ],
                    "target_ranking": [
                        item.get("paper_id")
                        for item in sorted(
                            batch,
                            key=lambda value: {"SELECT": 2, "BORDERLINE": 1, "REJECT": 0}[value["label"]],
                            reverse=True,
                        )
                    ],
                }
            )
    count = write_jsonl(args.output, examples)
    stats["examples"] = count
    stats["queries"] = len({item["query_id"] for item in examples})
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
