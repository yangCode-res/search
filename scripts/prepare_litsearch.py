#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pnsearch.datasets import write_jsonl
from pnsearch.text import tokenize


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LitSearch query splits and listwise data")
    parser.add_argument("--query-parquet", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=int, default=20)
    parser.add_argument("--easy-negatives", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install pyarrow first: pip install pyarrow") from exc

    query_rows = pq.read_table(args.query_parquet).to_pylist()
    corpus_dataset = ds.dataset(str(args.corpus_dir), format="parquet")
    corpus_rows = corpus_dataset.to_table(columns=["corpusid", "title", "abstract"]).to_pylist()
    corpus = {
        str(row["corpusid"]): {
            "paper_id": str(row["corpusid"]),
            "title": row.get("title") or "",
            "abstract": row.get("abstract") or "",
        }
        for row in corpus_rows
    }
    postings: dict[str, set[str]] = defaultdict(set)
    for paper_id, paper in corpus.items():
        for token in set(tokenize(f"{paper['title']} {paper['abstract']}")):
            postings[token].add(paper_id)

    splits: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reasoner_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    corpus_ids = sorted(corpus)
    rng = random.Random(args.seed)
    missing_gold = 0

    for index, row in enumerate(query_rows):
        query = str(row["query"])
        query_id = f"litsearch_{index:04d}"
        stratum = f"{row.get('query_set')}|{row.get('specificity')}|{row.get('quality')}"
        split = stratified_split(query_id, stratum)
        gold_ids = {str(value) for value in row.get("corpusids") or []}
        positives = []
        for paper_id in sorted(gold_ids):
            paper = corpus.get(paper_id)
            if paper:
                positives.append(paper)
            else:
                missing_gold += 1
                positives.append({"paper_id": paper_id, "title": "", "abstract": ""})
        query_record = {
            "query_id": query_id,
            "query": query,
            "source": "litsearch",
            "split": split,
            "query_type": "semantic",
            "positive_papers": [
                {"paper_id": item["paper_id"], "title": item["title"]} for item in positives
            ],
            "known_bad": [],
            "relevance_criteria": [],
            "metadata_constraints": {},
            "benchmark_metadata": {
                "query_set": row.get("query_set"),
                "specificity": row.get("specificity"),
                "quality": row.get("quality"),
            },
        }
        splits[split].append(query_record)

        query_tokens = set(tokenize(query))
        overlap_counts: Counter[str] = Counter()
        for token in query_tokens:
            overlap_counts.update(postings.get(token, ()))
        scored = [
            (count / max(1, len(query_tokens)), paper_id)
            for paper_id, count in overlap_counts.items()
            if paper_id not in gold_ids
        ]
        scored.sort(reverse=True)
        hard_ids = [paper_id for _, paper_id in scored[: args.hard_negatives]]
        excluded = gold_ids | set(hard_ids)
        easy_pool = [paper_id for paper_id in corpus_ids if paper_id not in excluded]
        easy_ids = rng.sample(easy_pool, min(args.easy_negatives, len(easy_pool)))

        for paper in positives:
            candidates_by_split[split].append(
                {
                    "query_id": query_id,
                    **paper,
                    "teacher_label": "SELECT",
                    "reason_codes": ["LITSEARCH_GOLD"],
                }
            )
        for paper_id in hard_ids:
            candidates_by_split[split].append(
                {
                    "query_id": query_id,
                    **corpus[paper_id],
                    "teacher_label": "REJECT",
                    "reason_codes": ["CLOSED_CORPUS_HARD_NEGATIVE"],
                }
            )
        for paper_id in easy_ids:
            candidates_by_split[split].append(
                {
                    "query_id": query_id,
                    **corpus[paper_id],
                    "teacher_label": "REJECT",
                    "reason_codes": ["CLOSED_CORPUS_EASY_NEGATIVE"],
                }
            )

        terms = [term for term, _ in Counter(tokenize(query)).most_common(8)]
        reasoner_by_split[split].append(
            {
                "query_id": query_id,
                "query": query,
                "search_history": [],
                "feedback": {},
                "remaining_budget": 3,
                "target_actions": [
                    {
                        "type": "KEYWORD_SEARCH",
                        "query": query,
                        "purpose": "direct high-recall search"
                    },
                    {
                        "type": "SEMANTIC_SEARCH",
                        "query": " ".join(terms),
                        "purpose": "semantic concept search"
                    }
                ],
                "stop": False,
                "provenance": "deterministic_bootstrap; replace or augment with successful search trajectories"
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "princeton-nlp/LitSearch",
        "queries": len(query_rows),
        "corpus": len(corpus),
        "splits": {key: len(value) for key, value in splits.items()},
        "candidates": {key: len(value) for key, value in candidates_by_split.items()},
        "missing_gold": missing_gold,
        "split_policy": "deterministic hash within query_set/specificity/quality strata; 70/15/15",
        "negative_policy": (
            "Non-gold documents are negatives only for LitSearch's closed-corpus supervised split. "
            "Do not reuse this assumption for open-world PaSa/Asta semantic queries."
        ),
    }
    for split in ("train", "validation", "test"):
        write_jsonl(args.output / f"queries_{split}.jsonl", splits[split])
        write_jsonl(args.output / f"candidates_{split}.jsonl", candidates_by_split[split])
        write_jsonl(args.output / f"reasoner_{split}.jsonl", reasoner_by_split[split])
    (args.output / "litsearch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def stratified_split(query_id: str, stratum: str) -> str:
    digest = hashlib.sha1(f"{stratum}|{query_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "validation"
    return "test"
if __name__ == "__main__":
    main()
