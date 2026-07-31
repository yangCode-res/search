#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pnsearch.datasets import iter_json_records, write_jsonl


def parse_selector_record(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = record.get("messages") or []
    user = next(
        (str(item.get("content") or "") for item in messages if item.get("role") == "user"),
        "",
    )
    assistant = next(
        (
            str(item.get("content") or "")
            for item in reversed(messages)
            if item.get("role") == "assistant"
        ),
        "",
    )
    title_match = re.search(r"(?:^|\n)Title:\s*(.*?)\nAbstract:\s*", user, flags=re.DOTALL)
    abstract_match = re.search(
        r"\nAbstract:\s*(.*?)\n\s*User Query:\s*", user, flags=re.DOTALL
    )
    query_match = re.search(
        r"\n\s*User Query:\s*(.*?)(?:\n\s*Output format:|\Z)", user, flags=re.DOTALL
    )
    decision_match = re.search(r"(?:Decision:\s*)?(True|False)\b", assistant, flags=re.I)
    if not (title_match and abstract_match and query_match and decision_match):
        return None
    title = title_match.group(1).strip()
    abstract = abstract_match.group(1).strip()
    query = query_match.group(1).strip()
    if not title or not query:
        return None
    positive = decision_match.group(1).casefold() == "true"
    digest = hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()[:20]
    return {
        "query": query,
        "candidate": {
            "paper_id": f"pasa-selector:{digest}",
            "title": title,
            "abstract": abstract,
            "year": None,
            "venue": "",
            "label": "SELECT" if positive else "REJECT",
            "reason_codes": ["PASA_HUMAN_POSITIVE" if positive else "PASA_HUMAN_NEGATIVE"],
            "teacher_rationale": assistant,
        },
    }


def build_listwise(
    parsed: list[dict[str, Any]], *, list_size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed:
        grouped[item["query"]].append(item["candidate"])
    examples = []
    stats = {"query_groups": len(grouped), "groups_without_both_labels": 0}
    for group_index, (query, candidates) in enumerate(grouped.items()):
        positives = [dict(item) for item in candidates if item["label"] == "SELECT"]
        negatives = [dict(item) for item in candidates if item["label"] == "REJECT"]
        if not positives or not negatives:
            stats["groups_without_both_labels"] += 1
            continue
        rng.shuffle(positives)
        rng.shuffle(negatives)
        positive_take = max(1, list_size // 3)
        positive_chunks = [
            positives[i : i + positive_take]
            for i in range(0, len(positives), positive_take)
        ]
        negative_offset = 0
        for chunk_index, positive_chunk in enumerate(positive_chunks):
            needed = max(1, list_size - len(positive_chunk))
            negative_chunk = [
                negatives[(negative_offset + i) % len(negatives)] for i in range(needed)
            ]
            negative_offset += needed
            batch = positive_chunk + [dict(item) for item in negative_chunk]
            rng.shuffle(batch)
            query_hash = hashlib.sha1(query.encode("utf-8")).hexdigest()[:16]
            examples.append(
                {
                    "query_id": f"pasa-selector:{query_hash}:{group_index}:{chunk_index}",
                    "query": query,
                    "criteria": {},
                    "candidates": batch,
                    "target_ranking": [
                        item["paper_id"]
                        for item in sorted(
                            batch,
                            key=lambda value: value["label"] == "SELECT",
                            reverse=True,
                        )
                    ],
                }
            )
    stats["listwise_examples"] = len(examples)
    return examples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PaSa Selector SFT into PN-Search data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pointwise-output", type=Path, required=True)
    parser.add_argument("--listwise-output", type=Path, required=True)
    parser.add_argument("--list-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    parsed = []
    malformed = 0
    for record in iter_json_records(args.input):
        item = parse_selector_record(record)
        if item is None:
            malformed += 1
        else:
            parsed.append(item)
    pointwise = [
        {
            "query_id": f"pasa-selector-point:{index}",
            "query": item["query"],
            "criteria": {},
            "candidates": [item["candidate"]],
            "target_ranking": [item["candidate"]["paper_id"]],
        }
        for index, item in enumerate(parsed)
    ]
    listwise, stats = build_listwise(parsed, list_size=args.list_size, seed=args.seed)
    write_jsonl(args.pointwise_output, pointwise)
    write_jsonl(args.listwise_output, listwise)
    stats.update(
        {
            "parsed": len(parsed),
            "malformed": malformed,
            "select": sum(item["candidate"]["label"] == "SELECT" for item in parsed),
            "reject": sum(item["candidate"]["label"] == "REJECT" for item in parsed),
            "pointwise_examples": len(pointwise),
        }
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
