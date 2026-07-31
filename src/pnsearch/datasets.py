from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(slots=True)
class BenchmarkQuery:
    query_id: str
    query: str
    source: str
    split: str
    query_type: str = "semantic"
    positive_papers: list[dict[str, str]] = field(default_factory=list)
    known_bad: list[str] = field(default_factory=list)
    relevance_criteria: list[dict[str, Any]] = field(default_factory=list)
    metadata_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "source": self.source,
            "split": self.split,
            "query_type": self.query_type,
            "positive_papers": self.positive_papers,
            "known_bad": self.known_bad,
            "relevance_criteria": self.relevance_criteria,
            "metadata_constraints": self.metadata_constraints,
        }


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        data = data.get("dataset") or data.get("data") or [data]
    yield from data


def load_pasa(root: str | Path) -> list[BenchmarkQuery]:
    root = Path(root)
    records: list[BenchmarkQuery] = []
    patterns = [
        ("AutoScholarQuery/train.jsonl", "train", "pasa_auto"),
        ("AutoScholarQuery/dev.jsonl", "validation", "pasa_auto"),
        ("AutoScholarQuery/test.jsonl", "test", "pasa_auto"),
        ("RealScholarQuery/test.jsonl", "test", "pasa_real"),
    ]
    for relative, split, source in patterns:
        path = root / relative
        if not path.exists():
            continue
        for index, item in enumerate(iter_json_records(path)):
            query = _first_text(item, "query", "question", "instruction", "prompt")
            if not query:
                continue
            positive_values = (
                item.get("answer")
                or item.get("answers")
                or item.get("relevant_papers")
                or item.get("papers")
                or item.get("positive")
                or []
            )
            positives = _normalize_positive_papers(positive_values)
            arxiv_ids = item.get("answer_arxiv_id") or []
            if isinstance(arxiv_ids, list):
                for positive, arxiv_id in zip(positives, arxiv_ids):
                    if arxiv_id and not positive["paper_id"]:
                        positive["paper_id"] = str(arxiv_id)
            query_id = str(
                item.get("query_id")
                or item.get("qid")
                or item.get("id")
                or f"{source}_{split}_{index}"
            )
            records.append(
                BenchmarkQuery(
                    query_id=query_id,
                    query=query,
                    source=source,
                    split=split,
                    positive_papers=positives,
                )
            )
    return records


def load_asta(paths: Iterable[str | Path]) -> list[BenchmarkQuery]:
    records: list[BenchmarkQuery] = []
    for raw_path in paths:
        path = Path(raw_path)
        split = "validation" if "validation" in path.name or "dev" in path.name else "test"
        for index, item in enumerate(iter_json_records(path)):
            input_data = item.get("input") or {}
            criteria = item.get("scorer_criteria") or {}
            query = input_data.get("query") or item.get("query") or ""
            query_id = str(input_data.get("query_id") or item.get("query_id") or f"asta_{index}")
            if not query:
                continue
            corpus_ids = criteria.get("corpus_ids") or criteria.get("known_to_be_good") or []
            relevance = criteria.get("relevance_criteria") or []
            if not relevance and criteria.get("relevance_prompt"):
                relevance = [
                    {
                        "name": "semantic_relevance",
                        "description": criteria["relevance_prompt"],
                        "weight": 1.0,
                    }
                ]
            query_type = query_id.split("_", 1)[0] if "_" in query_id else "semantic"
            records.append(
                BenchmarkQuery(
                    query_id=query_id,
                    query=query,
                    source="asta_paper_finder",
                    split=split,
                    query_type=query_type,
                    positive_papers=[{"paper_id": str(value), "title": ""} for value in corpus_ids],
                    known_bad=[str(value) for value in criteria.get("known_to_be_bad") or []],
                    relevance_criteria=relevance,
                    metadata_constraints={
                        key: criteria.get(key)
                        for key in ("venue_must_be_one_of", "author_must_be_one_of", "year_must_be_in_range")
                        if criteria.get(key) is not None
                    },
                )
            )
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def deterministic_split(query_id: str, train: float = 0.8, validation: float = 0.1) -> str:
    bucket = int(hashlib.sha1(query_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    extra = item.get("extra") or {}
    for key in keys:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_positive_papers(values: Any) -> list[dict[str, str]]:
    if isinstance(values, (str, dict)):
        values = [values]
    result: list[dict[str, str]] = []
    for value in values or []:
        if isinstance(value, str):
            result.append({"paper_id": "", "title": value})
        elif isinstance(value, dict):
            result.append(
                {
                    "paper_id": str(
                        value.get("paper_id")
                        or value.get("corpus_id")
                        or value.get("id")
                        or ""
                    ),
                    "title": str(value.get("title") or value.get("name") or ""),
                }
            )
    return [item for item in result if item["paper_id"] or item["title"]]
