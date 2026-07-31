#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pnsearch.datasets import iter_json_records


def analyze(path: Path) -> dict[str, Any]:
    rows = list(iter_json_records(path))
    unique = {
        (str(item.get("query_id", "")), str(item.get("paper_id", ""))): item
        for item in rows
    }
    records = list(unique.values())
    labels = Counter(str(item.get("teacher_label", "UNKNOWN")) for item in records)
    gold = [item for item in records if item.get("gold_match") or item.get("gold_injected")]
    gold_labels = Counter(str(item.get("teacher_label", "UNKNOWN")) for item in gold)
    total = len(records)
    gold_total = len(gold)
    return {
        "path": str(path),
        "rows": len(rows),
        "unique_rows": total,
        "duplicate_rows": len(rows) - total,
        "queries": len({str(item.get("query_id", "")) for item in records}),
        "labels": dict(labels),
        "select_rate": round(labels["SELECT"] / total, 6) if total else 0.0,
        "borderline_rate": round(labels["BORDERLINE"] / total, 6) if total else 0.0,
        "reject_rate": round(labels["REJECT"] / total, 6) if total else 0.0,
        "gold_rows": gold_total,
        "gold_labels": dict(gold_labels),
        "gold_reject_rate": (
            round(gold_labels["REJECT"] / gold_total, 6) if gold_total else None
        ),
        "teacher_models": sorted(
            {str(item["teacher_model"]) for item in records if item.get("teacher_model")}
        ),
        "prompt_versions": sorted(
            {
                str(item["teacher_prompt_version"])
                for item in records
                if item.get("teacher_prompt_version")
            }
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit MiMo teacher label quality")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"datasets": [analyze(path) for path in args.input]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
