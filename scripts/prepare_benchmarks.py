#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pnsearch.datasets import load_asta, load_pasa, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize PaSa and AstaBench query datasets")
    parser.add_argument("--pasa-root", type=Path)
    parser.add_argument("--asta", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    if args.pasa_root:
        records.extend(load_pasa(args.pasa_root))
    if args.asta:
        records.extend(load_asta(args.asta))
    if not records:
        raise SystemExit("No benchmark records found. Check --pasa-root and --asta paths.")

    seen: set[tuple[str, str]] = set()
    deduplicated = []
    for item in records:
        key = (item.source, item.query_id)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)

    counts = Counter(item.split for item in deduplicated)
    source_counts = Counter(item.source for item in deduplicated)
    for split in ("train", "validation", "test"):
        split_records = [item.to_dict() for item in deduplicated if item.split == split]
        write_jsonl(args.output / f"queries_{split}.jsonl", split_records)

    manifest = {
        "total": len(deduplicated),
        "splits": dict(counts),
        "sources": dict(source_counts),
        "leakage_policy": {
            "unit": "query_id",
            "pasa": "preserve official AutoScholarQuery train/dev/test and RealScholarQuery test",
            "asta": "preserve official validation/test; never use test labels for training",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

