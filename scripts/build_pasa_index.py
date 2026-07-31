#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import zipfile
from collections import defaultdict
from pathlib import Path

from pnsearch.offline import (
    fallback_paper_id,
    infer_arxiv_year,
    initialize_index,
    insert_papers,
    normalize_pasa_title,
    read_pasa_document,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an SQLite FTS index over PaSa papers")
    parser.add_argument("--paper-zip", type=Path, required=True)
    parser.add_argument("--id-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild and args.output.exists():
        args.output.unlink()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.id_map.open(encoding="utf-8") as handle:
        id_to_title = json.load(handle)
    title_to_ids: dict[str, list[str]] = defaultdict(list)
    for paper_id, title in id_to_title.items():
        title_to_ids[normalize_pasa_title(str(title))].append(str(paper_id))

    stats = {"zip_entries": 0, "parsed": 0, "malformed": 0, "mapped_ids": 0}

    def records():
        with zipfile.ZipFile(args.paper_zip) as archive:
            for index, info in enumerate(archive.infolist(), start=1):
                if args.limit is not None and stats["parsed"] >= args.limit:
                    break
                stats["zip_entries"] += 1
                try:
                    title, abstract, venue = read_pasa_document(archive.read(info))
                except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
                    stats["malformed"] += 1
                    continue
                if not title:
                    stats["malformed"] += 1
                    continue
                normalized = normalize_pasa_title(title)
                ids = title_to_ids.get(normalized) or []
                paper_id = ids[0] if ids else fallback_paper_id(title)
                if ids:
                    stats["mapped_ids"] += 1
                stats["parsed"] += 1
                if index % 10000 == 0:
                    print(json.dumps(stats, ensure_ascii=False), flush=True)
                yield paper_id, title, abstract, infer_arxiv_year(paper_id), venue

    connection = sqlite3.connect(args.output)
    try:
        initialize_index(connection)
        inserted = insert_papers(connection, records())
        count = connection.execute("SELECT count(*) FROM papers").fetchone()[0]
    finally:
        connection.close()
    stats.update({"inserted": inserted, "index_papers": count, "output": str(args.output)})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
