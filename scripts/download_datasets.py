#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ASTA_REVISION = "a600dc767f850385f4664772e3ba7a7f8be17d5e"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official PN-Search benchmark data")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--include-paper-db", action="store_true")
    parser.add_argument("--hf-token")
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: pip install huggingface_hub") from exc

    args.data_root.mkdir(parents=True, exist_ok=True)
    pasa_patterns = [
        "AutoScholarQuery/*.jsonl",
        "RealScholarQuery/*.jsonl",
        "sft_crawler/*.jsonl",
        "sft_selector/*.jsonl",
        "paper_database/id2paper.json",
    ]
    if args.include_paper_db:
        pasa_patterns.append("paper_database/*.zip")
    pasa_dir = args.data_root / "raw" / "pasa"
    asta_dir = args.data_root / "raw" / "asta-bench"
    snapshot_download(
        repo_id="CarlanLark/pasa-dataset",
        repo_type="dataset",
        local_dir=pasa_dir,
        allow_patterns=pasa_patterns,
        token=args.hf_token,
    )
    snapshot_download(
        repo_id="allenai/asta-bench",
        repo_type="dataset",
        revision=ASTA_REVISION,
        local_dir=asta_dir,
        allow_patterns=["tasks/paper_finder_bench/*.json"],
        token=args.hf_token,
    )
    manifest = {
        "pasa": str(pasa_dir),
        "asta": str(asta_dir),
        "include_paper_db": args.include_paper_db,
        "asta_revision": ASTA_REVISION,
    }
    (args.data_root / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

