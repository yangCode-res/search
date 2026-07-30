from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pnsearch.config import Settings
from pnsearch.pipeline import PNSearchPipeline


async def _search(args: argparse.Namespace) -> None:
    settings = Settings.from_env(args.config)
    if args.mode:
        settings.mode = args.mode
    result = await PNSearchPipeline(settings).search(args.query)
    payload = result.to_dict()
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + "\n", encoding="utf-8")
    else:
        print(content)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pnsearch", description="PN-Search academic paper agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search", help="run iterative academic paper search")
    search_parser.add_argument("query")
    search_parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    search_parser.add_argument("--mode", choices=["heuristic", "llm"])
    search_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "search":
        asyncio.run(_search(args))


if __name__ == "__main__":
    main()

