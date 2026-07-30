#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pnsearch.datasets import iter_json_records
from pnsearch.evaluation import evaluate_ids, evaluate_titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PN-Search predictions against known gold sets")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    queries = {str(item["query_id"]): item for item in iter_json_records(args.queries)}
    metrics = []
    for prediction in iter_json_records(args.predictions):
        query = queries.get(str(prediction.get("query_id")))
        if not query:
            continue
        results = prediction.get("results") or prediction.get("highly_relevant") or []
        predicted_ids = [str(item.get("paper_id") or "") for item in results]
        predicted_titles = [str(item.get("title") or "") for item in results]
        gold = query.get("positive_papers") or []
        gold_ids = [str(item.get("paper_id") or "") for item in gold]
        gold_titles = [str(item.get("title") or "") for item in gold]
        result = evaluate_ids(predicted_ids, gold_ids) if any(gold_ids) else evaluate_titles(predicted_titles, gold_titles)
        metrics.append(result)
    if not metrics:
        raise SystemExit("No matching predictions and queries")
    fields = asdict(metrics[0]).keys()
    aggregate = {key: sum(getattr(item, key) for item in metrics) / len(metrics) for key in fields}
    aggregate["queries"] = len(metrics)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
