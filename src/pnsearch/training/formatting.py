from __future__ import annotations

import json
from typing import Any


RERANKER_SYSTEM = (
    "你是双边界学术论文 Listwise Reranker。根据查询和规则，将论文分为 SELECT、BORDERLINE、REJECT，"
    "逐项判断纳入与排除条件，并按相关性排序。只输出 JSON。"
)

REASONER_SYSTEM = (
    "你是高召回学术搜索 Reasoner。根据查询、搜索历史和正负反馈生成互补且不重复的下一轮搜索动作，"
    "在边际收益不足或预算耗尽时停止。只输出 JSON。"
)


def reranker_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    user = {
        "query": example["query"],
        "criteria": example.get("criteria") or {},
        "candidates": [
            {key: item.get(key) for key in ("paper_id", "title", "abstract", "year", "venue")}
            for item in example["candidates"]
        ],
    }
    results = []
    for item in example["candidates"]:
        results.append(
            {
                "paper_id": item.get("paper_id"),
                "label": item.get("label"),
                "reason_codes": item.get("reason_codes") or [],
            }
        )
    assistant = {"results": results, "ranking": example.get("target_ranking") or []}
    return [
        {"role": "system", "content": RERANKER_SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
    ]


def reasoner_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    user = {
        "query": example["query"],
        "search_history": example.get("search_history") or [],
        "feedback": example.get("feedback") or {},
        "remaining_budget": example.get("remaining_budget"),
    }
    assistant = {
        "actions": example.get("target_actions") or [],
        "stop": bool(example.get("stop", False)),
    }
    return [
        {"role": "system", "content": REASONER_SYSTEM},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
    ]

