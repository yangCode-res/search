from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "using", "with", "paper", "papers", "study",
    "研究", "论文", "方法", "使用", "进行", "相关", "基于", "以及", "系统", "寻找", "搜索",
}


def tokenize(text: str) -> list[str]:
    text = text.casefold()
    latin = re.findall(r"[a-z0-9][a-z0-9_+.-]*", text)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
    chinese: list[str] = []
    for run in chinese_runs:
        if len(run) == 1:
            chinese.append(run)
        else:
            chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
    return [token for token in latin + chinese if token not in STOPWORDS and len(token) > 1]


def keyword_overlap(query: str, document: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    document_tokens = set(tokenize(document))
    return len(query_tokens & document_tokens) / len(query_tokens)


def top_terms(texts: list[str], limit: int = 8) -> list[str]:
    counts = Counter(token for text in texts for token in tokenize(text))
    return [token for token, _ in counts.most_common(limit)]


def compact(text: str, limit: int = 500) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"

