from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class RetrievalMetrics:
    precision: float
    recall: float
    f1: float
    recall_at_20: float
    recall_at_50: float
    recall_at_100: float
    ndcg: float


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", title.casefold())


def evaluate_ids(predicted: list[str], gold: Iterable[str]) -> RetrievalMetrics:
    gold_set = {str(value).casefold() for value in gold if value}
    prediction = [str(value).casefold() for value in predicted if value]
    return _evaluate(prediction, gold_set)


def evaluate_titles(predicted: list[str], gold: Iterable[str]) -> RetrievalMetrics:
    gold_set = {normalize_title(value) for value in gold if value}
    prediction = [normalize_title(value) for value in predicted if value]
    return _evaluate(prediction, gold_set)


def _evaluate(prediction: list[str], gold_set: set[str]) -> RetrievalMetrics:
    unique_prediction = list(dict.fromkeys(prediction))
    hits = [1 if value in gold_set else 0 for value in unique_prediction]
    true_positive = sum(hits)
    precision = true_positive / len(unique_prediction) if unique_prediction else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def recall_at(k: int) -> float:
        return sum(hits[:k]) / len(gold_set) if gold_set else 0.0

    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = [1] * min(len(gold_set), len(hits))
    idcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        recall_at_20=recall_at(20),
        recall_at_50=recall_at(50),
        recall_at_100=recall_at(100),
        ndcg=ndcg,
    )

