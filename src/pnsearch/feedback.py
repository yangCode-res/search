from __future__ import annotations

from collections import Counter

from pnsearch.schema import DecisionLabel, Feedback, Paper, PaperJudgment
from pnsearch.text import top_terms


def analyze_feedback(
    papers: dict[str, Paper],
    judgments: list[PaperJudgment],
    *,
    query_terms: set[str] | None = None,
) -> Feedback:
    selected = [item for item in judgments if item.label == DecisionLabel.SELECT]
    rejected = [item for item in judgments if item.label == DecisionLabel.REJECT]
    borderline = [item for item in judgments if item.label == DecisionLabel.BORDERLINE]

    selected_texts = [
        f"{papers[item.paper_id].title} {papers[item.paper_id].abstract}"
        for item in selected
        if item.paper_id in papers
    ]
    expansions = top_terms(selected_texts, limit=12)
    if query_terms:
        expansions = [term for term in expansions if term not in query_terms]

    reason_counts = Counter(code for item in rejected for code in item.reason_codes)
    negative_patterns = [
        {"type": code, "count": count}
        for code, count in reason_counts.most_common(8)
        if code not in {"MISSING_ABSTRACT", "METADATA_CONSTRAINT_FAILED"}
    ]
    borderline_needs = [
        {
            "paper_id": item.paper_id,
            "title": papers[item.paper_id].title if item.paper_id in papers else "",
            "missing_evidence": _missing_evidence(item),
            "next_action": "READ_FULLTEXT" if item.evidence_sufficiency < 0.5 else "CITATION_FORWARD",
        }
        for item in sorted(borderline, key=lambda value: value.relevance_score, reverse=True)[:5]
    ]
    return Feedback(
        positive_expansions=expansions,
        negative_patterns=negative_patterns,
        borderline_needs=borderline_needs,
    )


def _missing_evidence(judgment: PaperJudgment) -> str:
    unknown = [key for key, value in judgment.inclusion_judgments.items() if value == "unknown"]
    if unknown:
        return "摘要未充分证明纳入条件：" + ", ".join(unknown)
    return "摘要证据不足，需要正文或引用关系确认"

