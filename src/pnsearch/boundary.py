from __future__ import annotations

from pnsearch.schema import DecisionLabel


_EVIDENCE_ABSENCE_MARKERS = (
    "does not mention",
    "doesn't mention",
    "no mention",
    "not explicitly",
    "does not specify",
    "abstract lacks",
    "insufficient evidence",
    "evidence is insufficient",
    "unclear whether",
    "未提及",
    "没有明确",
    "证据不足",
)

_EXPLICIT_MISMATCH_MARKERS = (
    "unrelated",
    "different field",
    "different task",
    "rather than",
    "outside the scope",
    "refers to",
    "not image retrieval",
    "not semantic segmentation",
    "explicitly contradict",
    "无关",
    "不同领域",
    "而非",
    "不是图像检索",
)


def normalize_evidence_boundary(
    label: DecisionLabel, rationale: str | None
) -> DecisionLabel:
    """Keep explicit mismatches negative, but treat abstract-only uncertainty as borderline."""
    if label != DecisionLabel.REJECT or not rationale:
        return label
    text = rationale.casefold()
    evidence_absent = any(marker in text for marker in _EVIDENCE_ABSENCE_MARKERS)
    explicit_mismatch = any(marker in text for marker in _EXPLICIT_MISMATCH_MARKERS)
    if evidence_absent and not explicit_mismatch:
        return DecisionLabel.BORDERLINE
    return label
