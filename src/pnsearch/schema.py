from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    KEYWORD_SEARCH = "KEYWORD_SEARCH"
    SEMANTIC_SEARCH = "SEMANTIC_SEARCH"
    QUERY_REWRITE = "QUERY_REWRITE"
    CITATION_FORWARD = "CITATION_FORWARD"
    CITATION_BACKWARD = "CITATION_BACKWARD"
    SIMILAR_PAPER = "SIMILAR_PAPER"
    AUTHOR_EXPANSION = "AUTHOR_EXPANSION"
    READ_FULLTEXT = "READ_FULLTEXT"
    STOP = "STOP"


class DecisionLabel(str, Enum):
    SELECT = "SELECT"
    BORDERLINE = "BORDERLINE"
    REJECT = "REJECT"


class CriterionStatus(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_VIOLATED = "not_violated"


@dataclass(slots=True)
class Criterion:
    id: str
    text: str
    required: bool = True


@dataclass(slots=True)
class MetadataConstraints:
    year_min: int | None = None
    year_max: int | None = None
    venues: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    paper_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QuerySpec:
    original_query: str
    research_intent: str
    inclusion_criteria: list[Criterion]
    exclusion_criteria: list[Criterion]
    metadata: MetadataConstraints = field(default_factory=MetadataConstraints)


@dataclass(slots=True)
class SearchAction:
    type: ActionType
    query: str = ""
    source: str = "semantic_scholar"
    purpose: str = ""
    max_results: int = 20
    seed_paper_id: str | None = None
    exclude_concepts: list[str] = field(default_factory=list)
    feedback_source: str = "initial"


@dataclass(slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str = ""
    authors: list[str] = field(default_factory=list)
    citation_count: int = 0
    url: str = ""
    doi: str = ""
    source: str = ""
    retrieved_by: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PaperJudgment:
    paper_id: str
    label: DecisionLabel
    relevance_score: float
    inclusion_judgments: dict[str, str] = field(default_factory=dict)
    exclusion_judgments: dict[str, str] = field(default_factory=dict)
    evidence_sufficiency: float = 0.0
    reason_codes: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class Feedback:
    positive_expansions: list[str] = field(default_factory=list)
    negative_patterns: list[dict[str, Any]] = field(default_factory=list)
    borderline_needs: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RoundRecord:
    round_index: int
    actions: list[SearchAction]
    retrieved_count: int
    unique_new_count: int
    new_select_count: int
    reject_ratio: float
    duplicate_ratio: float
    api_calls: int
    feedback: Feedback


@dataclass(slots=True)
class SearchState:
    query_spec: QuerySpec
    papers: dict[str, Paper] = field(default_factory=dict)
    judgments: dict[str, PaperJudgment] = field(default_factory=dict)
    history: list[RoundRecord] = field(default_factory=list)
    api_calls: int = 0
    stop_reason: str = ""


@dataclass(slots=True)
class SearchResult:
    query_spec: QuerySpec
    selected: list[tuple[Paper, PaperJudgment]]
    borderline: list[tuple[Paper, PaperJudgment]]
    rejected_count: int
    rounds: list[RoundRecord]
    api_calls: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        def pair_to_dict(pair: tuple[Paper, PaperJudgment]) -> dict[str, Any]:
            paper, judgment = pair
            return {"paper": asdict(paper), "judgment": asdict(judgment)}

        return _jsonable({
            "query_spec": asdict(self.query_spec),
            "highly_relevant": [pair_to_dict(pair) for pair in self.selected],
            "partially_relevant": [pair_to_dict(pair) for pair in self.borderline],
            "rejected_count": self.rejected_count,
            "rounds": [asdict(item) for item in self.rounds],
            "api_calls": self.api_calls,
            "stop_reason": self.stop_reason,
        })


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
