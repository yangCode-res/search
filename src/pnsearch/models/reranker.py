from __future__ import annotations

import json
from abc import ABC, abstractmethod

from pnsearch.clients.llm import OpenAICompatibleClient
from pnsearch.schema import DecisionLabel, Paper, PaperJudgment, QuerySpec
from pnsearch.text import compact, keyword_overlap


class ListwiseReranker(ABC):
    @abstractmethod
    async def rank(self, spec: QuerySpec, papers: list[Paper]) -> list[PaperJudgment]:
        raise NotImplementedError


class HeuristicListwiseReranker(ListwiseReranker):
    def __init__(self, select_threshold: float = 0.62, borderline_threshold: float = 0.35):
        self.select_threshold = select_threshold
        self.borderline_threshold = borderline_threshold

    async def rank(self, spec: QuerySpec, papers: list[Paper]) -> list[PaperJudgment]:
        judgments: list[PaperJudgment] = []
        for paper in papers:
            document = f"{paper.title}. {paper.abstract}"
            intent_score = keyword_overlap(spec.research_intent, document)
            criterion_scores = {
                criterion.id: keyword_overlap(criterion.text, document)
                for criterion in spec.inclusion_criteria
            }
            required_scores = [
                criterion_scores[item.id] for item in spec.inclusion_criteria if item.required
            ]
            preferred_scores = [
                criterion_scores[item.id] for item in spec.inclusion_criteria if not item.required
            ]
            required = sum(required_scores) / len(required_scores) if required_scores else intent_score
            preferred = sum(preferred_scores) / len(preferred_scores) if preferred_scores else intent_score
            exclusion_scores = {
                criterion.id: keyword_overlap(criterion.text, document)
                for criterion in spec.exclusion_criteria
            }
            exclusion = max(exclusion_scores.values(), default=0.0)
            evidence = min(1.0, len(paper.abstract) / 500) if paper.abstract else 0.1
            score = max(0.0, min(1.0, 0.45 * intent_score + 0.4 * required + 0.15 * preferred - 0.4 * exclusion))
            hard_constraint_failed = _hard_metadata_failure(spec, paper)
            if hard_constraint_failed or exclusion >= 0.7:
                label = DecisionLabel.REJECT
            elif score >= self.select_threshold and evidence >= 0.25:
                label = DecisionLabel.SELECT
            elif score >= self.borderline_threshold or (score >= 0.25 and evidence < 0.25):
                label = DecisionLabel.BORDERLINE
            else:
                label = DecisionLabel.REJECT
            reasons = []
            if intent_score >= 0.5:
                reasons.append("DIRECT_TASK_MATCH")
            if required >= 0.5:
                reasons.append("REQUIRED_CRITERIA_MATCH")
            if exclusion >= 0.5:
                reasons.append("EXCLUSION_MATCH")
            if not paper.abstract:
                reasons.append("MISSING_ABSTRACT")
            if hard_constraint_failed:
                reasons.append("METADATA_CONSTRAINT_FAILED")
            judgments.append(
                PaperJudgment(
                    paper_id=paper.paper_id,
                    label=label,
                    relevance_score=round(score, 6),
                    inclusion_judgments={
                        key: "satisfied" if value >= 0.45 else "unknown"
                        for key, value in criterion_scores.items()
                    },
                    exclusion_judgments={
                        key: "violated" if value >= 0.7 else "not_violated"
                        for key, value in exclusion_scores.items()
                    },
                    evidence_sufficiency=round(evidence, 6),
                    reason_codes=reasons,
                    rationale="heuristic lexical boundary estimate",
                )
            )
        return sorted(judgments, key=lambda item: item.relevance_score, reverse=True)


class LLMListwiseReranker(ListwiseReranker):
    def __init__(
        self,
        client: OpenAICompatibleClient,
        model: str,
        batch_size: int = 8,
        fallback: ListwiseReranker | None = None,
    ):
        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.fallback = fallback

    async def rank(self, spec: QuerySpec, papers: list[Paper]) -> list[PaperJudgment]:
        all_judgments: list[PaperJudgment] = []
        for offset in range(0, len(papers), self.batch_size):
            batch = papers[offset : offset + self.batch_size]
            try:
                all_judgments.extend(await self._rank_batch(spec, batch))
            except Exception:
                if self.fallback is None:
                    raise
                all_judgments.extend(await self.fallback.rank(spec, batch))
        by_id = {item.paper_id: item for item in all_judgments}
        missing = [paper for paper in papers if paper.paper_id not in by_id]
        if missing and self.fallback:
            for item in await self.fallback.rank(spec, missing):
                by_id[item.paper_id] = item
        return sorted(by_id.values(), key=lambda item: item.relevance_score, reverse=True)

    async def _rank_batch(self, spec: QuerySpec, papers: list[Paper]) -> list[PaperJudgment]:
        candidates = [
            {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "abstract": compact(paper.abstract, 2200),
                "year": paper.year,
                "venue": paper.venue,
            }
            for paper in papers
        ]
        criteria = {
            "inclusion": [
                {"id": item.id, "criterion": item.text, "required": item.required}
                for item in spec.inclusion_criteria
            ],
            "exclusion": [
                {"id": item.id, "criterion": item.text} for item in spec.exclusion_criteria
            ],
        }
        data = await self.client.chat_json(
            model=self.model,
            system=(
                "你是双边界学术论文 Listwise Reranker。只输出 JSON。必须比较整组论文，并逐项检查纳入和排除条件。"
                "SELECT 仅用于摘要有明确证据满足所有必需条件的论文；证据不足用 BORDERLINE；明确不符用 REJECT。"
                "不要因关键词相似而选择论文，不要假设摘要中未陈述的信息。"
            ),
            user=(
                f"原始查询：{spec.original_query}\n规则：{json.dumps(criteria, ensure_ascii=False)}\n"
                f"候选论文：{json.dumps(candidates, ensure_ascii=False)}\n"
                "输出：{results:[{paper_id,label,relevance_score,inclusion_judgments,exclusion_judgments,"
                "evidence_sufficiency,reason_codes,rationale}],ranking:[paper_id]}"
            ),
            max_tokens=4000,
        )
        allowed_ids = {paper.paper_id for paper in papers}
        judgments: list[PaperJudgment] = []
        for item in data.get("results") or []:
            paper_id = str(item.get("paper_id", ""))
            if paper_id not in allowed_ids:
                continue
            try:
                label = DecisionLabel(str(item.get("label", "BORDERLINE")).upper())
            except ValueError:
                label = DecisionLabel.BORDERLINE
            judgments.append(
                PaperJudgment(
                    paper_id=paper_id,
                    label=label,
                    relevance_score=max(0.0, min(1.0, float(item.get("relevance_score", 0.5)))),
                    inclusion_judgments=item.get("inclusion_judgments") or {},
                    exclusion_judgments=item.get("exclusion_judgments") or {},
                    evidence_sufficiency=max(0.0, min(1.0, float(item.get("evidence_sufficiency", 0.5)))),
                    reason_codes=item.get("reason_codes") or [],
                    rationale=item.get("rationale") or "",
                )
            )
        return judgments


def _hard_metadata_failure(spec: QuerySpec, paper: Paper) -> bool:
    metadata = spec.metadata
    if metadata.year_min is not None and paper.year is not None and paper.year < metadata.year_min:
        return True
    if metadata.year_max is not None and paper.year is not None and paper.year > metadata.year_max:
        return True
    if metadata.venues and paper.venue:
        allowed = {venue.casefold() for venue in metadata.venues}
        if paper.venue.casefold() not in allowed:
            return True
    return False
