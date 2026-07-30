from __future__ import annotations

import re
import unicodedata

from pnsearch.schema import MetadataConstraints, Paper
from pnsearch.text import keyword_overlap


def canonical_key(paper: Paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.casefold().strip()}"
    title = unicodedata.normalize("NFKC", paper.title).casefold()
    title = re.sub(r"[^\w\u4e00-\u9fff]+", "", title)
    return f"title:{title}"


def merge_candidates(existing: dict[str, Paper], incoming: list[Paper]) -> tuple[list[Paper], float]:
    key_to_id = {canonical_key(paper): paper_id for paper_id, paper in existing.items()}
    new_papers: list[Paper] = []
    duplicate_count = 0
    for paper in incoming:
        key = canonical_key(paper)
        if key in key_to_id:
            duplicate_count += 1
            current = existing[key_to_id[key]]
            current.retrieved_by = sorted(set(current.retrieved_by + paper.retrieved_by))
            if not current.abstract and paper.abstract:
                current.abstract = paper.abstract
            if not current.doi and paper.doi:
                current.doi = paper.doi
            continue
        key_to_id[key] = paper.paper_id
        existing[paper.paper_id] = paper
        new_papers.append(paper)
    ratio = duplicate_count / len(incoming) if incoming else 0.0
    return new_papers, ratio


def apply_hard_filters(papers: list[Paper], metadata: MetadataConstraints) -> list[Paper]:
    result = []
    venues = {item.casefold() for item in metadata.venues}
    for paper in papers:
        if metadata.year_min is not None and paper.year is not None and paper.year < metadata.year_min:
            continue
        if metadata.year_max is not None and paper.year is not None and paper.year > metadata.year_max:
            continue
        if venues and paper.venue and paper.venue.casefold() not in venues:
            continue
        result.append(paper)
    return result


def coarse_rank(query: str, papers: list[Paper], limit: int) -> list[Paper]:
    def score(paper: Paper) -> tuple[float, int]:
        lexical = keyword_overlap(query, f"{paper.title}. {paper.abstract}")
        authority = min(paper.citation_count, 1000)
        return lexical, authority

    return sorted(papers, key=score, reverse=True)[:limit]

