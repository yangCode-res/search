from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import Any

from pnsearch.schema import Paper, SearchAction

from .http import get_json


class AcademicSearchClient(ABC):
    name: str

    @abstractmethod
    async def search(self, action: SearchAction) -> list[Paper]:
        raise NotImplementedError


class SemanticScholarClient(AcademicSearchClient):
    name = "semantic_scholar"
    base_url = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = "", user_agent: str = "pnsearch/0.1", timeout: float = 30.0):
        self.api_key = api_key
        self.user_agent = user_agent
        self.timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def search(self, action: SearchAction) -> list[Paper]:
        if action.type.value.startswith("CITATION_") and action.seed_paper_id:
            endpoint = "citations" if action.type.value == "CITATION_FORWARD" else "references"
            url = f"{self.base_url}/paper/{action.seed_paper_id}/{endpoint}"
            payload = await get_json(
                url,
                params={
                    "limit": action.max_results,
                    "fields": "paperId,title,abstract,year,venue,authors,citationCount,url,externalIds",
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            key = "citingPaper" if endpoint == "citations" else "citedPaper"
            records = [item.get(key) or {} for item in payload.get("data", [])]
        else:
            payload = await get_json(
                f"{self.base_url}/paper/search",
                params={
                    "query": action.query,
                    "limit": min(action.max_results, 100),
                    "fields": "paperId,title,abstract,year,venue,authors,citationCount,url,externalIds",
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            records = payload.get("data", [])
        return [self._parse(item, action) for item in records if item.get("title")]

    def _parse(self, item: dict[str, Any], action: SearchAction) -> Paper:
        external_ids = item.get("externalIds") or {}
        paper_id = str(item.get("paperId") or external_ids.get("DOI") or _stable_id(item.get("title", "")))
        return Paper(
            paper_id=paper_id,
            title=item.get("title") or "",
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            venue=item.get("venue") or "",
            authors=[author.get("name", "") for author in item.get("authors") or [] if author.get("name")],
            citation_count=item.get("citationCount") or 0,
            url=item.get("url") or "",
            doi=external_ids.get("DOI") or "",
            source=self.name,
            retrieved_by=[action.query or action.type.value],
        )


class OpenAlexClient(AcademicSearchClient):
    name = "openalex"
    base_url = "https://api.openalex.org/works"

    def __init__(self, user_agent: str = "pnsearch/0.1", timeout: float = 30.0):
        self.user_agent = user_agent
        self.timeout = timeout

    async def search(self, action: SearchAction) -> list[Paper]:
        payload = await get_json(
            self.base_url,
            params={"search": action.query, "per-page": min(action.max_results, 100)},
            headers={"User-Agent": self.user_agent},
            timeout=self.timeout,
        )
        return [self._parse(item, action) for item in payload.get("results", []) if item.get("title")]

    def _parse(self, item: dict[str, Any], action: SearchAction) -> Paper:
        primary = item.get("primary_location") or {}
        source = primary.get("source") or {}
        doi = (item.get("doi") or "").removeprefix("https://doi.org/")
        paper_id = str(item.get("id") or doi or _stable_id(item.get("title", "")))
        return Paper(
            paper_id=paper_id,
            title=item.get("title") or "",
            abstract=_decode_openalex_abstract(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            venue=source.get("display_name") or "",
            authors=[
                authorship.get("author", {}).get("display_name", "")
                for authorship in item.get("authorships") or []
                if authorship.get("author", {}).get("display_name")
            ],
            citation_count=item.get("cited_by_count") or 0,
            url=primary.get("landing_page_url") or item.get("doi") or "",
            doi=doi,
            source=self.name,
            retrieved_by=[action.query or action.type.value],
        )


class CompositeAcademicClient:
    def __init__(self, clients: list[AcademicSearchClient]):
        self.clients = {client.name: client for client in clients}

    async def execute(self, actions: list[SearchAction]) -> tuple[list[Paper], int, list[str]]:
        jobs = []
        accepted_actions = []
        errors: list[str] = []
        for action in actions:
            client = self.clients.get(action.source)
            if client is None:
                errors.append(f"unknown search source: {action.source}")
                continue
            jobs.append(client.search(action))
            accepted_actions.append(action)
        results = await asyncio.gather(*jobs, return_exceptions=True)
        papers: list[Paper] = []
        for action, result in zip(accepted_actions, results):
            if isinstance(result, Exception):
                errors.append(f"{action.source}:{action.query}: {result}")
            else:
                papers.extend(result)
        return papers, len(accepted_actions), errors


def _decode_openalex_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, token_positions in index.items():
        positions.extend((position, token) for position in token_positions)
    return " ".join(token for _, token in sorted(positions))


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.casefold().strip().encode("utf-8")).hexdigest()[:20]

