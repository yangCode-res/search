from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pnsearch.schema import ActionType, Paper, SearchAction


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}|\d{4}")
_STOPWORDS = {
    "about", "after", "also", "among", "and", "are", "can", "could", "find",
    "for", "from", "give", "have", "into", "list", "more", "paper", "papers",
    "research", "show", "some", "such", "tell", "that", "the", "their", "these",
    "they", "this", "those", "using", "what", "when", "which", "who", "with",
    "work", "works", "would", "you",
}


@dataclass(slots=True)
class OfflineHit:
    paper: Paper
    score: float
    rank: int


def normalize_pasa_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def infer_arxiv_year(paper_id: str) -> int | None:
    match = re.match(r"(\d{2})(?:\d{2})\.\d+", paper_id)
    if not match:
        return None
    prefix = int(match.group(1))
    return 1900 + prefix if prefix >= 91 else 2000 + prefix


def fallback_paper_id(title: str) -> str:
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:20]
    return f"pasa:{digest}"


def fts_query(text: str, max_terms: int = 18) -> str:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        token = token.casefold().strip("-_")
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return " OR ".join(f'"{term}"' for term in terms)


class PasaOfflineIndex:
    def __init__(self, path: str | Path, *, read_only: bool = True):
        path = Path(path)
        uri = f"file:{path}?mode=ro" if read_only else str(path)
        self.connection = sqlite3.connect(uri, uri=read_only)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "PasaOfflineIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(self, query: str, limit: int = 30) -> list[OfflineHit]:
        expression = fts_query(query)
        if not expression:
            return []
        rows = self.connection.execute(
            """
            SELECT p.paper_id, p.title, p.abstract, p.year, p.venue,
                   bm25(papers_fts, 6.0, 1.0) AS raw_score
            FROM papers_fts
            JOIN papers AS p ON p.rowid = papers_fts.rowid
            WHERE papers_fts MATCH ?
            ORDER BY raw_score
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        return [
            OfflineHit(
                paper=_row_to_paper(row, retrieved_by=[query]),
                score=-float(row["raw_score"]),
                rank=rank,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def get_by_id(self, paper_id: str) -> Paper | None:
        row = self.connection.execute(
            "SELECT paper_id, title, abstract, year, venue FROM papers WHERE paper_id = ?",
            (paper_id,),
        ).fetchone()
        return _row_to_paper(row) if row else None

    def get_by_title(self, title: str) -> Paper | None:
        row = self.connection.execute(
            """
            SELECT paper_id, title, abstract, year, venue
            FROM papers WHERE normalized_title = ? LIMIT 1
            """,
            (normalize_pasa_title(title),),
        ).fetchone()
        return _row_to_paper(row) if row else None


class PasaOfflineSearchClient:
    def __init__(self, index_path: str | Path):
        self.index = PasaOfflineIndex(index_path)

    async def execute(self, actions: Iterable[SearchAction]) -> tuple[list[Paper], int, list[str]]:
        papers: list[Paper] = []
        calls = 0
        errors: list[str] = []
        for action in actions:
            if action.type == ActionType.STOP:
                continue
            if action.type not in {
                ActionType.KEYWORD_SEARCH,
                ActionType.SEMANTIC_SEARCH,
                ActionType.QUERY_REWRITE,
                ActionType.SIMILAR_PAPER,
                ActionType.AUTHOR_EXPANSION,
            }:
                errors.append(f"offline index does not support {action.type.value}")
                continue
            calls += 1
            query = action.query
            if action.seed_paper_id and not query:
                seed = self.index.get_by_id(action.seed_paper_id)
                query = seed.title if seed else ""
            hits = self.index.search(query, action.max_results)
            for hit in hits:
                hit.paper.source = "pasa_offline"
                hit.paper.retrieved_by = [action.query]
                papers.append(hit.paper)
        return papers, calls, errors

    def close(self) -> None:
        self.index.close()


def initialize_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS papers (
            rowid INTEGER PRIMARY KEY,
            paper_id TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            abstract TEXT NOT NULL DEFAULT '',
            year INTEGER,
            venue TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_papers_normalized_title
            ON papers(normalized_title);
        CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
            title, abstract, content='papers', content_rowid='rowid',
            tokenize='porter unicode61'
        );
        """
    )


def insert_papers(
    connection: sqlite3.Connection,
    records: Iterable[tuple[str, str, str, int | None, str]],
    *,
    batch_size: int = 2000,
) -> int:
    count = 0
    batch: list[tuple[str, str, str, str, int | None, str]] = []
    for paper_id, title, abstract, year, venue in records:
        batch.append((paper_id, title, normalize_pasa_title(title), abstract, year, venue))
        if len(batch) >= batch_size:
            count += _insert_batch(connection, batch)
            batch.clear()
    if batch:
        count += _insert_batch(connection, batch)
    connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
    connection.commit()
    return count


def _insert_batch(
    connection: sqlite3.Connection,
    batch: list[tuple[str, str, str, str, int | None, str]],
) -> int:
    before = connection.total_changes
    connection.executemany(
        """
        INSERT OR IGNORE INTO papers
            (paper_id, title, normalized_title, abstract, year, venue)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    connection.commit()
    return connection.total_changes - before


def _row_to_paper(row: sqlite3.Row, retrieved_by: list[str] | None = None) -> Paper:
    return Paper(
        paper_id=str(row["paper_id"]),
        title=str(row["title"]),
        abstract=str(row["abstract"] or ""),
        year=row["year"],
        venue=str(row["venue"] or ""),
        source="pasa_offline",
        retrieved_by=retrieved_by or [],
    )


def read_pasa_document(raw: bytes) -> tuple[str, str, str]:
    data = json.loads(raw)
    return str(data.get("title") or ""), str(data.get("abstract") or ""), str(
        data.get("venue") or ""
    )
