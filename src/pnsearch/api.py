from __future__ import annotations

import os
from functools import lru_cache

from pnsearch.config import Settings
from pnsearch.pipeline import PNSearchPipeline

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without API extra
    raise RuntimeError("Install API dependencies with: pip install -e '.[api]'") from exc


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)


@lru_cache(maxsize=1)
def get_pipeline() -> PNSearchPipeline:
    config = os.getenv("PNSEARCH_CONFIG", "configs/default.json")
    return PNSearchPipeline(Settings.from_env(config))


app = FastAPI(title="PN-Search API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
async def search(request: SearchRequest) -> dict:
    try:
        result = await get_pipeline().search(request.query)
        return result.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
