from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Settings:
    mode: str = "heuristic"
    max_rounds: int = 3
    max_api_calls: int = 10
    batch_size: int = 8
    candidate_limit: int = 80
    final_limit: int = 20
    search_sources: tuple[str, ...] = ("semantic_scholar", "openalex")
    min_select_score: float = 0.62
    min_borderline_score: float = 0.35
    stop_new_select_threshold: int = 1
    stop_reject_ratio: float = 0.85
    stop_duplicate_ratio: float = 0.8
    llm_base_url: str = "http://127.0.0.1:8000/v1"
    llm_api_key: str = "EMPTY"
    reasoner_model: str = "Qwen/Qwen3-8B"
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    semantic_scholar_api_key: str = ""
    user_agent: str = "pnsearch/0.1"
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls, config_path: str | Path | None = None) -> "Settings":
        values: dict[str, Any] = {}
        if config_path:
            with Path(config_path).open(encoding="utf-8") as handle:
                values.update(json.load(handle))

        env_map = {
            "mode": "PNSEARCH_MODE",
            "max_rounds": "PNSEARCH_MAX_ROUNDS",
            "max_api_calls": "PNSEARCH_MAX_API_CALLS",
            "batch_size": "PNSEARCH_BATCH_SIZE",
            "candidate_limit": "PNSEARCH_CANDIDATE_LIMIT",
            "final_limit": "PNSEARCH_FINAL_LIMIT",
            "llm_base_url": "PNSEARCH_LLM_BASE_URL",
            "llm_api_key": "PNSEARCH_LLM_API_KEY",
            "reasoner_model": "PNSEARCH_REASONER_MODEL",
            "reranker_model": "PNSEARCH_RERANKER_MODEL",
            "semantic_scholar_api_key": "SEMANTIC_SCHOLAR_API_KEY",
            "user_agent": "PNSEARCH_USER_AGENT",
        }
        controller_fallbacks = {
            "llm_base_url": "CL_GISM_CONTROLLER_BASE_URL",
            "llm_api_key": "CL_GISM_CONTROLLER_API_KEY",
            "reasoner_model": "CL_GISM_CONTROLLER_MODEL",
            "reranker_model": "CL_GISM_CONTROLLER_MODEL",
        }
        type_map = {item.name: item.type for item in fields(cls)}
        for key, env_name in env_map.items():
            raw = os.getenv(env_name)
            if raw is None:
                continue
            if key in {"max_rounds", "max_api_calls", "batch_size", "candidate_limit", "final_limit"}:
                values[key] = int(raw)
            else:
                values[key] = raw
        for key, env_name in controller_fallbacks.items():
            if key in values or os.getenv(env_map[key]) is not None:
                continue
            raw = os.getenv(env_name)
            if raw is not None:
                values[key] = raw
        if "search_sources" in values:
            values["search_sources"] = tuple(values["search_sources"])
        allowed = set(type_map)
        return cls(**{key: value for key, value in values.items() if key in allowed})
