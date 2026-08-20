"""Reasoning settings, mixed into app.config.Settings.

Kept in its own module so the reasoning package can be lifted into another
service without dragging the whole application config with it.
"""
from __future__ import annotations

from pydantic import Field


class ReasoningSettings:
    """Mixin. See app/config.py for how it is composed into Settings."""

    # models (Chapter 9: the critic is deliberately a different model)
    reasoner_model: str = Field(default="qwen2.5:3b")
    critic_model: str = Field(default="qwen3:1.7b")
    vision_model: str = Field(default="qwen2.5-vl:3b")
    reranker_model: str = Field(default="")     # empty -> LLM reranker

    # -- retrieval ---------------------------------------------------------
    rerank_score_floor: float = Field(default=0.0)
    retrieval_max_hops: int = Field(default=3)
    retrieval_rerank_to: int = Field(default=6)
    index_version: str = Field(default="v1")

    # -- graph -------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="reasoning")
    neo4j_database: str = Field(default="neo4j")
    graph_enabled: bool = Field(default=True)

    # -- tools -------------------------------------------------------------
    sql_statement_timeout_ms: int = Field(default=8_000)
    sandbox_wall_s: float = Field(default=5.0)
    sandbox_memory_mb: int = Field(default=256)

    # -- compute policy (Chapter 8) ----------------------------------------
    default_tier: str = Field(default="standard")
    adaptive_routing: bool = Field(default=True)
    early_stop_patience: int = Field(default=2)
    self_consistency_n: int = Field(default=5)

    # -- operational -------------------------------------------------------
    labs_enabled: bool = Field(default=True)   # MUST be false in production
    trace_retention_days: int = Field(default=30)
