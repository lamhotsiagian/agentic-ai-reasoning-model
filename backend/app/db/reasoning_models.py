"""Persistence for the reasoning layer (Chapter 10).

Four kinds of state with four retention policies. The trace store is separate
from conversation state because its retention is a legal decision rather than
an engineering one.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index,
                        Integer, String, Text, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models import Base


class ReasoningDocument(Base):
    """Corpus metadata. `authority` and `superseded_by` are EXPLICIT tiers --
    conflict resolution reads metadata, it does not ask the model to judge."""

    __tablename__ = "reasoning_documents"

    doc_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(512))
    authority: Mapped[int] = mapped_column(Integer, default=0)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime,
                                                            nullable=True)
    superseded_by: Mapped[str | None] = mapped_column(String(128),
                                                      nullable=True)
    language: Mapped[str] = mapped_column(String(8), default="en")
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(),
                                                 onupdate=func.now())


class ReasoningChunk(Base):
    """Parent--child retrieval: embed the child for precision, return the
    parent for context. Metadata is prepended before embedding so a chunk
    never loses the heading that gave it meaning."""

    __tablename__ = "reasoning_chunks"

    chunk_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        ForeignKey("reasoning_documents.doc_id", ondelete="CASCADE"),
        index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    parent_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("ix_chunks_tenant_doc", "tenant_id", "doc_id"),
    )


class ReasoningTrace(Base):
    """Append-only trajectory store. Access-controlled, short retention,
    version-stamped so 'what changed' is a query, not an investigation."""

    __tablename__ = "reasoning_traces"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id",
                                                     ondelete="CASCADE"))
    thread_id: Mapped[UUID | None] = mapped_column(nullable=True)
    tier: Mapped[str] = mapped_column(String(16), index=True)
    goal: Mapped[str] = mapped_column(Text)
    steps: Mapped[dict] = mapped_column(JSONB)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    halted_reason: Mapped[str | None] = mapped_column(String(128),
                                                      nullable=True,
                                                      index=True)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, default=0)
    retrieval_hops: Mapped[int] = mapped_column(Integer, default=0)
    replans: Mapped[int] = mapped_column(Integer, default=0)
    wall_ms: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str] = mapped_column(String(64), default="")
    index_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(),
                                                 index=True)

    __table_args__ = (
        # Token distribution is the LEADING indicator; this index makes the
        # p99-tokens alert query cheap.
        Index("ix_traces_created_tokens", "created_at", "tokens"),
    )


class EvalRun(Base):
    """A stored evaluation report, so regressions are diffable over time."""

    __tablename__ = "reasoning_eval_runs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    suite: Mapped[str] = mapped_column(String(64), index=True)
    tier: Mapped[str] = mapped_column(String(16))
    model_version: Mapped[str] = mapped_column(String(64))
    index_version: Mapped[str] = mapped_column(String(32))
    report: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
