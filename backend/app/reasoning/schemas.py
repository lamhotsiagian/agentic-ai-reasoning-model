"""Public request/response contracts for the reasoning API (Chapter 10)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ReasoningRequest(BaseModel):
    prompt: str
    task_class: str | None = None          # "summarize", "analyze", ...
    strategy: Literal["auto", "cot", "least_to_most", "react",
                      "hybrid", "search"] = "auto"
    tier_override: Literal["fast", "standard", "deep"] | None = None
    max_hops: int = 3
    stream_trace: bool = False             # internal/debug builds only


class ProgressEvent(BaseModel):
    """The ONLY reasoning shape that crosses the network boundary."""

    step: str
    label: str
    status: Literal["ok", "retrying", "failed"] = "ok"
    duration_ms: int = 0


class AnswerEvent(BaseModel):
    step: Literal["answer"] = "answer"
    content: str
    citations: list[str] = Field(default_factory=list)
    partial: bool = False
    halted_reason: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class LabRunRequest(BaseModel):
    """Lab endpoints expose knobs the production API does not."""

    prompt: str
    lab: int
    config: dict[str, Any] = Field(default_factory=dict)


class LabStreamEvent(BaseModel):
    """One streaming event from a lab run — the shape the frontend receives."""

    phase: str
    label: str
    status: Literal["started", "completed", "failed", "retrying"] = "started"
    data: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int = 0
    step_index: int = 0
