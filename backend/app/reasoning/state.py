"""Core reasoning state: the typed trajectory and its budget envelope.

Chapter 1. This module is deliberately dependency-free (pydantic only) so
that every other component (planner, searcher, tool router, verifier,
evaluator) can import it without a cycle.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    """Every unit of work a reasoning run can perform.

    Kinds are closed on purpose: an unknown kind means the orchestrator
    produced a step no evaluator knows how to score, which is a bug.
    """

    DECOMPOSE = "decompose"
    PLAN = "plan"
    RETRIEVE = "retrieve"
    TOOL = "tool"
    REASON = "reason"
    VERIFY = "verify"
    REPLAN = "replan"
    ANSWER = "answer"


class ReasoningStep(BaseModel):
    """One append-only entry in a trajectory."""

    step_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: StepKind
    thought: str = ""                        # model-visible deliberation
    payload: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None           # tool / retrieval result
    ok: bool = True
    error: str | None = None
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: float = Field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Budget(BaseModel):
    """The externally enforced spending envelope for one run.

    The controller is external to the model on purpose. A model asked to
    police its own budget keeps deliberating. That is the behaviour RL on
    outcome reward selects for.
    """

    max_steps: int = 12
    max_tokens: int = 24_000
    max_wall_ms: int = 60_000
    max_tool_calls: int = 8

    def exceeded(self, traj: "Trajectory") -> str | None:
        """Return a human-readable reason, or None if the run may continue."""
        if len(traj.steps) >= self.max_steps:
            return f"step budget exhausted ({self.max_steps})"
        if traj.tokens_used >= self.max_tokens:
            return f"token budget exhausted ({self.max_tokens})"
        if traj.elapsed_ms >= self.max_wall_ms:
            return f"wall-clock budget exhausted ({self.max_wall_ms} ms)"
        if traj.tool_calls >= self.max_tool_calls:
            return f"tool-call budget exhausted ({self.max_tool_calls})"
        return None


class Trajectory(BaseModel):
    """Append-only record of a single reasoning run.

    This object is the unit of debugging, the unit of evaluation, and the
    payload the UI renders as an execution trace.
    """

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tier: Literal["fast", "standard", "deep"] = "standard"
    steps: list[ReasoningStep] = Field(default_factory=list)
    final_answer: str | None = None
    halted_reason: str | None = None
    citations: list[str] = Field(default_factory=list)
    model_version: str = ""
    index_version: str = ""
    started_at: float = Field(default_factory=time.time)

    # ---- derived metrics -------------------------------------------------
    @property
    def tokens_used(self) -> int:
        return sum(s.total_tokens for s in self.steps)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind is StepKind.TOOL)

    @property
    def retrieval_hops(self) -> int:
        return sum(1 for s in self.steps if s.kind is StepKind.RETRIEVE)

    @property
    def replans(self) -> int:
        return sum(1 for s in self.steps if s.kind is StepKind.REPLAN)

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def append(self, step: ReasoningStep) -> ReasoningStep:
        self.steps.append(step)
        return step

    def last(self, kind: StepKind) -> ReasoningStep | None:
        for step in reversed(self.steps):
            if step.kind is kind:
                return step
        return None

    def loop_signature(self, window: int = 3) -> bool:
        """True when the last `window` tool steps repeat the same call.

        Cheapest useful loop detector in production: hash the tool name and
        the normalised arguments, and abort on repetition.
        """
        tool_steps = [s for s in self.steps if s.kind is StepKind.TOOL]
        if len(tool_steps) < window:
            return False
        sigs = {
            (s.payload.get("tool"),
             repr(sorted(s.payload.get("args", {}).items())))
            for s in tool_steps[-window:]
        }
        return len(sigs) == 1

    def summary(self) -> dict[str, Any]:
        """Compact metrics record: what the evaluation harness consumes."""
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "steps": len(self.steps),
            "tokens": self.tokens_used,
            "tool_calls": self.tool_calls,
            "retrieval_hops": self.retrieval_hops,
            "replans": self.replans,
            "elapsed_ms": self.elapsed_ms,
            "halted_reason": self.halted_reason,
            "answered": self.final_answer is not None,
        }
