"""Chapter 1: the budget must be enforced OUTSIDE the model."""
import time

import pytest

from app.reasoning.state import Budget, ReasoningStep, StepKind, Trajectory


def _step(kind=StepKind.REASON, tokens=0, **kw):
    return ReasoningStep(kind=kind, completion_tokens=tokens, **kw)


def test_step_budget_halts():
    traj = Trajectory(goal="x")
    budget = Budget(max_steps=3, max_tokens=10_000, max_wall_ms=60_000)
    for _ in range(3):
        traj.append(_step())
    assert "step budget exhausted" in (budget.exceeded(traj) or "")


def test_token_budget_halts():
    traj = Trajectory(goal="x")
    budget = Budget(max_steps=99, max_tokens=100)
    traj.append(_step(tokens=150))
    assert "token budget exhausted" in (budget.exceeded(traj) or "")


def test_tool_call_budget_counts_only_tool_steps():
    traj = Trajectory(goal="x")
    budget = Budget(max_steps=99, max_tokens=99_999, max_tool_calls=2)
    traj.append(_step(StepKind.REASON))
    traj.append(_step(StepKind.TOOL))
    assert budget.exceeded(traj) is None
    traj.append(_step(StepKind.TOOL))
    assert "tool-call budget exhausted" in (budget.exceeded(traj) or "")


def test_wall_clock_budget_is_the_backstop():
    traj = Trajectory(goal="x")
    traj.started_at = time.time() - 5
    assert "wall-clock" in (Budget(max_wall_ms=1_000).exceeded(traj) or "")


def test_loop_signature_detects_identical_repeats():
    traj = Trajectory(goal="x")
    for _ in range(3):
        traj.append(_step(StepKind.TOOL,
                          payload={"tool": "sql_read", "args": {"q": "a"}}))
    assert traj.loop_signature(window=3) is True


def test_loop_signature_ignores_varied_calls():
    traj = Trajectory(goal="x")
    for i in range(3):
        traj.append(_step(StepKind.TOOL,
                          payload={"tool": "sql_read", "args": {"q": i}}))
    assert traj.loop_signature(window=3) is False
