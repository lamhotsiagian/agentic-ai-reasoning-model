"""Chapter 3: plan validation runs BEFORE any side effect."""
import pytest

from app.reasoning.planner import (Plan, PlanStep, PlanValidationError,
                                   Reversibility, ReplanPolicy, validate)

REGISTRY = {"sql_read": object(), "issue_refund": object()}


def test_missing_tool_is_a_capability_gap_not_an_improvisation():
    plan = Plan(goal="g", steps=[PlanStep(sub_problem_id="s1",
                                          tool="nonexistent_tool")])
    with pytest.raises(PlanValidationError, match="unavailable tool"):
        validate(plan, REGISTRY)


def test_irreversible_step_requires_compensation():
    plan = Plan(goal="g", steps=[
        PlanStep(sub_problem_id="s1", tool="issue_refund",
                 reversibility=Reversibility.IRREVERSIBLE),
    ])
    with pytest.raises(PlanValidationError, match="compensating action"):
        validate(plan, REGISTRY)


def test_irreversible_before_reversible_is_rejected():
    plan = Plan(goal="g", steps=[
        PlanStep(sub_problem_id="s1", tool="issue_refund",
                 reversibility=Reversibility.IRREVERSIBLE,
                 compensation="reverse_refund"),
        PlanStep(sub_problem_id="s2", tool="sql_read"),
    ])
    with pytest.raises(PlanValidationError, match="reorder"):
        validate(plan, REGISTRY)


def test_reversible_first_ordering_passes():
    plan = Plan(goal="g", steps=[
        PlanStep(sub_problem_id="s1", tool="sql_read"),
        PlanStep(sub_problem_id="s2", tool="issue_refund",
                 reversibility=Reversibility.IRREVERSIBLE,
                 compensation="reverse_refund"),
    ])
    validate(plan, REGISTRY)          # must not raise


def test_signature_detects_oscillation():
    a = Plan(goal="g", steps=[PlanStep(sub_problem_id="s1", tool="sql_read")])
    b = Plan(goal="g", steps=[PlanStep(sub_problem_id="s1", tool="sql_read")])
    assert a.signature == b.signature


def test_replan_escalates_only_when_forced():
    policy = ReplanPolicy()
    attempts = {"step": 0, "tactic": 0, "strategy": 0}
    assert policy.level_for("tool_error", attempts) == "step"
    attempts["step"] = 2
    assert policy.level_for("tool_error", attempts) == "tactic"
    attempts["tactic"] = 2
    assert policy.level_for("tool_error", attempts) == "strategy"
    attempts["strategy"] = 1
    assert policy.level_for("tool_error", attempts) == "abort"


def test_infeasible_goes_straight_to_strategy():
    policy = ReplanPolicy()
    assert policy.level_for(
        "infeasible", {"step": 0, "tactic": 0, "strategy": 0}) == "strategy"
