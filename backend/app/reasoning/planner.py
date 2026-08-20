"""Decomposition -> validated, ordered, resource-annotated plan (Chapter 3).

The planner enforces reversible-first ordering and rejects infeasible plans
BEFORE any side effect occurs. A plan that fails validation never runs.
"""
from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import Callable, Literal

from pydantic import BaseModel, Field

from app.reasoning.decomposer import Decomposition, SubProblem
from app.reasoning.state import ReasoningStep, StepKind, Trajectory


class Reversibility(IntEnum):
    """Lower value == safer. Used directly as a sort key."""

    REVERSIBLE = 0        # reads, sandboxed compute, drafts
    RECOVERABLE = 1       # audited writes with an undo path
    COSTLY = 2            # paid quota, provisioning
    IRREVERSIBLE = 3      # payments, customer email, deletion


class PlanStep(BaseModel):
    sub_problem_id: str
    question: str = ""
    tool: str | None = None
    args: dict = Field(default_factory=dict)
    preconditions: list[str] = Field(default_factory=list)
    reversibility: Reversibility = Reversibility.REVERSIBLE
    compensation: str | None = None      # required for RECOVERABLE and above
    est_tokens: int = 500


class Plan(BaseModel):
    goal: str
    steps: list[PlanStep]
    global_constraints: list[str] = Field(default_factory=list)

    @property
    def signature(self) -> str:
        """Stable hash for oscillation detection (plan A -> B -> A)."""
        material = "|".join(
            f"{s.sub_problem_id}:{s.tool}:{sorted(s.args.items())}"
            for s in self.steps
        )
        return hashlib.blake2s(material.encode(), digest_size=8).hexdigest()

    def est_cost(self) -> int:
        return sum(s.est_tokens for s in self.steps)

    def needs_evidence(self, evidence) -> bool:
        return any(s.tool in {"search", "retrieve", "graph_query"}
                   for s in self.steps) and not getattr(evidence, "passages", [])

    def open_subquestions(self) -> list[str]:
        return [s.question or s.sub_problem_id for s in self.steps]


class PlanValidationError(Exception):
    """Raised before execution. A plan that fails validation never runs."""


def validate(plan: Plan, registry: dict[str, object]) -> None:
    """Reject infeasible or unsafe plans BEFORE any side effect occurs."""
    for step in plan.steps:
        if step.tool and step.tool not in registry:
            raise PlanValidationError(
                f"step {step.sub_problem_id} needs unavailable tool "
                f"'{step.tool}'. Report the capability gap, do not improvise"
            )
        if step.reversibility >= Reversibility.RECOVERABLE and not step.compensation:
            raise PlanValidationError(
                f"step {step.sub_problem_id} is {step.reversibility.name} "
                f"but registers no compensating action"
            )

    # Reversible-first: no irreversible act may precede a reversible one.
    levels = [s.reversibility for s in plan.steps]
    if levels != sorted(levels):
        raise PlanValidationError(
            "irreversible step scheduled before reversible work; reorder so "
            "all recoverable work completes and verifies first"
        )


def order(dec: Decomposition,
          classify: Callable[[SubProblem], PlanStep]) -> list[PlanStep]:
    """Topological layers, then reversibility, then fail-fast within a layer."""
    steps: list[PlanStep] = []
    for layer in dec.execution_layers():
        annotated = [classify(sp) for sp in layer]
        # Within a layer: safest first, then cheapest (probe before spend).
        annotated.sort(key=lambda s: (s.reversibility, s.est_tokens))
        steps.extend(annotated)
    return steps


def build_plan(dec: Decomposition, goal: str,
               classify: Callable[[SubProblem], PlanStep],
               registry: dict[str, object],
               traj: Trajectory) -> Plan:
    plan = Plan(goal=goal, steps=order(dec, classify),
                global_constraints=dec.global_constraints)
    validate(plan, registry)
    traj.append(ReasoningStep(
        kind=StepKind.PLAN,
        thought=f"{len(plan.steps)} steps, est {plan.est_cost()} tokens",
        payload={"signature": plan.signature,
                 "order": [s.sub_problem_id for s in plan.steps],
                 "reversibility": [s.reversibility.name for s in plan.steps]},
    ))
    return plan


FailureKind = Literal["tool_error", "predicate", "precondition", "infeasible"]
ReplanLevel = Literal["step", "tactic", "strategy", "abort"]


class ReplanPolicy(BaseModel):
    """Which level absorbs a given failure. Escalate only when forced.

    A system that escalates a transient API timeout to a strategy replan will
    burn its entire budget on a network blip.
    """

    max_step_retries: int = 2
    max_tactic_replans: int = 2
    max_strategy_replans: int = 1

    def level_for(self, failure: FailureKind,
                  attempts: dict[str, int]) -> ReplanLevel:
        if failure == "infeasible":
            return ("strategy" if attempts.get("strategy", 0) <
                    self.max_strategy_replans else "abort")
        if failure in ("tool_error", "predicate") and \
                attempts.get("step", 0) < self.max_step_retries:
            return "step"
        if attempts.get("tactic", 0) < self.max_tactic_replans:
            return "tactic"
        return ("strategy" if attempts.get("strategy", 0) <
                self.max_strategy_replans else "abort")


class OscillationGuard:
    """Forbid re-attempting a plan signature already tried and failed."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def accept(self, plan: Plan) -> bool:
        if plan.signature in self._seen:
            return False
        self._seen.add(plan.signature)
        return True
