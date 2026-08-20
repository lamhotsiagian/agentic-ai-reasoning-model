"""Supervisor over typed specialists communicating via a blackboard (Ch.9).

Reliability compounds multiplicatively, so agents are added only when the
isolation they provide (of context, authority, or perspective) is worth
the arithmetic.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from app.reasoning.decomposer import Decomposition, decompose
from app.reasoning.hybrid import compose, compose_partial, react_node
from app.reasoning.llm import text_call
from app.reasoning.retrieval.loop import EvidenceSet, retrieve_for_reasoning
from app.reasoning.state import (Budget, ReasoningStep, StepKind, Trajectory)
from app.reasoning.tools.builtin import build_registry
from app.reasoning.verify import check_critique, verify


class Role(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REASONER = "reasoner"
    CRITIC = "critic"
    VERIFIER = "verifier"
    EXECUTOR = "executor"


# Negative space is the design: what each role CANNOT touch.
TOOL_SCOPE: dict[Role, set[str]] = {
    Role.PLANNER:    set(),                       # emits data, never acts
    Role.RESEARCHER: {"search_documents", "query_graph", "sql_read"},
    Role.REASONER:   {"python_sandbox"},          # compute only, no I/O
    Role.CRITIC:     set(),
    Role.VERIFIER:   {"python_sandbox"},
    Role.EXECUTOR:   {"sql_write", "send_email", "issue_refund"},
}


class Blackboard(BaseModel):
    """Typed shared state.

    Agents never message each other directly, because free-text hand-offs lose
    information at every hop, because each agent re-summarises what it got.
    """

    goal: str
    tenant: str = "demo"
    dag: Decomposition | None = None
    evidence: EvidenceSet = Field(default_factory=EvidenceSet)
    candidate: str | None = None
    objections: list[str] = Field(default_factory=list)
    verdict: Literal["pending", "pass", "fail"] = "pending"
    rounds: int = 0
    critic_sees_trace: bool = False        # lab toggle; False in production
    trajectory: Trajectory = Field(default_factory=Trajectory)
    budget: Budget = Field(default_factory=Budget)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
async def planner_node(bb: Blackboard) -> Blackboard:
    router = build_registry()
    bb.dag = await decompose(bb.goal, router.names(), bb.trajectory)
    return bb


async def researcher_node(bb: Blackboard) -> Blackboard:
    """Read-only by construction: it cannot be prompt-injected into writing."""
    assert bb.dag is not None
    for sub in bb.dag.sub_problems:
        if sub.kind != "retrieve":
            continue
        found = await retrieve_for_reasoning(sub.question, bb.tenant,
                                             bb.trajectory, max_hops=2)
        bb.evidence.add(found.passages)
    if not bb.evidence.passages:
        # Retrieval found nothing; the reasoner must NOT invent evidence.
        found = await retrieve_for_reasoning(bb.goal, bb.tenant,
                                             bb.trajectory, max_hops=2)
        bb.evidence.add(found.passages)
    return bb


REASON_PROMPT = """Goal: {goal}

Evidence (every claim you make must be entailed by one of these, and must
carry its citation in [doc#section] form):
{evidence}

{objections}

Answer the goal. If the evidence does not support an answer, say exactly what
is missing rather than filling the gap."""


async def reasoner_node(bb: Blackboard) -> Blackboard:
    objections = ("Objections raised against your previous answer:\n"
                  + "\n".join(f"- {o}" for o in bb.objections)
                  if bb.objections else "")
    text, usage = await text_call(REASON_PROMPT.format(
        goal=bb.goal, evidence=bb.evidence.render(), objections=objections))
    bb.candidate = text.strip()
    bb.trajectory.append(ReasoningStep(
        kind=StepKind.REASON, thought="candidate answer drafted",
        latency_ms=usage.latency_ms, prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    ))
    return bb


async def critic_node(bb: Blackboard) -> Blackboard:
    """Adversarial and BLIND to the reasoner's trace.

    Showing the critic the reasoning it is meant to critique correlates its
    errors with the reasoner's, which is exactly the property that makes a
    verifier worthless (Chapter 7). `critic_sees_trace` exists only so the
    lab can demonstrate the degradation.
    """
    evidence = bb.evidence
    if bb.critic_sees_trace:
        logger.warning("critic is seeing the reasoner's trace, so "
                       "error independence is compromised (lab mode)")
    verdict = await check_critique(bb.candidate or "", evidence)
    if not verdict.passed and verdict.detail:
        bb.objections.append(verdict.detail)
    return bb


async def verifier_node(bb: Blackboard) -> Blackboard:
    verdict = await verify(bb.candidate or "", bb.trajectory, bb.evidence,
                           require_citations=bool(bb.evidence.passages),
                           run_critique=False)   # the critic already ran
    bb.verdict = "pass" if verdict.passed else "fail"
    bb.rounds += 1
    if verdict.passed:
        bb.trajectory.final_answer = bb.candidate
        bb.trajectory.citations = bb.evidence.citations()
    return bb


# ---------------------------------------------------------------------------
# Routing. Every exit path is bounded.
# ---------------------------------------------------------------------------
def route(bb: Blackboard) -> str:
    if (why := bb.budget.exceeded(bb.trajectory)):
        bb.trajectory.halted_reason = why
        bb.trajectory.final_answer = bb.candidate or compose_partial(
            bb.goal, {}, why)
        return END
    if bb.dag is None:
        return Role.PLANNER.value
    if not bb.evidence.passages and any(
            s.kind == "retrieve" for s in bb.dag.sub_problems):
        return Role.RESEARCHER.value
    if bb.candidate is None:
        return Role.REASONER.value
    if bb.verdict == "pending":
        return Role.CRITIC.value
    if bb.verdict == "fail" and bb.rounds < 2:
        return Role.REASONER.value          # bounded repair, not open-ended
    return END


def build_supervisor():
    g = StateGraph(Blackboard)
    g.add_node(Role.PLANNER.value, planner_node)
    g.add_node(Role.RESEARCHER.value, researcher_node)
    g.add_node(Role.REASONER.value, reasoner_node)
    g.add_node(Role.CRITIC.value, critic_node)
    g.add_node(Role.VERIFIER.value, verifier_node)

    g.set_conditional_entry_point(route)
    g.add_conditional_edges(Role.PLANNER.value, route)
    g.add_conditional_edges(Role.RESEARCHER.value, route)
    g.add_conditional_edges(Role.REASONER.value, route)
    g.add_edge(Role.CRITIC.value, Role.VERIFIER.value)
    g.add_conditional_edges(Role.VERIFIER.value, route)
    return g.compile()


async def run_single_agent(goal: str, tenant: str, traj: Trajectory,
                           budget: Budget) -> str:
    """The baseline the multi-agent system must beat.

    On tasks that fit one context, this usually wins, so the multi-agent
    decision is a real constraint, not conservatism.
    """
    from app.reasoning.hybrid import solve_hybrid

    return await solve_hybrid(goal, build_registry(), traj, budget)
