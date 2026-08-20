"""The production solve loop: plan for structure, bounded ReAct per node.

Chapter 2. The plan gives parallelism, inspectability before side effects, and
a global-constraint channel. The inner ReAct loop gives resilience to the API
that returns something unexpected. The step cap on the inner loop is the
critical detail, because unbounded ReAct inside a node reintroduces exactly the
runaway cost the plan was meant to bound.
"""
from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from app.reasoning.decomposer import Decomposition, SubProblem, decompose
from app.reasoning.llm import text_call
from app.reasoning.state import (Budget, ReasoningStep, StepKind, Trajectory)
from app.reasoning.tools.contract import ToolError, ToolOk
from app.reasoning.tools.router import ToolRouter

PREDICATES: dict[str, Any] = {
    "non_empty":   lambda out: bool(out and out.strip()),
    "numeric":     lambda out: any(c.isdigit() for c in out or ""),
    "single_row":  lambda out: (out or "").count("\n") <= 3,
    "json_object": lambda out: (out or "").strip().startswith("{"),
    "boolean":     lambda out: (out or "").strip().lower()[:5]
                               in {"true", "false", "yes", "no"},
}


def passes(node: SubProblem, output: str | None) -> bool:
    check = PREDICATES.get(node.success_predicate, PREDICATES["non_empty"])
    try:
        return bool(check(output))
    except Exception:
        return False


NODE_PROMPT = """You are solving ONE sub-problem of a larger goal.

Sub-problem: {question}

Global constraints that apply to EVERY sub-problem (violating one produces a
locally correct, globally wrong answer):
{constraints}

Results already established:
{established}

Available tools:
{tools}

Work in short steps. To call a tool, emit exactly:
TOOL: <name> {{"arg": "value"}}
When you have the answer to THIS sub-problem only, emit:
ANSWER: <the answer>"""


async def react_node(node: SubProblem, established: dict[str, str],
                     constraints: list[str], router: ToolRouter,
                     traj: Trajectory, max_steps: int = 4) -> str:
    """A ReAct loop bounded INSIDE the node."""
    scoped = router.scope(tags={node.kind, "answer"})
    context = NODE_PROMPT.format(
        question=node.question,
        constraints="\n".join(f"- {c}" for c in constraints) or "- (none)",
        established="\n".join(f"- {k}: {v}" for k, v in established.items())
                    or "- (none yet)",
        tools=router.describe(scoped),
    )

    for _ in range(max_steps):
        text, usage = await text_call(context)
        traj.append(ReasoningStep(
            kind=StepKind.REASON, thought=text[:600],
            payload={"node": node.id},
            latency_ms=usage.latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ))

        if "ANSWER:" in text:
            return text.split("ANSWER:", 1)[1].strip()

        if "TOOL:" in text:
            name, args = _parse_tool_call(text)
            result = await router.call_with_retry(name, args, traj)
            if isinstance(result, ToolError) and not result.model_visible:
                # Orchestrator-owned error: tell the node the capability is
                # gone so it can work around it, without the raw detail.
                context += (f"\n\nOBSERVATION: '{name}' is unavailable for "
                            f"this run. Answer from what you have or use "
                            f"another tool.")
            else:
                observation = (result.data if isinstance(result, ToolOk)
                               else getattr(result, "detail",
                                            getattr(result, "hint", "")))
                context += f"\n\nOBSERVATION: {str(observation)[:2000]}"
            continue

        context += "\n\nYou emitted neither TOOL: nor ANSWER:. Do so now."

    return ""


def _parse_tool_call(text: str) -> tuple[str, dict]:
    import json
    import re

    m = re.search(r"TOOL:\s*(\w+)\s*(\{.*?\})", text, re.S)
    if not m:
        return "", {}
    try:
        return m.group(1), json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1), {}


COMPOSE_PROMPT = """Goal: {goal}

Established sub-results:
{results}

Global constraints (every one must hold in the final answer):
{constraints}

Compose the final answer. Cite the sub-results you used. Do NOT perform
arithmetic here. If a number is needed and not present above, say so."""


async def compose(goal: str, results: dict[str, str], constraints: list[str],
                  traj: Trajectory) -> str:
    text, usage = await text_call(COMPOSE_PROMPT.format(
        goal=goal,
        results="\n".join(f"- {k}: {v}" for k, v in results.items()),
        constraints="\n".join(f"- {c}" for c in constraints) or "- (none)",
    ))
    traj.append(ReasoningStep(
        kind=StepKind.ANSWER, thought="composition",
        latency_ms=usage.latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    ))
    return text.strip()


def compose_partial(goal: str, results: dict[str, str], why: str) -> str:
    """A budget trip returns something useful with an honest caveat."""
    established = "\n".join(f"- {k}: {v}" for k, v in results.items()) \
                  or "- (nothing established)"
    return (f"Partial answer for: {goal}\n\n"
            f"What I established:\n{established}\n\n"
            f"I stopped before finishing because the {why}. The remaining "
            f"sub-problems were not attempted, so treat the above as "
            f"incomplete rather than as a conclusion.")


async def replan_subtree(node: SubProblem, dag: Decomposition,
                         results: dict[str, str], router: ToolRouter,
                         traj: Trajectory) -> str:
    """Scoped replan: re-plan beneath the failing node, keep valid work."""
    traj.append(ReasoningStep(
        kind=StepKind.REPLAN, thought=f"re-planning sub-tree at {node.id}",
        payload={"node": node.id, "kept": sorted(results)},
    ))
    sub = await decompose(node.question, router.names(), traj)
    partial: dict[str, str] = {}
    for layer in sub.execution_layers():
        outs = await asyncio.gather(*[
            react_node(n, {**results, **partial},
                       dag.global_constraints + sub.global_constraints,
                       router, traj, max_steps=3)
            for n in layer
        ], return_exceptions=True)
        for n, out in zip(layer, outs):
            partial[n.id] = "" if isinstance(out, BaseException) else out
    return await compose(node.question, partial,
                         dag.global_constraints, traj)


async def solve_hybrid(goal: str, router: ToolRouter, traj: Trajectory,
                       budget: Budget, max_node_steps: int = 4) -> str:
    """Plan at the top for structure; bounded ReAct inside each node."""
    dag = await decompose(goal, router.names(), traj)
    results: dict[str, str] = {}

    for layer in dag.execution_layers():
        # One gather per topological layer, so independent nodes never wait.
        outs = await asyncio.gather(*[
            react_node(node, results, dag.global_constraints, router, traj,
                       max_steps=max_node_steps)
            for node in layer
        ], return_exceptions=True)

        for node, out in zip(layer, outs):
            if isinstance(out, BaseException):
                logger.warning(f"node {node.id} raised: {out}")
                out = ""
            if not passes(node, out):
                out = await replan_subtree(node, dag, results, router, traj)
            results[node.id] = out

        if (why := budget.exceeded(traj)):
            traj.halted_reason = why
            return compose_partial(goal, results, why)

    return await compose(goal, results, dag.global_constraints, traj)
