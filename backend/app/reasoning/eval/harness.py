"""The evaluation harness (Chapter 10).

Built on the trajectory object, not on request/response pairs: every metric
except task success is computable directly from a stored trajectory, so an
eval set can be assembled from production traffic by labelling only outcomes.
"""
from __future__ import annotations

import re
import time
from typing import Any

from loguru import logger

from app.reasoning.compute import BUDGETS, canonical
from app.reasoning.eval.fixtures import (TASK_FIXTURES, TOOL_FIXTURES,
                                         TaskFixture, ToolFixture)
from app.reasoning.eval.metrics import (EvalReport, efficiency, percentile,
                                        rate)
from app.reasoning.hybrid import solve_hybrid
from app.reasoning.retrieval.loop import EvidenceSet
from app.reasoning.state import StepKind, Trajectory
from app.reasoning.tools.builtin import build_registry

# Rough local-serving cost model, in dollars per 1k tokens. Replace with your
# own numbers; the point is that every report carries a cost column.
COST_PER_1K_TOKENS = 0.0004


def _correct(fixture: TaskFixture, answer: str) -> bool:
    got = (answer or "").strip()
    if fixture.checker == "exact":
        return canonical(got) == canonical(fixture.answer)
    if fixture.checker == "numeric":
        nums = re.findall(r"-?\d+(?:\.\d+)?", got.replace(",", ""))
        try:
            target = float(fixture.answer.replace(",", ""))
        except ValueError:
            return False
        return any(abs(float(n) - target) <= 1e-2 * max(abs(target), 1.0)
                   for n in nums)
    return fixture.answer.lower() in got.lower()


def _chain_complete(fixture: TaskFixture, traj: Trajectory) -> bool:
    """Ceiling on answer accuracy: no generator can exceed it."""
    if not fixture.required_evidence:
        return True
    seen = " ".join(s.observation or "" for s in traj.steps)
    return all(ev.split("#")[0] in seen for ev in fixture.required_evidence)


def _step_accuracy(traj: Trajectory) -> tuple[int, int]:
    """Programmatic step check: how many reasoning steps survive
    re-execution of their own stated arithmetic."""
    from app.reasoning.verify import check_arithmetic

    checked = ok = 0
    for step in traj.steps:
        if step.kind is not StepKind.REASON or not step.thought:
            continue
        checked += 1
        if check_arithmetic(step.thought).passed:
            ok += 1
    return ok, max(checked, 1)


async def run_tool_fixtures(fixtures: list[ToolFixture] | None = None
                            ) -> dict[str, Any]:
    """Fully automatic; runs in CI on every schema or model change."""
    from app.reasoning.llm import text_call

    fixtures = fixtures or TOOL_FIXTURES
    router = build_registry()
    correct_tool = correct_args = 0

    for fx in fixtures:
        scoped = router.scope(tags=fx.tags)
        prompt = (f"Available tools:\n{router.describe(scoped)}\n\n"
                  f"User request: {fx.request}\n\n"
                  f"Reply with exactly one line:\nTOOL: <name> {{json args}}")
        text, _ = await text_call(prompt)
        m = re.search(r"TOOL:\s*(\w+)", text)
        chosen = m.group(1) if m else ""
        if chosen == fx.expected_tool:
            correct_tool += 1
            if not fx.expected_args_subset:
                correct_args += 1
            else:
                correct_args += int(all(
                    str(v) in text for v in fx.expected_args_subset.values()))

    n = len(fixtures)
    return {"tool_selection": rate("tool_selection", correct_tool, n).as_dict(),
            "argument_accuracy": rate("argument_accuracy", correct_args,
                                      n).as_dict()}


async def run_suite(suite: str = "default", tenant: str = "demo",
                    fixtures: list[TaskFixture] | None = None,
                    tier: str = "standard") -> dict[str, Any]:
    """Run the task suite and return a decomposed, per-slice report."""
    fixtures = fixtures or TASK_FIXTURES
    budget = BUDGETS[tier]
    router = build_registry()

    trajectories: list[Trajectory] = []
    wall: list[int] = []
    by_slice: dict[str, list[bool]] = {}

    success = chain_ok = faithful = 0
    step_ok = step_total = 0
    # The four-way cross-tab from Chapter 6: the (no chain, correct) cell is
    # the one nobody looks at and the most informative.
    crosstab = {"complete_correct": 0, "complete_wrong": 0,
                "incomplete_correct": 0, "incomplete_wrong": 0}

    for fx in fixtures:
        traj = Trajectory(goal=fx.question, difficulty=fx.difficulty)
        started = time.perf_counter()
        try:
            answer = await solve_hybrid(fx.question, router, traj, budget)
        except Exception as exc:
            logger.warning(f"eval item failed: {exc}")
            answer = ""
        elapsed = int((time.perf_counter() - started) * 1000)

        traj.final_answer = answer
        trajectories.append(traj)
        wall.append(elapsed)

        is_correct = _correct(fx, answer)
        is_complete = _chain_complete(fx, traj)
        success += int(is_correct)
        chain_ok += int(is_complete)
        by_slice.setdefault(fx.slice, []).append(is_correct)

        key = (("complete" if is_complete else "incomplete") + "_"
               + ("correct" if is_correct else "wrong"))
        crosstab[key] += 1

        ok, total = _step_accuracy(traj)
        step_ok += ok
        step_total += total

        # Faithfulness: does the trace's own arithmetic reproduce?
        from app.reasoning.verify import check_arithmetic
        faithful += int(check_arithmetic(
            "\n".join(s.thought for s in traj.steps) + "\n" + answer).passed)

    n = len(fixtures)
    total_tokens = sum(t.tokens_used for t in trajectories)
    cost = total_tokens / 1000 * COST_PER_1K_TOKENS

    report = EvalReport(suite=suite)
    report.metrics = [
        rate("task_success_rate", success, n),
        rate("reasoning_step_accuracy", step_ok, step_total),
        rate("retrieval_chain_completeness", chain_ok, n),
        rate("faithfulness", faithful, n),
    ]
    report.per_slice = {
        name: [rate(f"task_success_rate[{name}]", sum(vals), len(vals))]
        for name, vals in by_slice.items()
    }
    report.cost_per_task = cost / n if n else 0.0
    report.cost_per_success = cost / success if success else float("inf")
    report.p50_ms = percentile(wall, 0.50)
    report.p95_ms = percentile(wall, 0.95)
    report.diagnostics = {
        "efficiency": efficiency(trajectories),
        # Read the last cell: an answer that is right WITHOUT the evidence
        # means the model used parametric memory and will fail silently when
        # that memory goes stale.
        "chain_vs_correct": crosstab,
        "tier": tier,
    }
    return report.as_dict()
