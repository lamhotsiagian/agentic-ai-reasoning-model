"""Goal -> validated DAG of typed sub-problems (Chapter 2).

Deliberately separate from the planner, which orders and schedules the DAG.
A decomposition is a property of the question; a plan is a property of the
question AND the current state of the world.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.reasoning.llm import structured_call
from app.reasoning.state import ReasoningStep, StepKind, Trajectory

SubKind = Literal["retrieve", "compute", "tool", "infer"]


class SubProblem(BaseModel):
    """One node of the decomposition DAG."""

    id: str = Field(pattern=r"^s\d{1,2}$")            # s1, s2, ...
    question: str
    kind: SubKind
    depends_on: list[str] = Field(default_factory=list)
    # A machine-evaluable predicate over the node's output. If you cannot
    # write one, the node is too coarse.
    success_predicate: str = "non_empty"
    tool_hint: str | None = None


class Decomposition(BaseModel):
    """The planner's contract: a DAG plus the constraints that cut across it."""

    sub_problems: list[SubProblem]
    # Cross-cutting constraints are injected into EVERY node's context.
    # Omitting them is the classic decomposition bug: locally correct
    # sub-answers that compose into a wrong result.
    global_constraints: list[str] = Field(default_factory=list)

    @field_validator("sub_problems")
    @classmethod
    def _acyclic_and_resolvable(cls, subs: list[SubProblem]) -> list[SubProblem]:
        if not subs:
            raise ValueError("decomposition must contain at least one node")
        ids = {s.id for s in subs}
        if len(ids) != len(subs):
            raise ValueError("duplicate sub-problem ids")
        for s in subs:
            unknown = set(s.depends_on) - ids
            if unknown:
                raise ValueError(f"{s.id} depends on unknown node(s) {unknown}")
            if s.id in s.depends_on:
                raise ValueError(f"{s.id} depends on itself")

        # Kahn's algorithm: a cycle leaves nodes unemitted.
        indeg = {s.id: len(s.depends_on) for s in subs}
        ready = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while ready:
            node = ready.pop()
            seen += 1
            for s in subs:
                if node in s.depends_on:
                    indeg[s.id] -= 1
                    if indeg[s.id] == 0:
                        ready.append(s.id)
        if seen != len(subs):
            raise ValueError("decomposition contains a cycle")
        return subs

    def execution_layers(self) -> list[list[SubProblem]]:
        """Group nodes into parallel-executable layers (topological levels).

        Layer k contains every node whose dependencies all live in layers
        < k, so an orchestrator can gather() each layer concurrently.
        """
        by_id = {s.id: s for s in self.sub_problems}
        depth: dict[str, int] = {}

        def d(node_id: str) -> int:
            if node_id not in depth:
                deps = by_id[node_id].depends_on
                depth[node_id] = 0 if not deps else 1 + max(d(x) for x in deps)
            return depth[node_id]

        layers: dict[int, list[SubProblem]] = {}
        for s in self.sub_problems:
            layers.setdefault(d(s.id), []).append(s)
        return [layers[k] for k in sorted(layers)]

    def open_subquestions(self, answered: set[str] | None = None
                          ) -> list[SubProblem]:
        answered = answered or set()
        return [s for s in self.sub_problems if s.id not in answered]


DECOMPOSE_PROMPT = """Decompose the goal into the FEWEST sub-problems that are
each independently verifiable. Rules:
1. A sub-problem must be answerable by ONE retrieval, ONE tool call, or ONE
   bounded inference. If it needs two, split it.
2. Declare depends_on only for genuine data dependencies, because independent
   run in parallel and must not be serialised.
3. List every constraint that applies across sub-problems in
   global_constraints. Missing one produces locally correct, globally wrong
   answers. Time windows, currency, entity scope, and access filters are the
   usual suspects.
4. success_predicate must be checkable without another model call. Use one of:
   non_empty, numeric, single_row, json_object, boolean.

Goal: {goal}
Available tools: {tools}"""


async def decompose(goal: str, tools: list[str], traj: Trajectory,
                    max_repairs: int = 1) -> Decomposition:
    """Produce a validated DAG, repairing one malformed generation."""
    prompt = DECOMPOSE_PROMPT.format(goal=goal, tools=", ".join(tools))

    for attempt in range(max_repairs + 1):
        result, usage = await structured_call(prompt, schema=Decomposition)
        traj.append(ReasoningStep(
            kind=StepKind.DECOMPOSE,
            thought=f"decomposition attempt {attempt + 1}",
            payload={"nodes": [s.id for s in result.value.sub_problems]}
                    if result.ok else {},
            ok=result.ok,
            error=result.error,
            latency_ms=usage.latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        ))
        if result.ok:
            return result.value
        # Feed the validation error back; repair beats resampling.
        prompt = f"{prompt}\n\nYour previous output was invalid: {result.error}"

    # Degrade honestly rather than inventing a plan.
    return Decomposition(
        sub_problems=[SubProblem(id="s1", question=goal, kind="infer")],
        global_constraints=[],
    )
