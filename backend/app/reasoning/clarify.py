"""Ambiguity detection by action divergence (Chapter 3).

Do not ask the model whether a request is ambiguous, because it will say no.
Generate k interpretations and compare the ACTIONS they imply.
"""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from app.reasoning.llm import structured_call
from app.reasoning.planner import Reversibility


class Interpretation(BaseModel):
    reading: str
    tool_calls: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    max_reversibility: int = 0

    def action_signature(self) -> str:
        return "|".join(sorted(self.tool_calls) + sorted(self.target_entities))


@dataclass
class ClarifyDecision:
    ask: bool
    question: str | None = None
    reason: str | None = None
    assumption: str | None = None
    alternatives: list[str] = field(default_factory=list)


INTERPRET_PROMPT = """Restate the user's request as ONE concrete
interpretation: the reading, the tool calls it implies, and the entities it
would act on. Do not hedge; commit to a single reading.

Available tools: {tools}
Request: {goal}"""


async def needs_clarification(goal: str, tools: list[str],
                              k: int = 3) -> ClarifyDecision:
    """Ambiguity == interpretations that imply materially different actions."""
    prompt = INTERPRET_PROMPT.format(goal=goal, tools=", ".join(tools))
    results = await asyncio.gather(*[
        structured_call(prompt, schema=Interpretation, temperature=0.8)
        for _ in range(k)
    ])
    reads = [r.value for r, _ in results if r.ok]
    if len(reads) < 2:
        return ClarifyDecision(ask=False)

    # Compare action signatures, not phrasing: two readings that produce the
    # same tool calls with the same arguments are not ambiguous in any way
    # the user would notice.
    sigs = {r.action_signature() for r in reads}
    if len(sigs) == 1:
        return ClarifyDecision(ask=False)

    worst = max(r.max_reversibility for r in reads)
    if worst >= int(Reversibility.IRREVERSIBLE):
        return ClarifyDecision(
            ask=True,
            question=_contrast(reads),
            reason="divergent interpretations imply irreversible actions",
        )

    # Reversible divergence: act on the modal reading, disclose the
    # assumption, and offer the alternative in the response.
    modal = Counter(r.action_signature() for r in reads).most_common(1)[0][0]
    chosen = next(r for r in reads if r.action_signature() == modal)
    return ClarifyDecision(
        ask=False,
        assumption=chosen.reading,
        alternatives=[r.reading for r in reads
                      if r.action_signature() != modal],
    )


def _contrast(reads: list[Interpretation]) -> str:
    """A good clarifying question presents the specific alternatives."""
    options = "; or ".join(f"({i + 1}) {r.reading}"
                           for i, r in enumerate(reads[:3]))
    return (f"That request has more than one reading and the action is not "
            f"reversible. Did you mean {options}?")
