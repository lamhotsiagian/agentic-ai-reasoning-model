"""Tier routing (Chapter 1).

Route by task, not by prestige. A router that sends every request to the
reasoning tier is the most common and most expensive mistake in this space.
"""
from __future__ import annotations

from typing import Literal

from app.reasoning.llm import reasoner
from app.reasoning.schemas import ReasoningRequest

Tier = Literal["fast", "deliberate"]

TIER_BY_TASK: dict[str, Tier] = {
    "summarize": "fast",
    "classify": "fast",
    "extract": "fast",
    "rewrite": "fast",
    "translate": "fast",
    "analyze": "deliberate",
    "plan": "deliberate",
    "debug": "deliberate",
    "compare": "deliberate",
}

CONF_THRESHOLD = -0.85


async def route(req: ReasoningRequest) -> Tier:
    """Cheap signals first; a probe only for the unknown residual."""
    if req.task_class and (tier := TIER_BY_TASK.get(req.task_class)):
        return tier

    from app.reasoning.compute import static_difficulty

    static = static_difficulty(req.prompt, req.task_class)
    if static == "fast":
        return "fast"
    if static in ("standard", "deep"):
        return "deliberate"

    # Unknown task class: one cheap probe, escalate on low confidence.
    model = reasoner(temperature=0.0)
    probe = await model.ainvoke(req.prompt)
    logprob = _mean_logprob(probe)
    return "deliberate" if logprob < CONF_THRESHOLD else "fast"


def _mean_logprob(response) -> float:
    """Ollama does not always return logprobs; fall back to a neutral value
    so the caller degrades to the deliberate tier rather than guessing."""
    meta = getattr(response, "response_metadata", {}) or {}
    if (lp := meta.get("mean_logprob")) is not None:
        return float(lp)
    return CONF_THRESHOLD - 0.01
