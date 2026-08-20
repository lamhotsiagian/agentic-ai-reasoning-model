"""Adaptive test-time compute (Chapter 8).

Estimate difficulty, assign a budget, and stop as soon as the answer is
settled. A single global reasoning budget is always wrong for a non-uniform
traffic mix: too small for the hard slice, wasteful for everything else.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal

from app.reasoning.state import Budget, Trajectory

Tier = Literal["fast", "standard", "deep"]

BUDGETS: dict[Tier, Budget] = {
    "fast":     Budget(max_steps=4,  max_tokens=2_000,  max_wall_ms=5_000,
                       max_tool_calls=2),
    "standard": Budget(max_steps=12, max_tokens=12_000, max_wall_ms=30_000,
                       max_tool_calls=8),
    "deep":     Budget(max_steps=30, max_tokens=48_000, max_wall_ms=180_000,
                       max_tool_calls=25),
}

SIMPLE_TASKS = {"summarize", "classify", "extract", "rewrite", "translate"}
_NUM = re.compile(r"\d[\d,.]*")
_CONSTRAINT = re.compile(
    r"\b(and|but|except|unless|only if|before|after|while|"
    r"more than|less than|between|per|each|compared to)\b", re.I)


def canonical(answer: str) -> str:
    """Normalise for equality comparison across samples."""
    text = answer.strip().lower()
    text = re.sub(r"[\s,]+", " ", text)
    text = re.sub(r"[.!?]+$", "", text)
    # A bare number is the most common comparable answer shape.
    if (m := re.search(r"-?\d+(?:\.\d+)?", text)) and len(text) < 40:
        return m.group(0).rstrip("0").rstrip(".") if "." in m.group(0) \
               else m.group(0)
    return text


def static_difficulty(prompt: str, task_class: str | None) -> Tier | None:
    """Free signals. Returns None when they cannot classify the request."""
    if task_class in SIMPLE_TASKS:
        return "fast"
    numbers = len(_NUM.findall(prompt))
    constraints = len(_CONSTRAINT.findall(prompt))
    if numbers >= 3 or constraints >= 3:
        return "standard"           # interacting constraints need decomposition
    if len(prompt) < 120 and numbers == 0 and constraints == 0:
        return "fast"
    return None                     # ambiguous: pay for a probe


@dataclass
class TierDecision:
    tier: Tier
    gate: str
    detail: dict = field(default_factory=dict)


async def choose_tier(prompt: str, task_class: str | None,
                      probe: Callable[[str], Awaitable[tuple[str, float]]],
                      sample: Callable[[str, int], Awaitable[list[str]]],
                      conf_floor: float = -0.85,
                      history: dict[str, Tier] | None = None) -> TierDecision:
    """Cheap signals first; pay for disagreement only on the residual."""
    # Gate 0: historical difficulty by normalised query template (free lookup).
    if history and (known := history.get(_template(prompt))):
        return TierDecision(known, "history")

    # Gate 1: static signals (free).
    if (tier := static_difficulty(prompt, task_class)) is not None:
        return TierDecision(tier, "static")

    # Gate 2: one fast probe.
    _, mean_logprob = await probe(prompt)
    if mean_logprob >= conf_floor:
        return TierDecision("standard", "probe", {"logprob": mean_logprob})

    # Gate 3: disagreement. Most reliable, most expensive, smallest share.
    answers = await sample(prompt, 3)
    counts = Counter(canonical(a) for a in answers)
    _, count = counts.most_common(1)[0]
    if count == 3:
        return TierDecision("standard", "agreement", {"agree": "3/3"})
    return TierDecision("deep", "agreement", {"agree": f"{count}/3"})


def _template(prompt: str) -> str:
    """Strip literals so structurally identical queries share a key."""
    return re.sub(r"\d+", "#", prompt.lower())[:180]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
@dataclass
class StabilityStopper:
    """Freeze the first answer that survives `patience` consecutive steps.

    Directly targets the overthinking failure mode: a correct answer reached
    early and then abandoned after the model second-guesses itself.
    """

    patience: int = 2
    _last: str | None = None
    _stable_for: int = 0
    frozen: str | None = None

    def observe(self, candidate: str | None) -> bool:
        if candidate is None:
            return False
        canon = canonical(candidate)
        if canon == self._last:
            self._stable_for += 1
        else:
            self._last, self._stable_for = canon, 0
        if self._stable_for >= self.patience:
            self.frozen = candidate
            return True
        return False


async def adaptive_self_consistency(prompt: str, n_max: int,
                                    sample: Callable[[str], Awaitable[str]],
                                    ) -> tuple[str, int]:
    """Stop sampling once the majority is decided beyond reach.

    If the first 3 of a planned 5 agree, samples 4 and 5 cannot change the
    vote, so do not pay for them. Saves 30-40% on agreeable traffic.
    """
    votes: Counter[str] = Counter()
    originals: dict[str, str] = {}
    for i in range(1, n_max + 1):
        raw = await sample(prompt)
        key = canonical(raw)
        votes[key] += 1
        originals.setdefault(key, raw)
        leader, lead_count = votes.most_common(1)[0]
        remaining = n_max - i
        runner_up = max((c for a, c in votes.items() if a != leader), default=0)
        if lead_count > runner_up + remaining:        # mathematically decided
            return originals[leader], i
    leader = votes.most_common(1)[0][0]
    return originals[leader], n_max


async def parallel_self_consistency(prompt: str, n: int,
                                    sample: Callable[[str], Awaitable[str]],
                                    ) -> tuple[str, dict[str, int]]:
    """Fully parallel variant: latency is the max over n, not the sum.

    Note the throughput cost: n concurrent slots per request divides cluster
    capacity by n. Benchmark at production concurrency, not in isolation.
    """
    answers = await asyncio.gather(*[sample(prompt) for _ in range(n)])
    votes = Counter(canonical(a) for a in answers)
    leader = votes.most_common(1)[0][0]
    chosen = next(a for a in answers if canonical(a) == leader)
    return chosen, dict(votes)


def agreement_bucket(votes: dict[str, int], n: int) -> str:
    """The routing signal from Chapter 7's calibration table."""
    top = max(votes.values()) if votes else 0
    return f"{top}/{n}"


def budget_for(tier: Tier, traj: Trajectory) -> Budget:
    traj.tier = tier
    traj.difficulty = {"fast": "easy", "standard": "medium",
                       "deep": "hard"}[tier]         # type: ignore[assignment]
    return BUDGETS[tier]
