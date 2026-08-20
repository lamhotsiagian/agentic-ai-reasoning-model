"""Pluggable search over reasoning states: BoN, beam, and ToT (Chapter 4).

One expansion/evaluation interface so the strategies differ only by policy
and can be A/B'd on the same traffic without changing calling code.
"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from app.reasoning.state import ReasoningStep, StepKind, Trajectory


@dataclass
class SearchNode:
    """A partial reasoning state."""

    steps: list[str] = field(default_factory=list)
    score: float = 0.0
    terminal: bool = False
    answer: str | None = None

    @property
    def signature(self) -> str:
        """Normalised hash for deduplication, the cheapest prune there is."""
        norm = "|".join(s.strip().lower() for s in self.steps)
        return hashlib.blake2s(norm.encode(), digest_size=8).hexdigest()

    def child(self, step: str, terminal: bool = False,
              answer: str | None = None) -> "SearchNode":
        return SearchNode(steps=[*self.steps, step], terminal=terminal,
                          answer=answer)


class Expander(Protocol):
    async def __call__(self, node: SearchNode, k: int) -> list[SearchNode]: ...


class Evaluator(Protocol):
    async def __call__(self, nodes: list[SearchNode]) -> list[float]: ...


@dataclass
class SearchConfig:
    strategy: str = "beam"         # "bon" | "beam" | "tot"
    k: int = 3                     # candidates generated per expansion
    beam_width: int = 2            # survivors per depth
    max_depth: int = 4
    prune_threshold: float = 0.25  # absolute score floor
    max_nodes: int = 40            # hard cost ceiling, enforced outside model
    dedup: bool = True
    batch_eval: bool = True


@dataclass
class SearchReport:
    best: SearchNode
    nodes_expanded: int = 0
    nodes_deduped: int = 0
    depth_reached: int = 0
    eval_calls: int = 0


async def search(root: SearchNode, expand: Expander, evaluate: Evaluator,
                 cfg: SearchConfig, traj: Trajectory) -> SearchReport:
    """Breadth-limited search with structural, duplicate, and score pruning."""
    frontier = [root]
    seen: set[str] = {root.signature}
    best = root
    report = SearchReport(best=root)

    for depth in range(cfg.max_depth):
        # ---- expand -------------------------------------------------------
        batches = await asyncio.gather(*[expand(n, cfg.k) for n in frontier])
        children = [c for batch in batches for c in batch]
        report.nodes_expanded += len(children)
        report.depth_reached = depth + 1

        # ---- free prunes: duplicates first, they cost nothing --------------
        unique: list[SearchNode] = []
        for c in children:
            if not cfg.dedup:
                unique.append(c)
                continue
            if c.signature not in seen:
                seen.add(c.signature)
                unique.append(c)
        report.nodes_deduped += len(children) - len(unique)
        if not unique:
            break

        # ---- evaluate: ONE comparative call, not len(unique) absolute ones -
        if cfg.batch_eval:
            scores = await evaluate(unique)
            report.eval_calls += 1
        else:
            scores = [s[0] for s in await asyncio.gather(
                *[evaluate([n]) for n in unique])]
            report.eval_calls += len(unique)
        for node, s in zip(unique, scores):
            node.score = s

        traj.append(ReasoningStep(
            kind=StepKind.REASON,
            thought=f"search depth {depth}: expanded {len(children)}, "
                    f"deduped to {len(unique)}",
            payload={"scores": [round(float(s), 3) for s in scores],
                     "nodes_used": report.nodes_expanded,
                     "strategy": cfg.strategy},
        ))

        # ---- select --------------------------------------------------------
        survivors = [n for n in unique if n.score >= cfg.prune_threshold]
        survivors.sort(key=lambda n: n.score, reverse=True)

        if cfg.strategy == "bon":
            # No pruning across depth: BoN evaluates only complete answers.
            terminal = [n for n in survivors if n.terminal] or survivors
            report.best = max(terminal, key=lambda n: n.score, default=best)
            return report

        frontier = survivors[: cfg.beam_width]
        if frontier and frontier[0].score > best.score:
            best = frontier[0]
        if not frontier or frontier[0].terminal or \
                report.nodes_expanded >= cfg.max_nodes:
            break

    report.best = best
    return report


# ---------------------------------------------------------------------------
# Default expander / evaluator backed by the reasoner and the critic.
# ---------------------------------------------------------------------------
EXPAND_PROMPT = """Problem: {goal}

Steps taken so far:
{history}

Propose {k} DIFFERENT next steps. Diversity must be structural, not lexical:
vary the approach, not the wording. Mark a step terminal only when it states a
final answer.
Return one step per line, prefixed with "STEP:" or "FINAL:"."""

SCORE_PROMPT = """Problem: {goal}

Score each candidate partial solution from 0.0 to 1.0 on how likely it is to
lead to a correct final answer. Compare them against each other, because relative
judgements are more reliable than absolute ones.

{candidates}

Return one score per line, in order, as a bare decimal number."""


def make_expander(goal: str) -> Expander:
    from app.reasoning.llm import text_call

    async def expand(node: SearchNode, k: int) -> list[SearchNode]:
        history = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(node.steps)) \
                  or "(none yet)"
        text, _ = await text_call(
            EXPAND_PROMPT.format(goal=goal, history=history, k=k),
            temperature=0.9,
        )
        out: list[SearchNode] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("FINAL:"):
                body = line[6:].strip()
                out.append(node.child(body, terminal=True, answer=body))
            elif line.startswith("STEP:"):
                out.append(node.child(line[5:].strip()))
            if len(out) >= k:
                break
        return out

    return expand


def make_evaluator(goal: str) -> Evaluator:
    """Scoring runs on the small model: judging is easier than generating."""
    from app.reasoning.llm import critic, text_call

    async def evaluate(nodes: list[SearchNode]) -> list[float]:
        listing = "\n\n".join(
            f"[{i + 1}] " + " -> ".join(n.steps) for i, n in enumerate(nodes)
        )
        text, _ = await text_call(
            SCORE_PROMPT.format(goal=goal, candidates=listing),
            model=critic(),
        )
        scores: list[float] = []
        for line in text.splitlines():
            try:
                scores.append(max(0.0, min(1.0, float(line.strip().split()[0]))))
            except (ValueError, IndexError):
                continue
        # Degrade to neutral rather than crashing on a malformed score list.
        while len(scores) < len(nodes):
            scores.append(0.5)
        return scores[: len(nodes)]

    return evaluate
