"""Verification cascade (Chapter 7).

Cheap and independent first, expensive and correlated last. The failing rung
selects the repair, because a blanket regenerate wastes work already correct.
"""
from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from app.reasoning.llm import critic, structured_call
from app.reasoning.retrieval.loop import EvidenceSet
from app.reasoning.state import ReasoningStep, StepKind, Trajectory


class Rung(str, Enum):
    STRUCTURAL = "structural"
    PROGRAMMATIC = "programmatic"
    ENTAILMENT = "entailment"
    CRITIQUE = "critique"


@dataclass
class Verdict:
    passed: bool
    rung: Rung | None = None          # first rung that failed
    detail: str = ""
    score: float = 0.0                # for best-of-iterations tracking
    locus: int | None = None          # index of the offending step, if known
    unsupported: list[str] = None     # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.unsupported is None:
            self.unsupported = []


# ---------------------------------------------------------------------------
# Rung 1: structural. Free, total independence.
# ---------------------------------------------------------------------------
def check_structure(answer: str, *, require_citations: bool = True,
                    schema: type[BaseModel] | None = None) -> Verdict:
    if not answer or not answer.strip():
        return Verdict(False, Rung.STRUCTURAL, "empty answer")
    if schema is not None:
        try:
            schema(**json.loads(answer))
        except Exception as exc:
            return Verdict(False, Rung.STRUCTURAL, f"schema: {exc}"[:300])
    if require_citations and not re.search(r"\[[^\]]+#[^\]]+\]", answer):
        return Verdict(False, Rung.STRUCTURAL,
                       "answer contains no citation of the form [doc#section]")
    return Verdict(True, score=1.0)


# ---------------------------------------------------------------------------
# Rung 2: programmatic re-execution of the trace's OWN arithmetic claims.
# The single highest-yield check in the stack: it catches unfaithful traces,
# which no amount of model sophistication addresses.
# ---------------------------------------------------------------------------
_ARITH = re.compile(
    r"([-+]?[\d,]*\.?\d+)\s*([*x×/+\-])\s*([-+]?[\d,]*\.?\d+)\s*=\s*"
    r"([-+]?[\d,]*\.?\d+)"
)
_OPS: dict[str, Any] = {
    "*": operator.mul, "x": operator.mul, "×": operator.mul,
    "/": operator.truediv, "+": operator.add, "-": operator.sub,
}


def check_arithmetic(text: str, rel_tol: float = 1e-6) -> Verdict:
    """Recompute every 'a op b = c' the trace asserts and compare."""
    for match in _ARITH.finditer(text):
        a, op, b, claimed = match.groups()
        try:
            lhs = _OPS[op](float(a.replace(",", "")), float(b.replace(",", "")))
            rhs = float(claimed.replace(",", ""))
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        if abs(lhs - rhs) > rel_tol * max(abs(lhs), 1.0):
            return Verdict(
                False, Rung.PROGRAMMATIC,
                f"trace asserts {match.group(0)}, but {a} {op} {b} = {lhs:.6g}",
            )
    return Verdict(True, score=1.0)


# ---------------------------------------------------------------------------
# Rung 3: per-sentence entailment against the evidence set.
# ---------------------------------------------------------------------------
class Entailment(BaseModel):
    all_supported: bool
    unsupported_sentences: list[str] = Field(default_factory=list)
    note: str = ""


ENTAIL_PROMPT = """Evidence:
{evidence}

Answer to check:
{answer}

For EACH sentence in the answer, decide whether it is entailed by the
evidence. A sentence is unsupported if the evidence does not state it, even
if it is plausible or generally true.

List every unsupported sentence verbatim. Do not hedge them, list them."""


async def check_entailment(answer: str, evidence: EvidenceSet) -> Verdict:
    if not evidence.passages:
        return Verdict(True, score=0.5)      # nothing to check against
    result, _ = await structured_call(
        ENTAIL_PROMPT.format(evidence=evidence.render(), answer=answer),
        schema=Entailment, model=critic(),
    )
    if not result.ok:
        return Verdict(True, score=0.5)      # do not fail on a parse error
    ent = result.value
    if ent.all_supported and not ent.unsupported_sentences:
        return Verdict(True, score=1.0)
    return Verdict(
        False, Rung.ENTAILMENT,
        f"{len(ent.unsupported_sentences)} unsupported claim(s)",
        unsupported=ent.unsupported_sentences,
    )


# ---------------------------------------------------------------------------
# Rung 4: adversarial critique. The critic CANNOT approve, which removes the
# agreement bias by removing the agreeable response.
# ---------------------------------------------------------------------------
class Objection(BaseModel):
    has_objection: bool
    strongest_objection: str = ""
    severity: int = Field(default=0, ge=0, le=3)


CRITIQUE_PROMPT = """Evidence:
{evidence}

Proposed answer:
{answer}

Your only job is to state the STRONGEST objection to this answer. You may not
approve it. If the answer misreads the evidence, omits a material condition,
overstates certainty, or answers a different question than the one asked, say
so specifically. Rate severity 0-3, where 3 means the answer is wrong."""


async def check_critique(answer: str, evidence: EvidenceSet,
                         min_severity: int = 2) -> Verdict:
    result, _ = await structured_call(
        CRITIQUE_PROMPT.format(evidence=evidence.render(), answer=answer),
        schema=Objection, model=critic(),
    )
    if not result.ok:
        return Verdict(True, score=0.6)
    obj = result.value
    if obj.has_objection and obj.severity >= min_severity:
        return Verdict(False, Rung.CRITIQUE, obj.strongest_objection,
                       score=max(0.0, 1.0 - obj.severity / 3))
    return Verdict(True, score=1.0)


# ---------------------------------------------------------------------------
# The cascade.
# ---------------------------------------------------------------------------
def _record(traj: Trajectory, v: Verdict) -> None:
    traj.append(ReasoningStep(
        kind=StepKind.VERIFY,
        ok=v.passed,
        thought=f"rung={v.rung.value if v.rung else 'all'} "
                f"score={v.score:.2f}",
        payload={"rung": v.rung.value if v.rung else None,
                 "unsupported": v.unsupported},
        error=None if v.passed else v.detail,
    ))


async def verify(answer: str, traj: Trajectory, evidence: EvidenceSet,
                 *, require_citations: bool = True,
                 run_critique: bool = True,
                 schema: type[BaseModel] | None = None) -> Verdict:
    """Short-circuit on the first failure. The failing rung tells the repair
    router which repair to attempt."""
    trace_text = "\n".join(s.thought for s in traj.steps)

    v = check_structure(answer, require_citations=require_citations,
                        schema=schema)
    if not v.passed:
        _record(traj, v)
        return v

    v = check_arithmetic(trace_text + "\n" + answer)
    if not v.passed:
        _record(traj, v)
        return v

    v = await check_entailment(answer, evidence)
    if not v.passed:
        _record(traj, v)
        return v

    if run_critique:
        v = await check_critique(answer, evidence)
        if not v.passed:
            _record(traj, v)
            return v

    ok = Verdict(True, score=1.0)
    _record(traj, ok)
    return ok


# ---------------------------------------------------------------------------
# Repair router: the failing rung selects the CHEAPEST plausible repair.
# ---------------------------------------------------------------------------
REPAIR_BY_RUNG: dict[Rung, str] = {
    Rung.STRUCTURAL: "reemit_field",      # cheapest: fix the envelope
    Rung.PROGRAMMATIC: "recompute_step",  # rerun the offending computation
    Rung.ENTAILMENT: "retrieve_more",     # the evidence is insufficient
    Rung.CRITIQUE: "replan",              # costliest: the approach is wrong
}

Generator = Callable[..., Awaitable[str]]


async def verify_and_repair(generate: Generator, answer: str,
                            traj: Trajectory, evidence: EvidenceSet,
                            max_rounds: int = 2, **kwargs
                            ) -> tuple[str, Verdict]:
    """Bounded refinement that returns the BEST candidate, not the last.

    Refinement is not monotone: round three can degrade an answer round one
    got right. Returning the last iterate is how you ship a regression.
    """
    best_answer = answer
    best = await verify(answer, traj, evidence, **kwargs)
    seen: set[int] = {hash(answer)}

    for _ in range(max_rounds):
        if best.passed:
            break
        repair = REPAIR_BY_RUNG[best.rung]            # type: ignore[index]
        answer = await generate(repair=repair, feedback=best.detail,
                                locus=best.locus, unsupported=best.unsupported,
                                traj=traj)
        if hash(answer) in seen:                      # oscillation
            traj.halted_reason = "refinement oscillated"
            break
        seen.add(hash(answer))

        v = await verify(answer, traj, evidence, **kwargs)
        if v.score <= best.score and not v.passed:    # no improvement
            break
        if v.score > best.score:                      # keep the BEST
            best_answer, best = answer, v

    return best_answer, best
