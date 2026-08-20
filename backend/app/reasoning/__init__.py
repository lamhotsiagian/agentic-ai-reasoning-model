"""Reasoning engine for the agentic AI reasoning-model reference project.

Layering (each module maps to one chapter of the ebook):

    state.py        Ch.1   trajectory, steps, budget envelope
    router.py       Ch.1   fast/deliberate tier routing
    decomposer.py   Ch.2   goal -> validated DAG of sub-problems
    hybrid.py       Ch.2   plan-then-bounded-ReAct solve loop
    planner.py      Ch.3   ordering, validation, replan policy
    progress.py     Ch.3   non-convergence detection
    clarify.py      Ch.3   ambiguity detection by action divergence
    search.py       Ch.4   Best-of-N / beam / Tree-of-Thoughts
    tools/          Ch.5   typed contracts, scoped registry, sandboxes
    retrieval/      Ch.6   hybrid channels, RRF, sufficiency-gated loop
    verify.py       Ch.7   verification cascade + repair router
    compute.py      Ch.8   adaptive budgets, early stopping
    agents/         Ch.9   supervisor over typed specialists
    guardrails.py   Ch.10  input/output guardrails
    tracing.py      Ch.10  trajectory persistence + OTel spans
    routes.py       Ch.10  FastAPI surface
    eval/           Ch.10  decomposed evaluation harness
"""

from app.reasoning.state import Budget, ReasoningStep, StepKind, Trajectory

__all__ = ["Budget", "ReasoningStep", "StepKind", "Trajectory"]
