"""Trace redaction (Chapter 1).

Streaming a raw reasoning trace to the browser is a data-exfiltration path.
The trace routinely contains retrieved fragments the user is not entitled to
see, internal tool schemas, and verbatim policy text.
"""
from __future__ import annotations

from app.reasoning.state import ReasoningStep, StepKind

# Whitelist, never blacklist: an unrecognised step kind is dropped, so a new
# internal step type cannot leak by default.
SAFE_KINDS: set[StepKind] = {
    StepKind.PLAN,
    StepKind.RETRIEVE,
    StepKind.TOOL,
    StepKind.VERIFY,
    StepKind.ANSWER,
}

PUBLIC_LABELS: dict[StepKind, str] = {
    StepKind.PLAN: "Planning the approach",
    StepKind.RETRIEVE: "Searching the knowledge base",
    StepKind.TOOL: "Running a tool",
    StepKind.VERIFY: "Verifying the answer",
    StepKind.ANSWER: "Composing the answer",
}


def to_progress_event(step: ReasoningStep) -> dict | None:
    """Project an internal step onto the public progress contract."""
    if step.kind not in SAFE_KINDS:
        return None
    return {
        "step": step.kind.value,
        "label": PUBLIC_LABELS[step.kind],
        "status": "ok" if step.ok else "retrying",
        "duration_ms": step.latency_ms,
        # Deliberately absent: thought, observation, payload, error detail.
    }
