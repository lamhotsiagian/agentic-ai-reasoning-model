"""Layered guardrails (Chapter 10). Every layer fails closed.

Prompt injection is not solved and will not be solved by better prompting.
These heuristics raise the cost of the obvious attacks; the controls that
actually hold are tool scoping (Chapter 5/9) and database-enforced
authorisation. Treat this module as defence in depth, not as the boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"ignore (all |the |your )?(previous|prior|above) instructions",
        r"disregard (all |the )?(previous|prior|system)",
        r"you are now (a|an|in) ",
        r"(reveal|print|repeat|output) (your |the )?(system )?prompt",
        r"</?(system|assistant|tool)>",
        r"\bdeveloper mode\b",
        r"forget (everything|all) (you|above)",
    )
]

# Deliberately narrow: broad PII regexes produce false positives that block
# legitimate traffic, and a blocked user is a real cost.
PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "api_key": re.compile(r"\b(sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9]{16,}\b"),
}

MAX_PROMPT_CHARS = 12_000


@dataclass
class Violation:
    layer: str
    kind: str
    public_message: str
    detail: str = ""


async def scan_input(prompt: str, tenant: str) -> Violation | None:
    """Input layer. Reject with a clear message; fail closed."""
    if len(prompt) > MAX_PROMPT_CHARS:
        return Violation("input", "too_large",
                         f"Request exceeds {MAX_PROMPT_CHARS} characters.")
    for pattern in INJECTION_PATTERNS:
        if pattern.search(prompt):
            return Violation(
                "input", "prompt_injection",
                "That request contains instructions that attempt to override "
                "system behaviour and was not run.",
                detail=pattern.pattern,
            )
    for kind, pattern in PII_PATTERNS.items():
        if kind in ("credit_card", "ssn") and pattern.search(prompt):
            return Violation(
                "input", f"pii_{kind}",
                "That request appears to contain sensitive personal data. "
                "Remove it and try again.",
            )
    return None


def scan_retrieved(text: str) -> bool:
    """Untrusted retrieved content is the harder injection vector.

    Returns True when the passage looks like it is addressing the model.
    The caller marks it as data rather than dropping it, because dropping evidence
    is its own failure mode.
    """
    return any(p.search(text) for p in INJECTION_PATTERNS)


def wrap_untrusted(text: str, citation: str) -> str:
    """Fence untrusted content so it reads as data, not as instruction."""
    return (f"<document citation=\"{citation}\" trust=\"untrusted\">\n"
            f"{text}\n</document>")


def redact_output(answer: str, tenant: str) -> str:
    """Output layer. Redact rather than block where the answer is otherwise
    sound; block only on policy violations."""
    out = answer
    for kind, pattern in PII_PATTERNS.items():
        if kind in ("credit_card", "ssn", "api_key"):
            out = pattern.sub(f"[redacted:{kind}]", out)
    return out
