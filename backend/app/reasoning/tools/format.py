"""Observation formatting (Chapter 5).

Answer-first, size-bounded, provenance-carrying. Structural summarisation
over truncation: truncating at a token boundary silently removes the rows
that mattered and produces confidently wrong answers.
"""
from __future__ import annotations

from statistics import mean
from typing import Any

from app.reasoning.tools.contract import ToolEmpty, ToolError, ToolOk, ToolResult

MAX_OBSERVATION_CHARS = 4_000


def _numeric_summary(rows: list[dict]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for col in rows[0]:
        values = [r[col] for r in rows
                  if isinstance(r.get(col), (int, float))]
        if len(values) >= 2:
            lines.append(f"  {col}: min={min(values):g} max={max(values):g} "
                         f"sum={sum(values):g} mean={mean(values):.4g}")
    return ("AGGREGATES:\n" + "\n".join(lines)) if lines else ""


def _table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    header = " | ".join(cols)
    body = "\n".join(" | ".join(str(r.get(c, "")) for c in cols) for r in rows)
    return f"{header}\n{'-' * len(header)}\n{body}"


def format_sql_result(rows: list[dict], sql: str, elapsed_ms: int,
                      max_rows: int = 5) -> str:
    """Answer-first, size-bounded, provenance-carrying observation."""
    if not rows:
        return (f"EMPTY: query returned 0 rows.\n"
                f"This is a valid result, not an error. The data does not "
                f"contain matching records.\nSQL: {sql}")

    head = rows[:max_rows]
    numeric = _numeric_summary(rows)
    omitted = len(rows) - len(head)
    return (
        f"ROWS: {len(rows)}\n"
        f"COLUMNS: {', '.join(rows[0])}\n"
        + (numeric + "\n" if numeric else "")
        + f"SAMPLE ({len(head)} of {len(rows)}):\n{_table(head)}\n"
        + (f"OMITTED: {omitted} further rows. Refine the query or aggregate "
           f"if you need them.\n" if omitted else "")
        + f"SOURCE: warehouse.read_replica @ {elapsed_ms}ms\nSQL: {sql}"
    )


def summarise(result: ToolResult) -> str:
    """Compact, stable rendering used for the trajectory observation field.

    Stable formatting matters: format churn across calls confuses the model
    and destroys prefix-cache hits.
    """
    if isinstance(result, ToolEmpty):
        return f"EMPTY ({result.source}): {result.hint or 'no matching records'}"
    if isinstance(result, ToolError):
        return f"ERROR[{result.kind}] ({result.source}): {result.detail[:200]}"
    if isinstance(result, ToolOk):
        body: Any = result.data
        if isinstance(body, list) and body and isinstance(body[0], dict):
            return format_sql_result(body, result.source, result.elapsed_ms)
        text = str(body)
        if len(text) > MAX_OBSERVATION_CHARS:
            return (text[:MAX_OBSERVATION_CHARS]
                    + f"\n[TRUNCATED: {len(text) - MAX_OBSERVATION_CHARS} "
                      f"further characters omitted; narrow the request]")
        return text
    return str(result)
