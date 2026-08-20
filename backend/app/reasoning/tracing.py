"""Observability (Chapter 10).

The trajectory IS the trace. An observability layer that records prompts and
completions but not the structured trajectory lets you read what happened and
not analyse it across runs.
"""
from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Iterator
from uuid import UUID

from loguru import logger

from app.config import settings
from app.reasoning.state import Trajectory

try:                                                # optional dependency
    from opentelemetry import trace as _otel
    _TRACER = _otel.get_tracer("reasoning")
except Exception:                                   # pragma: no cover
    _TRACER = None


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    started = time.perf_counter()
    if _TRACER is None:
        try:
            yield
        finally:
            logger.debug(f"{name} {attributes} "
                         f"{int((time.perf_counter() - started) * 1000)}ms")
        return
    with _TRACER.start_as_current_span(name) as s:   # pragma: no cover
        for k, v in attributes.items():
            s.set_attribute(f"reasoning.{k}", str(v))
        yield


INSERT_SQL = """
INSERT INTO reasoning_traces
    (run_id, user_id, thread_id, tier, goal, steps, final_answer,
     halted_reason, tokens, tool_calls, retrieval_hops, replans, wall_ms,
     model_version, index_version, created_at)
VALUES
    (:run_id, :user_id, :thread_id, :tier, :goal, :steps, :final_answer,
     :halted_reason, :tokens, :tool_calls, :retrieval_hops, :replans,
     :wall_ms, :model_version, :index_version, NOW())
"""


async def persist_trajectory(traj: Trajectory, *, user_id: UUID,
                             thread_id: UUID, tier: str,
                             wall_ms: int) -> None:
    """Persist ALWAYS, because a failed run is the one you most need to read.

    Trajectories live in their own access-controlled store with their own
    retention window, because that retention is a legal decision rather than
    an engineering one.
    """
    from sqlalchemy import text

    from app.db.main import async_session

    payload = {
        "run_id": traj.run_id,
        "user_id": str(user_id),
        "thread_id": str(thread_id),
        "tier": tier,
        "goal": traj.goal[:4000],
        "steps": json.dumps([s.model_dump(mode="json") for s in traj.steps]),
        "final_answer": traj.final_answer,
        "halted_reason": traj.halted_reason,
        "tokens": traj.tokens_used,
        "tool_calls": traj.tool_calls,
        "retrieval_hops": traj.retrieval_hops,
        "replans": traj.replans,
        "wall_ms": wall_ms,
        # Version-stamp every run: "what changed" must be a query, not an
        # investigation.
        "model_version": settings.reasoner_model,
        "index_version": settings.index_version,
    }
    try:
        async with async_session() as session:
            await session.execute(text(INSERT_SQL), payload)
            await session.commit()
    except Exception as exc:
        logger.warning(f"trace persistence failed (run {traj.run_id}): {exc}")

    _emit_metrics(traj, tier, wall_ms)


def _emit_metrics(traj: Trajectory, tier: str, wall_ms: int) -> None:
    """Token distribution is the LEADING indicator; latency is the lagging
    one. Alert on the p99 of completion tokens, not only on latency."""
    logger.info(
        "reasoning.run "
        + json.dumps({**traj.summary(), "tier": tier, "wall_ms": wall_ms})
    )
