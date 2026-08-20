"""Production entry point for the reasoning service (Chapter 10)."""
from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.auth.dependencies import CurrentUserDep
from app.reasoning.compute import BUDGETS, budget_for, choose_tier
from app.reasoning.guardrails import redact_output, scan_input
from app.reasoning.llm import reasoner, text_call
from app.reasoning.redaction import to_progress_event
from app.reasoning.schemas import LabRunRequest, ReasoningRequest
from app.reasoning.state import Trajectory
from app.reasoning.tracing import persist_trajectory, span

reasoning_router = APIRouter()
_GRAPH = None


def graph():
    """Compile the supervisor once, lazily, because import cost is not trivial."""
    global _GRAPH
    if _GRAPH is None:
        from app.reasoning.agents.supervisor import build_supervisor

        _GRAPH = build_supervisor()
    return _GRAPH


async def _probe(prompt: str) -> tuple[str, float]:
    text, _ = await text_call(prompt, model=reasoner(temperature=0.0))
    return text, -0.9          # Ollama does not surface logprobs uniformly


async def _sample(prompt: str, k: int) -> list[str]:
    import asyncio

    outs = await asyncio.gather(*[
        text_call(prompt, model=reasoner(temperature=0.8)) for _ in range(k)
    ])
    return [t for t, _ in outs]


@reasoning_router.post("/{thread_id}")
async def run_reasoning(thread_id: UUID, req: ReasoningRequest,
                        user: CurrentUserDep) -> StreamingResponse:
    """Stream structured progress events.

    The raw trajectory never leaves the server: it routinely contains
    retrieved fragments the user is not entitled to see, internal tool
    schemas, and verbatim policy text.
    """
    # ---- input guardrails: fail closed --------------------------------
    if (violation := await scan_input(req.prompt, tenant=str(user.id))):
        logger.info(f"guardrail blocked: {violation.kind}")
        raise HTTPException(422, detail=violation.public_message)

    # ---- adaptive budget ----------------------------------------------
    if req.tier_override:
        tier, gate = req.tier_override, "override"
    else:
        decision = await choose_tier(req.prompt, req.task_class,
                                     probe=_probe, sample=_sample)
        tier, gate = decision.tier, decision.gate

    traj = Trajectory(goal=req.prompt)
    budget = budget_for(tier, traj)

    async def event_stream():
        started = time.perf_counter()
        emitted = 0
        try:
            with span("reasoning.run", tier=tier, gate=gate,
                      user=str(user.id), thread=str(thread_id)):
                async for _ in graph().astream(
                    {"goal": req.prompt, "tenant": str(user.id),
                     "trajectory": traj, "budget": budget},
                    config={"configurable": {"thread_id": str(thread_id)}},
                ):
                    # Whitelist projection: an unrecognised step kind is
                    # dropped, so a new internal step cannot leak by default.
                    for step in traj.steps[emitted:]:
                        if (evt := to_progress_event(step)) is not None:
                            yield json.dumps(evt) + "\n"
                    emitted = len(traj.steps)

                answer = redact_output(traj.final_answer or "",
                                       tenant=str(user.id))

                # A budget trip returns a PARTIAL answer, never a timeout.
                yield json.dumps({
                    "step": "answer",
                    "content": answer,
                    "citations": traj.citations,
                    "partial": traj.halted_reason is not None,
                    "halted_reason": traj.halted_reason,
                    "metrics": {**traj.summary(), "tier": tier, "gate": gate},
                }) + "\n"

        except Exception as exc:                     # never leak internals
            logger.exception("reasoning run failed")
            traj.halted_reason = f"internal_error:{type(exc).__name__}"
            yield json.dumps({
                "step": "error",
                "content": "The reasoning run failed. Please retry.",
            }) + "\n"
        finally:
            # Persist ALWAYS: a failed run is the one you most need to read.
            await persist_trajectory(
                traj, user_id=user.id, thread_id=thread_id, tier=tier,
                wall_ms=int((time.perf_counter() - started) * 1000),
            )

    return StreamingResponse(event_stream(),
                             media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# Lab endpoints. These expose internals the production route deliberately
# hides, and are gated behind a setting so they cannot ship enabled.
# ---------------------------------------------------------------------------
@reasoning_router.post("/labs/run")
async def run_lab(req: LabRunRequest, user: CurrentUserDep) -> dict[str, Any]:
    from app.config import settings

    if not settings.labs_enabled:
        raise HTTPException(404, detail="labs are disabled")

    from app.reasoning.labs import LAB_RUNNERS

    runner = LAB_RUNNERS.get(req.lab)
    if runner is None:
        raise HTTPException(404, detail=f"no lab {req.lab}")
    return await runner(req.prompt, req.config, tenant=str(user.id))


@reasoning_router.post("/labs/stream")
async def stream_lab(req: LabRunRequest, user: CurrentUserDep) -> StreamingResponse:
    """Stream lab execution as NDJSON — one line per reasoning step."""
    from app.config import settings

    if not settings.labs_enabled:
        raise HTTPException(404, detail="labs are disabled")

    from app.reasoning.labs import LAB_STREAM_RUNNERS

    runner = LAB_STREAM_RUNNERS.get(req.lab)
    if runner is None:
        raise HTTPException(404, detail=f"no streaming lab {req.lab}")

    async def event_stream():
        try:
            async for event in runner(req.prompt, req.config, tenant=str(user.id)):
                yield event.to_ndjson()
        except Exception as exc:
            logger.exception(f"streaming lab {req.lab} failed")
            from app.reasoning.lab_streaming import LabEvent
            yield LabEvent(
                phase="error",
                label=f"Lab failed: {type(exc).__name__}",
                status="failed",
                data={"detail": str(exc)},
            ).to_ndjson()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@reasoning_router.get("/budgets")
async def list_budgets() -> dict[str, Any]:
    return {tier: b.model_dump() for tier, b in BUDGETS.items()}


@reasoning_router.get("/tools")
async def list_tools() -> list[dict[str, Any]]:
    from app.reasoning.tools.builtin import build_registry

    router = build_registry()
    return [
        {"name": s.name, "description": s.description,
         "tags": sorted(s.tags), "reversibility": s.reversibility,
         "args": s.args_model.model_json_schema()}
        for s in router.scope(tags={"compute", "retrieve", "structured",
                                    "unstructured", "graph", "answer",
                                    "verify"})
    ]
