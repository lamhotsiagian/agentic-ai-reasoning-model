"""Lab runners backing the /labs/N routes in the Next.js UI.

Each lab isolates one mechanism from one chapter and returns the internals
the production API deliberately hides (raw trajectories, per-channel
rankings, search trees, verifier verdicts) so a reader can see the
behaviour rather than take it on faith.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from app.reasoning.compute import (BUDGETS, StabilityStopper,
                                   adaptive_self_consistency, canonical,
                                   choose_tier, parallel_self_consistency,
                                   static_difficulty)
from app.reasoning.decomposer import decompose
from app.reasoning.hybrid import compose, react_node, solve_hybrid
from app.reasoning.llm import reasoner, text_call
from app.reasoning.planner import (PlanStep, PlanValidationError,
                                   Reversibility, build_plan)
from app.reasoning.redaction import to_progress_event
from app.reasoning.retrieval.channels import (fulltext_search, graph_search,
                                              pgvector_search)
from app.reasoning.retrieval.loop import EvidenceSet, retrieve_for_reasoning
from app.reasoning.search import (SearchConfig, SearchNode, make_evaluator,
                                  make_expander, search)
from app.reasoning.state import Budget, Trajectory
from app.reasoning.tools.builtin import build_registry
from app.reasoning.verify import verify

Runner = Callable[[str, dict[str, Any], str], Awaitable[dict[str, Any]]]


def _dump(traj: Trajectory) -> dict[str, Any]:
    return {"trajectory": traj.model_dump(mode="json"),
            "metrics": traj.summary(),
            "progress": [e for s in traj.steps
                         if (e := to_progress_event(s)) is not None]}


# --- Lab 1: fast path vs deliberate path, under a budget --------------------
async def lab_01(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    budget = Budget(**{**BUDGETS["standard"].model_dump(),
                       **{k: v for k, v in cfg.items()
                          if k in Budget.model_fields}})
    fast_traj = Trajectory(goal=prompt, tier="fast")

    started = time.perf_counter()
    fast_answer, usage = await text_call(prompt)
    fast_ms = int((time.perf_counter() - started) * 1000)

    deliberate = Trajectory(goal=prompt, tier="standard")
    router = build_registry()
    started = time.perf_counter()
    deep_answer = await solve_hybrid(prompt, router, deliberate, budget)
    deep_ms = int((time.perf_counter() - started) * 1000)

    return {
        "fast": {"answer": fast_answer, "wall_ms": fast_ms,
                 "tokens": usage.prompt_tokens + usage.completion_tokens},
        "deliberate": {"answer": deep_answer, "wall_ms": deep_ms,
                       **_dump(deliberate)},
        "budget": budget.model_dump(),
    }


# --- Lab 2: the decomposition DAG ------------------------------------------
async def lab_02(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    router = build_registry()
    dag = await decompose(prompt, router.names(), traj)
    if cfg.get("drop_global_constraints"):
        dag.global_constraints = []      # reproduce the classic failure

    layers = [[s.model_dump() for s in layer]
              for layer in dag.execution_layers()]

    node_result = None
    if (node_id := cfg.get("run_node")):
        node = next((s for s in dag.sub_problems if s.id == node_id), None)
        if node is not None:
            out = await react_node(node, {}, dag.global_constraints, router,
                                   traj, max_steps=cfg.get("node_steps", 3))
            from app.reasoning.hybrid import passes
            node_result = {"node": node_id, "output": out,
                           "predicate": node.success_predicate,
                           "passed": passes(node, out)}

    return {"dag": dag.model_dump(), "layers": layers,
            "node_result": node_result, **_dump(traj)}


# --- Lab 3: planning, validation, and chaos --------------------------------
def _classify(sp) -> PlanStep:
    reversibility = {"retrieve": Reversibility.REVERSIBLE,
                     "compute": Reversibility.REVERSIBLE,
                     "infer": Reversibility.REVERSIBLE,
                     "tool": Reversibility.RECOVERABLE}[sp.kind]
    return PlanStep(
        sub_problem_id=sp.id, question=sp.question, tool=sp.tool_hint,
        reversibility=reversibility,
        compensation="revert_write" if reversibility else None,
        est_tokens=600 if sp.kind == "infer" else 350,
    )


async def lab_03(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    router = build_registry()
    dag = await decompose(prompt, router.names(), traj)

    registry = {n: object() for n in router.names()}
    if (broken := cfg.get("break_tool")):
        registry.pop(broken, None)

    try:
        plan = build_plan(dag, prompt, _classify, registry, traj)
        return {"plan": plan.model_dump(mode="json"),
                "signature": plan.signature, "validated": True, **_dump(traj)}
    except PlanValidationError as err:
        return {"plan": None, "validated": False, "error": str(err),
                "note": "the plan was rejected BEFORE execution; "
                        "no partial writes occurred",
                **_dump(traj)}


# --- Lab 4: the live search tree -------------------------------------------
async def lab_04(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    scfg = SearchConfig(**{k: v for k, v in cfg.items()
                           if k in SearchConfig.__dataclass_fields__})
    report = await search(SearchNode(), make_expander(prompt),
                          make_evaluator(prompt), scfg, traj)
    return {
        "config": scfg.__dict__,
        "best": {"steps": report.best.steps, "score": report.best.score,
                 "answer": report.best.answer},
        "stats": {"nodes_expanded": report.nodes_expanded,
                  "nodes_deduped": report.nodes_deduped,
                  "depth_reached": report.depth_reached,
                  "eval_calls": report.eval_calls},
        **_dump(traj),
    }


# --- Lab 5: the tool workbench ---------------------------------------------
async def lab_05(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    router = build_registry()
    scope_tags = set(cfg.get("tags") or {"retrieve", "compute", "answer"})
    exposed = (router.scope(tags=scope_tags) if not cfg.get("flat")
               else router.scope(tags={"retrieve", "compute", "structured",
                                       "unstructured", "graph", "answer",
                                       "verify"}))

    result = None
    if (call := cfg.get("call")):
        result = await router.call_with_retry(
            call["tool"], call.get("args", {}), traj,
            max_retries=cfg.get("max_retries", 2))
        result = result.model_dump()

    return {"exposed": [{"name": s.name, "tags": sorted(s.tags),
                         "description": s.description} for s in exposed],
            "breaker": router.breaker_state, "result": result, **_dump(traj)}


# --- Lab 6: per-hop retrieval ----------------------------------------------
async def lab_06(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    channels = set(cfg.get("channels") or ["vector", "sparse", "graph"])

    # Show each channel's own ranking before fusion. That is the point of the lab.
    per_channel: dict[str, list[dict]] = {}
    tasks = {}
    if "vector" in channels:
        tasks["vector"] = pgvector_search(prompt, tenant, 8)
    if "sparse" in channels:
        tasks["sparse"] = fulltext_search(prompt, tenant, 8)
    if "graph" in channels:
        tasks["graph"] = graph_search(prompt, tenant, 8)
    for name, coro in zip(tasks, await asyncio.gather(*tasks.values(),
                                                      return_exceptions=True)):
        per_channel[name] = ([] if isinstance(coro, BaseException)
                             else [{"doc": p.doc_id, "chunk": p.chunk_id,
                                    "score": round(p.score, 4),
                                    "text": p.text[:280]} for p in coro])

    evidence = await retrieve_for_reasoning(
        prompt, tenant, traj,
        max_hops=cfg.get("max_hops", 3),
        channels=channels,
        use_reranker=cfg.get("use_reranker", True),
    )
    return {"per_channel": per_channel,
            "evidence": [{"citation": p.citation(), "source": p.source,
                          "score": round(p.score, 4), "text": p.text[:400]}
                         for p in evidence.passages],
            "citations": evidence.citations(), **_dump(traj)}


# --- Lab 7: the verification cascade ---------------------------------------
async def lab_07(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    traj = Trajectory(goal=prompt)
    evidence = EvidenceSet()
    if cfg.get("with_evidence", True):
        evidence = await retrieve_for_reasoning(prompt, tenant, traj,
                                                max_hops=2)
    answer = cfg.get("answer") or (await text_call(prompt))[0]
    verdict = await verify(answer, traj, evidence,
                           require_citations=cfg.get("require_citations",
                                                     bool(evidence.passages)),
                           run_critique=cfg.get("run_critique", True))
    return {"answer": answer,
            "verdict": {"passed": verdict.passed,
                        "rung": verdict.rung.value if verdict.rung else None,
                        "detail": verdict.detail,
                        "score": verdict.score,
                        "unsupported": verdict.unsupported},
            **_dump(traj)}


# --- Lab 8: the scaling curve and tier routing ------------------------------
async def lab_08(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    async def probe(p: str) -> tuple[str, float]:
        t, _ = await text_call(p)
        return t, -0.9

    async def sample_k(p: str, k: int) -> list[str]:
        outs = await asyncio.gather(*[
            text_call(p, model=reasoner(temperature=0.8)) for _ in range(k)])
        return [t for t, _ in outs]

    decision = await choose_tier(prompt, cfg.get("task_class"),
                                 probe=probe, sample=sample_k)

    sweep: list[dict] = []
    if cfg.get("sweep"):
        for max_steps in (2, 4, 6, 8, 12):
            traj = Trajectory(goal=prompt)
            budget = Budget(max_steps=max_steps, max_tokens=48_000,
                            max_wall_ms=120_000, max_tool_calls=20)
            started = time.perf_counter()
            answer = await solve_hybrid(prompt, build_registry(), traj, budget)
            sweep.append({"max_steps": max_steps,
                          "tokens": traj.tokens_used,
                          "wall_ms": int((time.perf_counter() - started)*1000),
                          "halted": traj.halted_reason,
                          "answer": answer[:400]})

    consistency = None
    if cfg.get("self_consistency"):
        async def one(p: str) -> str:
            t, _ = await text_call(p, model=reasoner(temperature=0.8))
            return t
        if cfg.get("adaptive_n", True):
            answer, used = await adaptive_self_consistency(
                prompt, cfg.get("n", 5), one)
            consistency = {"mode": "adaptive", "samples_used": used,
                           "answer": answer}
        else:
            answer, votes = await parallel_self_consistency(
                prompt, cfg.get("n", 5), one)
            consistency = {"mode": "parallel", "votes": votes,
                           "answer": answer}

    return {"tier": decision.tier, "gate": decision.gate,
            "detail": decision.detail,
            "static_signal": static_difficulty(prompt, cfg.get("task_class")),
            "sweep": sweep, "consistency": consistency,
            "budgets": {t: b.model_dump() for t, b in BUDGETS.items()}}


# --- Lab 9: supervisor vs single agent -------------------------------------
async def lab_09(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    from app.reasoning.agents.supervisor import (Blackboard, TOOL_SCOPE,
                                                 build_supervisor,
                                                 run_single_agent)

    budget = BUDGETS[cfg.get("tier", "standard")]

    if cfg.get("single_agent"):
        traj = Trajectory(goal=prompt)
        answer = await run_single_agent(prompt, tenant, traj, budget)
        return {"mode": "single", "answer": answer,
                "scope": {r.value: sorted(t) for r, t in TOOL_SCOPE.items()},
                **_dump(traj)}

    traj = Trajectory(goal=prompt)
    bb = Blackboard(goal=prompt, tenant=tenant, trajectory=traj,
                    budget=budget,
                    critic_sees_trace=bool(cfg.get("critic_sees_trace")))
    diffs: list[dict] = []
    async for state in build_supervisor().astream(bb):
        for node, value in state.items():
            diffs.append({"node": node,
                          "evidence": len(getattr(value, "evidence",
                                                  EvidenceSet()).passages),
                          "objections": len(getattr(value, "objections", [])),
                          "verdict": getattr(value, "verdict", "pending")})
    return {"mode": "supervisor", "answer": traj.final_answer,
            "blackboard_diffs": diffs,
            "scope": {r.value: sorted(t) for r, t in TOOL_SCOPE.items()},
            **_dump(traj)}


# --- Lab 10: the production console ----------------------------------------
async def lab_10(prompt: str, cfg: dict, tenant: str) -> dict[str, Any]:
    from app.reasoning.eval.harness import run_suite

    if cfg.get("run_eval"):
        return {"report": await run_suite(cfg.get("suite", "default"),
                                          tenant=tenant)}

    traj = Trajectory(goal=prompt)
    budget = BUDGETS[cfg.get("tier", "standard")]
    answer = await solve_hybrid(prompt, build_registry(), traj, budget)
    traj.final_answer = answer
    dump = _dump(traj)
    return {"answer": answer,
            "public_stream": dump["progress"],     # what the browser sees
            "raw_trajectory": dump["trajectory"],  # what stays server-side
            "metrics": dump["metrics"]}


LAB_RUNNERS: dict[int, Runner] = {
    1: lab_01, 2: lab_02, 3: lab_03, 4: lab_04, 5: lab_05,
    6: lab_06, 7: lab_07, 8: lab_08, 9: lab_09, 10: lab_10,
}


# ---------------------------------------------------------------------------
# Streaming variants — each mirrors its non-streaming counterpart but yields
# LabEvent objects at key checkpoints so the frontend can render progress in
# real time.
# ---------------------------------------------------------------------------
from app.reasoning.lab_streaming import LabEvent, LabProgress  # noqa: E402
from app.reasoning.llm import reasoner                         # noqa: E402


async def lab_01_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    budget = Budget(**{**BUDGETS["standard"].model_dump(),
                       **{k: v for k, v in cfg.items()
                          if k in Budget.model_fields}})

    yield p.emit("start", "Starting Lab 1: Fast path vs. deliberate path")

    # ---- fast path --------------------------------------------------------
    yield p.emit("fast_path", "Running fast path — single forward pass…")
    started = time.perf_counter()
    fast_answer, usage = await text_call(prompt)
    fast_ms = int((time.perf_counter() - started) * 1000)
    yield p.emit("fast_path", "Fast path complete", status="completed", data={
        "answer": fast_answer, "wall_ms": fast_ms,
        "tokens": usage.prompt_tokens + usage.completion_tokens,
    })

    # ---- deliberate path --------------------------------------------------
    deliberate = Trajectory(goal=prompt, tier="standard")
    router = build_registry()

    yield p.emit("decompose", "Decomposing problem into sub-goals…")
    started = time.perf_counter()
    deep_answer = await solve_hybrid(prompt, router, deliberate, budget)
    deep_ms = int((time.perf_counter() - started) * 1000)

    for i, step in enumerate(deliberate.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Step {i + 1}: {step.kind.value}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "observation": step.observation,
                           "latency_ms": step.latency_ms,
                           "tokens": step.prompt_tokens + step.completion_tokens})

    yield p.emit("deliberate_done", "Deliberate path complete",
                 status="completed", data={
                     "answer": deep_answer, "wall_ms": deep_ms,
                     **_dump(deliberate),
                 })

    yield p.emit("done", "Lab 1 complete", status="completed", data={
        "fast": {"answer": fast_answer, "wall_ms": fast_ms,
                 "tokens": usage.prompt_tokens + usage.completion_tokens},
        "deliberate": {"answer": deep_answer, "wall_ms": deep_ms,
                       **_dump(deliberate)},
        "budget": budget.model_dump(),
    })


async def lab_02_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 2: Decomposition DAG")

    traj = Trajectory(goal=prompt)
    router = build_registry()

    yield p.emit("decompose", "Decomposing problem into sub-goals…")
    dag = await decompose(prompt, router.names(), traj)
    if cfg.get("drop_global_constraints"):
        dag.global_constraints = []

    layers = [[s.model_dump() for s in layer]
              for layer in dag.execution_layers()]
    yield p.emit("decompose",
                 f"Decomposed into {len(dag.sub_problems)} sub-problems "
                 f"across {len(layers)} layers",
                 status="completed",
                 data={"dag": dag.model_dump(), "layers": layers})

    node_result = None
    if (node_id := cfg.get("run_node")):
        node = next((s for s in dag.sub_problems if s.id == node_id), None)
        if node is not None:
            yield p.emit("node_run", f"Running node {node_id}…")
            out = await react_node(node, {}, dag.global_constraints, router,
                                   traj, max_steps=cfg.get("node_steps", 3))
            from app.reasoning.hybrid import passes
            node_result = {"node": node_id, "output": out,
                           "predicate": node.success_predicate,
                           "passed": passes(node, out)}
            yield p.emit("node_run",
                         f"Node {node_id} "
                         f"{'passed' if node_result['passed'] else 'failed'}",
                         status="completed", data=node_result)

    for i, step in enumerate(traj.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Step {i + 1}: {step.kind.value}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "latency_ms": step.latency_ms,
                           "tokens": step.prompt_tokens + step.completion_tokens})

    yield p.emit("done", "Lab 2 complete", status="completed", data={
        "dag": dag.model_dump(), "layers": layers,
        "node_result": node_result, **_dump(traj),
    })


async def lab_03_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 3: Planning and validation")

    traj = Trajectory(goal=prompt)
    router = build_registry()

    yield p.emit("decompose", "Decomposing problem…")
    dag = await decompose(prompt, router.names(), traj)
    yield p.emit("decompose",
                 f"Decomposed into {len(dag.sub_problems)} sub-problems",
                 status="completed")

    registry = {n: object() for n in router.names()}
    if (broken := cfg.get("break_tool")):
        registry.pop(broken, None)
        yield p.emit("chaos", f"Removed tool '{broken}' from registry",
                     status="completed", data={"broken_tool": broken})

    yield p.emit("validate", "Building and validating plan…")
    try:
        plan = build_plan(dag, prompt, _classify, registry, traj)
        yield p.emit("validate", "Plan validated — all checks passed",
                     status="completed",
                     data={"plan": plan.model_dump(mode="json"),
                           "signature": plan.signature})
        yield p.emit("done", "Lab 3 complete", status="completed", data={
            "plan": plan.model_dump(mode="json"),
            "signature": plan.signature, "validated": True, **_dump(traj),
        })
    except PlanValidationError as err:
        yield p.emit("validate", f"Plan rejected: {err}", status="failed",
                     data={"error": str(err)})
        yield p.emit("done", "Lab 3 complete", status="completed", data={
            "plan": None, "validated": False, "error": str(err),
            "note": "the plan was rejected BEFORE execution; "
                    "no partial writes occurred",
            **_dump(traj),
        })


async def lab_04_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 4: Search-based reasoning")

    traj = Trajectory(goal=prompt)
    scfg = SearchConfig(**{k: v for k, v in cfg.items()
                           if k in SearchConfig.__dataclass_fields__})
    yield p.emit("config",
                 f"Strategy: {scfg.strategy}, k={scfg.k}, "
                 f"beam_width={scfg.beam_width}",
                 status="completed", data={"config": scfg.__dict__})

    yield p.emit("search",
                 f"Running {scfg.strategy} search (depth {scfg.max_depth})…")
    report = await search(SearchNode(), make_expander(prompt),
                          make_evaluator(prompt), scfg, traj)

    for i, step in enumerate(traj.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Search step {i + 1}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "latency_ms": step.latency_ms})

    yield p.emit("search", "Search complete", status="completed", data={
        "stats": {"nodes_expanded": report.nodes_expanded,
                  "nodes_deduped": report.nodes_deduped,
                  "depth_reached": report.depth_reached,
                  "eval_calls": report.eval_calls},
        "best": {"steps": report.best.steps, "score": report.best.score,
                 "answer": report.best.answer},
    })

    yield p.emit("done", "Lab 4 complete", status="completed", data={
        "config": scfg.__dict__,
        "best": {"steps": report.best.steps, "score": report.best.score,
                 "answer": report.best.answer},
        "stats": {"nodes_expanded": report.nodes_expanded,
                  "nodes_deduped": report.nodes_deduped,
                  "depth_reached": report.depth_reached,
                  "eval_calls": report.eval_calls},
        **_dump(traj),
    })


async def lab_05_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 5: Tool workbench")

    traj = Trajectory(goal=prompt)
    router = build_registry()
    scope_tags = set(cfg.get("tags") or {"retrieve", "compute", "answer"})
    exposed = list(router.scope(tags=scope_tags) if not cfg.get("flat")
                   else router.scope(tags={"retrieve", "compute", "structured",
                                           "unstructured", "graph", "answer",
                                           "verify"}))

    yield p.emit("scope", f"Scoped registry to {len(exposed)} tools",
                 status="completed", data={
                     "exposed": [{"name": s.name, "tags": sorted(s.tags),
                                  "description": s.description}
                                 for s in exposed],
                 })

    result = None
    if (call := cfg.get("call")):
        yield p.emit("tool_call", f"Calling tool '{call['tool']}'…",
                     data={"tool": call["tool"],
                           "args": call.get("args", {})})
        result = await router.call_with_retry(
            call["tool"], call.get("args", {}), traj,
            max_retries=cfg.get("max_retries", 2))
        result = result.model_dump()
        yield p.emit("tool_call", f"Tool '{call['tool']}' returned",
                     status="completed", data={"result": result})

    yield p.emit("done", "Lab 5 complete", status="completed", data={
        "exposed": [{"name": s.name, "tags": sorted(s.tags),
                     "description": s.description} for s in exposed],
        "breaker": router.breaker_state, "result": result, **_dump(traj),
    })


async def lab_06_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 6: Multi-hop hybrid retrieval")

    traj = Trajectory(goal=prompt)
    channels = set(cfg.get("channels") or ["vector", "sparse", "graph"])

    per_channel: dict[str, list[dict]] = {}
    tasks = {}
    if "vector" in channels:
        tasks["vector"] = pgvector_search(prompt, tenant, 8)
    if "sparse" in channels:
        tasks["sparse"] = fulltext_search(prompt, tenant, 8)
    if "graph" in channels:
        tasks["graph"] = graph_search(prompt, tenant, 8)

    for ch_name in tasks:
        yield p.emit(f"channel_{ch_name}", f"Searching {ch_name} channel…")

    raw = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, coro in zip(tasks, raw):
        rows = ([] if isinstance(coro, BaseException)
                else [{"doc": pa.doc_id, "chunk": pa.chunk_id,
                       "score": round(pa.score, 4),
                       "text": pa.text[:280]} for pa in coro])
        per_channel[name] = rows
        yield p.emit(
            f"channel_{name}",
            f"{name}: {len(rows)} results"
            if not isinstance(coro, BaseException)
            else f"{name}: failed ({type(coro).__name__})",
            status="completed" if not isinstance(coro, BaseException)
            else "failed",
            data={"results": rows})

    yield p.emit("fuse", "Fusing and reranking across channels…")
    evidence = await retrieve_for_reasoning(
        prompt, tenant, traj,
        max_hops=cfg.get("max_hops", 3),
        channels=channels,
        use_reranker=cfg.get("use_reranker", True),
    )
    evidence_list = [{"citation": pa.citation(), "source": pa.source,
                      "score": round(pa.score, 4), "text": pa.text[:400]}
                     for pa in evidence.passages]
    yield p.emit("fuse", f"Fused to {len(evidence_list)} passages",
                 status="completed",
                 data={"evidence": evidence_list,
                       "citations": evidence.citations()})

    for i, step in enumerate(traj.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Retrieval step {i + 1}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "latency_ms": step.latency_ms})

    yield p.emit("done", "Lab 6 complete", status="completed", data={
        "per_channel": per_channel, "evidence": evidence_list,
        "citations": evidence.citations(), **_dump(traj),
    })


async def lab_07_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 7: Verification cascade")

    traj = Trajectory(goal=prompt)
    evidence = EvidenceSet()

    if cfg.get("with_evidence", True):
        yield p.emit("retrieve", "Retrieving evidence…")
        evidence = await retrieve_for_reasoning(prompt, tenant, traj,
                                                max_hops=2)
        yield p.emit("retrieve",
                     f"Retrieved {len(evidence.passages)} passages",
                     status="completed")

    yield p.emit("generate", "Generating answer to verify…")
    answer = cfg.get("answer") or (await text_call(prompt))[0]
    yield p.emit("generate", "Answer generated", status="completed",
                 data={"answer": answer[:300]})

    yield p.emit("verify", "Running verification cascade…")
    verdict = await verify(answer, traj, evidence,
                           require_citations=cfg.get(
                               "require_citations",
                               bool(evidence.passages)),
                           run_critique=cfg.get("run_critique", True))

    RUNGS = ["structural", "programmatic", "entailment", "critique"]
    for rung in RUNGS:
        is_failing = (verdict.rung and verdict.rung.value == rung
                      and not verdict.passed)
        yield p.emit(
            f"rung_{rung}",
            f"Rung: {rung}"
            + (f" — FAILED: {verdict.detail}" if is_failing else " — passed"),
            status="failed" if is_failing else "completed")
        if is_failing:
            break

    yield p.emit("verify", "Verification complete", status="completed", data={
        "verdict": {"passed": verdict.passed,
                    "rung": verdict.rung.value if verdict.rung else None,
                    "detail": verdict.detail,
                    "score": verdict.score,
                    "unsupported": verdict.unsupported},
    })

    yield p.emit("done", "Lab 7 complete", status="completed", data={
        "answer": answer,
        "verdict": {"passed": verdict.passed,
                    "rung": verdict.rung.value if verdict.rung else None,
                    "detail": verdict.detail,
                    "score": verdict.score,
                    "unsupported": verdict.unsupported},
        **_dump(traj),
    })


async def lab_08_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 8: Test-time compute scaling")

    from app.reasoning.compute import static_difficulty

    async def probe(pr: str) -> tuple[str, float]:
        t, _ = await text_call(pr)
        return t, -0.9

    async def sample_k(pr: str, k: int) -> list[str]:
        outs = await asyncio.gather(*[
            text_call(pr, model=reasoner(temperature=0.8)) for _ in range(k)])
        return [t for t, _ in outs]

    yield p.emit("tier_decision", "Determining compute tier…")
    decision = await choose_tier(prompt, cfg.get("task_class"),
                                 probe=probe, sample=sample_k)
    yield p.emit("tier_decision",
                 f"Tier: {decision.tier} (gate: {decision.gate})",
                 status="completed", data={
                     "tier": decision.tier, "gate": decision.gate,
                     "detail": decision.detail,
                     "static_signal": static_difficulty(
                         prompt, cfg.get("task_class")),
                 })

    sweep: list[dict] = []
    if cfg.get("sweep"):
        for max_steps in (2, 4, 6, 8, 12):
            yield p.emit("sweep", f"Sweep: budget={max_steps} steps…",
                         data={"max_steps": max_steps})
            traj = Trajectory(goal=prompt)
            b = Budget(max_steps=max_steps, max_tokens=48_000,
                       max_wall_ms=120_000, max_tool_calls=20)
            started = time.perf_counter()
            ans = await solve_hybrid(prompt, build_registry(), traj, b)
            row = {"max_steps": max_steps, "tokens": traj.tokens_used,
                   "wall_ms": int((time.perf_counter() - started) * 1000),
                   "halted": traj.halted_reason, "answer": ans[:400]}
            sweep.append(row)
            yield p.emit("sweep",
                         f"Sweep: {max_steps} steps → "
                         f"{row['tokens']}t, {row['wall_ms']}ms",
                         status="completed", data=row)

    consistency = None
    if cfg.get("self_consistency"):
        async def one(pr: str) -> str:
            t, _ = await text_call(pr, model=reasoner(temperature=0.8))
            return t

        yield p.emit("consistency", "Running self-consistency…")
        if cfg.get("adaptive_n", True):
            ans, used = await adaptive_self_consistency(
                prompt, cfg.get("n", 5), one)
            consistency = {"mode": "adaptive", "samples_used": used,
                           "answer": ans}
        else:
            ans, votes = await parallel_self_consistency(
                prompt, cfg.get("n", 5), one)
            consistency = {"mode": "parallel", "votes": votes, "answer": ans}
        yield p.emit("consistency", "Self-consistency complete",
                     status="completed", data=consistency)

    yield p.emit("done", "Lab 8 complete", status="completed", data={
        "tier": decision.tier, "gate": decision.gate,
        "detail": decision.detail,
        "static_signal": static_difficulty(prompt, cfg.get("task_class")),
        "sweep": sweep, "consistency": consistency,
        "budgets": {t: b.model_dump() for t, b in BUDGETS.items()},
    })


async def lab_09_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 9: Multi-agent reasoning")

    from app.reasoning.agents.supervisor import (Blackboard, TOOL_SCOPE,
                                                 build_supervisor,
                                                 run_single_agent)

    budget = BUDGETS[cfg.get("tier", "standard")]

    if cfg.get("single_agent"):
        yield p.emit("single_agent", "Running single agent baseline…")
        traj = Trajectory(goal=prompt)
        answer = await run_single_agent(prompt, tenant, traj, budget)

        for i, step in enumerate(traj.steps):
            yield p.emit(step.kind.value,
                         step.thought or f"Step {i + 1}",
                         status="completed" if step.ok else "retrying",
                         data={"payload": step.payload,
                               "latency_ms": step.latency_ms})

        yield p.emit("done", "Lab 9 complete", status="completed", data={
            "mode": "single", "answer": answer,
            "scope": {r.value: sorted(t) for r, t in TOOL_SCOPE.items()},
            **_dump(traj),
        })
        return

    traj = Trajectory(goal=prompt)
    bb = Blackboard(goal=prompt, tenant=tenant, trajectory=traj,
                    budget=budget,
                    critic_sees_trace=bool(cfg.get("critic_sees_trace")))
    diffs: list[dict] = []

    yield p.emit("supervisor", "Running supervisor with specialists…")
    async for state in build_supervisor().astream(bb):
        for node, value in state.items():
            diff = {"node": node,
                    "evidence": len(getattr(value, "evidence",
                                            EvidenceSet()).passages),
                    "objections": len(getattr(value, "objections", [])),
                    "verdict": getattr(value, "verdict", "pending")}
            diffs.append(diff)
            yield p.emit("specialist", f"Specialist '{node}' completed",
                         status="completed", data=diff)

    for i, step in enumerate(traj.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Step {i + 1}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "latency_ms": step.latency_ms})

    yield p.emit("done", "Lab 9 complete", status="completed", data={
        "mode": "supervisor", "answer": traj.final_answer,
        "blackboard_diffs": diffs,
        "scope": {r.value: sorted(t) for r, t in TOOL_SCOPE.items()},
        **_dump(traj),
    })


async def lab_10_stream(prompt: str, cfg: dict, tenant: str
                        ) -> AsyncIterator[LabEvent]:
    p = LabProgress()
    yield p.emit("start", "Starting Lab 10: Production console")

    from app.reasoning.eval.harness import run_suite

    if cfg.get("run_eval"):
        yield p.emit("eval", "Running evaluation suite…")
        report = await run_suite(cfg.get("suite", "default"), tenant=tenant)
        yield p.emit("eval", "Evaluation complete", status="completed",
                     data={"report": report})
        yield p.emit("done", "Lab 10 complete", status="completed",
                     data={"report": report})
        return

    traj = Trajectory(goal=prompt)
    budget_cfg = BUDGETS[cfg.get("tier", "standard")]

    yield p.emit("solve", "Running hybrid reasoning…")
    answer = await solve_hybrid(prompt, build_registry(), traj, budget_cfg)
    traj.final_answer = answer

    for i, step in enumerate(traj.steps):
        yield p.emit(step.kind.value,
                     step.thought or f"Step {i + 1}: {step.kind.value}",
                     status="completed" if step.ok else "retrying",
                     data={"payload": step.payload,
                           "observation": step.observation,
                           "latency_ms": step.latency_ms,
                           "tokens": step.prompt_tokens
                                     + step.completion_tokens})

    dump = _dump(traj)
    yield p.emit("done", "Lab 10 complete", status="completed", data={
        "answer": answer,
        "public_stream": dump["progress"],
        "raw_trajectory": dump["trajectory"],
        "metrics": dump["metrics"],
    })


StreamRunner = Callable[[str, dict[str, Any], str],
                         AsyncIterator[LabEvent]]

LAB_STREAM_RUNNERS: dict[int, StreamRunner] = {
    1: lab_01_stream, 2: lab_02_stream, 3: lab_03_stream,
    4: lab_04_stream, 5: lab_05_stream, 6: lab_06_stream,
    7: lab_07_stream, 8: lab_08_stream, 9: lab_09_stream,
    10: lab_10_stream,
}
