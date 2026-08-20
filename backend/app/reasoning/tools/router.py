"""Scoped tool registry with validated execution and typed results (Ch.5)."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger
from pydantic import BaseModel, ValidationError

from app.reasoning.state import ReasoningStep, StepKind, Trajectory
from app.reasoning.tools.contract import ToolError, ToolResult
from app.reasoning.tools.format import summarise


@dataclass
class ToolSpec:
    name: str
    description: str            # must state what the tool does NOT cover
    args_model: type[BaseModel]
    fn: Callable[..., Awaitable[ToolResult]]
    reversibility: int = 0      # see Chapter 3
    timeout_s: float = 10.0
    tags: set[str] = field(default_factory=set)   # used for sub-goal scoping
    model_retryable: bool = True                  # False for irreversible acts


class CircuitBreaker:
    """Per-tool breaker. A tool that fails twice is removed for the run."""

    def __init__(self, threshold: int = 2, cooldown_s: float = 30.0) -> None:
        self.threshold, self.cooldown_s = threshold, cooldown_s
        self._fails: dict[str, int] = {}
        self._opened: dict[str, float] = {}

    def is_open(self, tool: str) -> bool:
        opened = self._opened.get(tool)
        if opened is None:
            return False
        if time.time() - opened > self.cooldown_s:
            self._opened.pop(tool, None)
            self._fails.pop(tool, None)
            return False
        return True

    def record(self, tool: str, ok: bool) -> None:
        if ok:
            self._fails.pop(tool, None)
            return
        self._fails[tool] = self._fails.get(tool, 0) + 1
        if self._fails[tool] >= self.threshold:
            self._opened[tool] = time.time()
            logger.warning(f"circuit opened for tool '{tool}'")

    def state(self) -> dict[str, str]:
        return {name: "open" if self.is_open(name) else "closed"
                for name in set(self._fails) | set(self._opened)}


def _compact(exc: ValidationError) -> str:
    """The validation error IS the repair signal, so keep it readable."""
    parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"
             for e in exc.errors()[:5]]
    return "; ".join(parts)


class ToolRouter:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {s.name: s for s in specs}
        self._breaker = CircuitBreaker()
        self._run_cache: dict[tuple[str, str], ToolResult] = {}

    # -- registry ----------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def scope(self, tags: set[str], core: set[str] | None = None
              ) -> list[ToolSpec]:
        """Expose only tools relevant to the current sub-goal.

        Selection accuracy degrades past ~15-20 tools in one context, so
        scoping is an accuracy mechanism, not just a token saving. A small
        always-on core set prevents 'the tool was never shown' failures.
        """
        core = core or {"final_answer"}
        return [s for s in self._specs.values()
                if (s.tags & tags or s.name in core)
                and not self._breaker.is_open(s.name)]

    def describe(self, specs: list[ToolSpec]) -> str:
        return "\n".join(f"- {s.name}: {s.description}" for s in specs)

    # -- execution ---------------------------------------------------------
    async def call(self, name: str, raw_args: dict[str, Any],
                   traj: Trajectory) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            # Hallucinated tool name: a semantic error the model can repair.
            return ToolError(kind="bad_args", retryable=False,
                             detail=f"unknown tool '{name}'; available: "
                                    f"{self.names()}")
        if self._breaker.is_open(name):
            return ToolError(kind="upstream", retryable=False, source=name,
                             detail=f"'{name}' is circuit-broken for this run")

        # ---- strict validation; the error text IS the repair signal --------
        try:
            args = spec.args_model(**raw_args)
        except ValidationError as exc:
            return ToolError(kind="bad_args", retryable=False, source=name,
                             detail=_compact(exc))

        # ---- within-run cache; a repeat is itself a signal to the model ----
        cache_key = (name, repr(sorted(args.model_dump().items())))
        if (cached := self._run_cache.get(cache_key)) is not None:
            traj.append(ReasoningStep(
                kind=StepKind.TOOL, ok=True,
                payload={"tool": name, "args": raw_args, "cached": True},
                observation="(cached) " + summarise(cached),
            ))
            return cached

        # ---- execute with a hard timeout -----------------------------------
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(spec.fn(**args.model_dump()),
                                            timeout=spec.timeout_s)
            ok = not isinstance(result, ToolError)
        except asyncio.TimeoutError:
            result = ToolError(kind="timeout", retryable=True, source=name,
                               detail=f"exceeded {spec.timeout_s}s")
            ok = False
        except Exception as exc:                      # never leak a stack trace
            logger.exception(f"tool '{name}' raised")
            result = ToolError(kind="upstream", retryable=True, source=name,
                               detail=type(exc).__name__)
            ok = False

        elapsed = int((time.perf_counter() - started) * 1000)
        self._breaker.record(name, ok)
        if ok:
            self._run_cache[cache_key] = result

        traj.append(ReasoningStep(
            kind=StepKind.TOOL, ok=ok,
            payload={"tool": name, "args": raw_args,
                     "reversibility": spec.reversibility},
            observation=summarise(result),
            error=None if ok else getattr(result, "detail", None),
            latency_ms=elapsed,
        ))

        # ---- loop guard: three identical calls means the model is stuck -----
        if traj.loop_signature(window=3):
            return ToolError(
                kind="bad_args", retryable=False, source=name,
                detail="repeated identical call detected; change your "
                       "approach or answer from what you already have",
            )
        return result

    async def call_with_retry(self, name: str, raw_args: dict[str, Any],
                              traj: Trajectory, max_retries: int = 2
                              ) -> ToolResult:
        """Orchestrator-owned recovery.

        Transient infrastructure failures never reach the model, because it would
        reason about them and waste tokens inventing workarounds.
        """
        delay = 0.25
        for attempt in range(max_retries + 1):
            result = await self.call(name, raw_args, traj)
            if not isinstance(result, ToolError) or not result.retryable:
                return result
            if attempt == max_retries:
                return result
            await asyncio.sleep(delay)
            delay *= 2
        return result

    @property
    def breaker_state(self) -> dict[str, str]:
        return self._breaker.state()
