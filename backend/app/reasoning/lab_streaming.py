"""Streaming infrastructure for interactive lab execution.

Lab runners yield LabEvent objects as they execute, so the frontend
can render each reasoning step in real time rather than waiting for
the entire run to complete.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, AsyncIterator


@dataclass
class LabEvent:
    """One streaming event emitted by a lab runner."""
    phase: str              # e.g. "decompose", "retrieve", "plan", "tool", "verify", "compose", "done"
    label: str              # Human-readable description
    status: str = "started" # "started", "completed", "failed", "retrying"
    data: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0
    step_index: int = 0

    def to_ndjson(self) -> str:
        return json.dumps(asdict(self), default=str) + "\n"


class LabProgress:
    """Context manager that lab runners use to emit streaming events.
    
    Usage inside a lab streaming function:
        progress = LabProgress()
        yield progress.emit("decompose", "Decomposing problem...", status="started")
        dag = await decompose(prompt, ...)
        yield progress.emit("decompose", "Decomposition complete", status="completed", 
                           data={"nodes": [...]})
    """
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._step = 0

    def emit(self, phase: str, label: str, *, 
             status: str = "started",
             data: dict[str, Any] | None = None) -> LabEvent:
        elapsed = int((time.perf_counter() - self._start) * 1000)
        event = LabEvent(
            phase=phase,
            label=label,
            status=status,
            data=data or {},
            elapsed_ms=elapsed,
            step_index=self._step,
        )
        self._step += 1
        return event
