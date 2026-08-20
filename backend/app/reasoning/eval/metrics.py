"""Evaluation metrics and their confidence intervals."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.reasoning.state import Trajectory


@dataclass
class MetricResult:
    name: str
    value: float
    n: int
    ci_low: float = 0.0
    ci_high: float = 0.0
    unit: str = "rate"

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": round(self.value, 4), "n": self.n,
                "ci": [round(self.ci_low, 4), round(self.ci_high, 4)],
                "unit": self.unit}


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at the small n an eval set actually has,
    unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def rate(name: str, successes: int, n: int) -> MetricResult:
    lo, hi = wilson_ci(successes, n)
    return MetricResult(name, successes / n if n else 0.0, n, lo, hi)


@dataclass
class EvalReport:
    suite: str
    metrics: list[MetricResult] = field(default_factory=list)
    per_slice: dict[str, list[MetricResult]] = field(default_factory=dict)
    cost_per_task: float = 0.0
    cost_per_success: float = 0.0
    p50_ms: int = 0
    p95_ms: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "metrics": [m.as_dict() for m in self.metrics],
            "per_slice": {k: [m.as_dict() for m in v]
                          for k, v in self.per_slice.items()},
            "cost_per_task": round(self.cost_per_task, 5),
            # The only cost metric safe to optimise: a cheaper configuration
            # that fails more often is not cheaper.
            "cost_per_success": round(self.cost_per_success, 5),
            "p50_ms": self.p50_ms, "p95_ms": self.p95_ms,
            "diagnostics": self.diagnostics,
        }


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def efficiency(trajectories: list[Trajectory]) -> dict[str, float]:
    if not trajectories:
        return {}
    return {
        "mean_tokens": sum(t.tokens_used for t in trajectories) / len(trajectories),
        "mean_steps": sum(len(t.steps) for t in trajectories) / len(trajectories),
        "mean_tool_calls": sum(t.tool_calls for t in trajectories) / len(trajectories),
        "mean_replans": sum(t.replans for t in trajectories) / len(trajectories),
        "halt_rate": sum(1 for t in trajectories if t.halted_reason)
                     / len(trajectories),
    }
