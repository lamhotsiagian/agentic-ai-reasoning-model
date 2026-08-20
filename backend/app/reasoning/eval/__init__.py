"""Decomposed evaluation (Chapter 10).

Aggregate accuracy cannot localise a regression. Every metric here except
task success is computable directly from a stored trajectory, which is why
the harness is built on the trajectory object rather than on request/response
pairs.
"""
from app.reasoning.eval.harness import run_suite
from app.reasoning.eval.metrics import EvalReport, MetricResult

__all__ = ["run_suite", "EvalReport", "MetricResult"]
