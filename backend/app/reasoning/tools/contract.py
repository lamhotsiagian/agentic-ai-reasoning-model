"""Tool results are a closed union. Errors are structural, never prose."""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel


class ToolOk(BaseModel):
    status: Literal["ok"] = "ok"
    data: dict | list
    row_count: int | None = None
    truncated: bool = False
    omitted: int = 0            # rows/bytes withheld; the model must know
    source: str = ""            # provenance: which system answered
    elapsed_ms: int = 0


class ToolEmpty(BaseModel):
    """Valid execution, zero results.

    Distinct from failure ON PURPOSE: 'no matching orders' is an ANSWER;
    'the orders service returned 503' is not. A model that receives both as
    strings will report that the customer has no orders during an outage.
    """

    status: Literal["empty"] = "empty"
    hint: str = ""              # e.g. "no rows for that date range"
    source: str = ""


ErrorKind = Literal["timeout", "auth", "rate_limit", "bad_args",
                    "upstream", "invalid_query", "forbidden"]

# Who owns the recovery. Transient infrastructure errors are the
# orchestrator's problem; semantic errors are the model's.
ORCHESTRATOR_OWNED: set[str] = {"timeout", "rate_limit", "auth", "upstream"}
MODEL_OWNED: set[str] = {"bad_args", "invalid_query", "forbidden"}


class ToolError(BaseModel):
    status: Literal["error"] = "error"
    kind: ErrorKind
    retryable: bool = False     # the orchestrator reads this, not the model
    detail: str = ""            # redacted before it reaches the model
    source: str = ""

    @property
    def model_visible(self) -> bool:
        return self.kind in MODEL_OWNED


ToolResult = Union[ToolOk, ToolEmpty, ToolError]
