"""Chapter 5: typed contracts, scoping, breakers, and SQL AST validation."""
import pytest
from pydantic import BaseModel

from app.reasoning.state import Trajectory
from app.reasoning.tools.contract import ToolEmpty, ToolError, ToolOk
from app.reasoning.tools.router import CircuitBreaker, ToolRouter, ToolSpec


class Args(BaseModel):
    q: str
    model_config = {"extra": "forbid"}


async def _ok(q: str):
    return ToolOk(data={"q": q}, source="stub")


async def _boom(q: str):
    raise RuntimeError("upstream exploded")


def _router():
    return ToolRouter([
        ToolSpec(name="good", description="d", args_model=Args, fn=_ok,
                 tags={"retrieve"}),
        ToolSpec(name="bad", description="d", args_model=Args, fn=_boom,
                 tags={"retrieve"}),
        ToolSpec(name="final_answer", description="d", args_model=Args,
                 fn=_ok, tags={"answer"}),
    ])


def test_empty_is_structurally_distinct_from_error():
    """'no matching orders' is an ANSWER; '503' is not."""
    assert ToolEmpty(hint="none").status == "empty"
    assert ToolError(kind="upstream").status == "error"
    assert ToolEmpty(hint="none").status != ToolError(kind="upstream").status


def test_error_ownership_split():
    assert ToolError(kind="bad_args").model_visible is True
    assert ToolError(kind="timeout").model_visible is False


@pytest.mark.asyncio
async def test_unknown_tool_returns_a_repairable_error():
    result = await _router().call("nope", {"q": "x"}, Trajectory())
    assert isinstance(result, ToolError) and result.kind == "bad_args"
    assert "available" in result.detail


@pytest.mark.asyncio
async def test_extra_argument_is_rejected_not_ignored():
    result = await _router().call("good", {"q": "x", "sneaky": 1},
                                  Trajectory())
    assert isinstance(result, ToolError) and result.kind == "bad_args"


@pytest.mark.asyncio
async def test_exception_never_leaks_a_stack_trace():
    result = await _router().call("bad", {"q": "x"}, Trajectory())
    assert isinstance(result, ToolError)
    assert result.detail == "RuntimeError"


def test_scoping_hides_irrelevant_tools_but_keeps_the_core_set():
    exposed = {s.name for s in _router().scope(tags={"retrieve"})}
    assert "good" in exposed and "final_answer" in exposed


def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker(threshold=2)
    breaker.record("t", ok=False)
    assert not breaker.is_open("t")
    breaker.record("t", ok=False)
    assert breaker.is_open("t")


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=2)
    breaker.record("t", ok=False)
    breaker.record("t", ok=True)
    breaker.record("t", ok=False)
    assert not breaker.is_open("t")


# --- SQL AST validation: the evasions a keyword blocklist misses ------------
sqlglot = pytest.importorskip("sqlglot")
from app.reasoning.tools.sandbox import SQLValidationError, validate_select_only


def test_plain_select_is_allowed():
    assert "SELECT" in validate_select_only(
        "SELECT id, total FROM orders WHERE total > 100").upper()


def test_delete_is_rejected():
    with pytest.raises(SQLValidationError):
        validate_select_only("DELETE FROM orders")


def test_stacked_statement_is_rejected():
    with pytest.raises(SQLValidationError, match="exactly one statement"):
        validate_select_only("SELECT 1; DROP TABLE orders")


def test_comment_obfuscated_drop_is_rejected():
    """A keyword blocklist lets this through; the AST does not."""
    with pytest.raises(SQLValidationError):
        validate_select_only("/*harmless*/ DrOp TaBlE orders")


def test_cte_select_is_allowed():
    validate_select_only(
        "WITH q AS (SELECT 1 AS n) SELECT n FROM q")
