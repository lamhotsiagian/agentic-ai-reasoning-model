"""Sandboxed execution for model-generated code and SQL (Chapter 5).

Never execute model-generated code in the application process. In production
this module shells out to a per-execution container (gVisor / Firecracker);
the in-process path below is a DEVELOPMENT fallback with an import allowlist
and hard resource caps, and it is explicitly not a security boundary.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

from loguru import logger

from app.config import settings
from app.reasoning.tools.contract import ToolError, ToolOk, ToolResult

# Modules a data-analysis snippet legitimately needs. Everything else --
# subprocess, socket, ctypes, importlib, is absent by construction.
IMPORT_ALLOWLIST = {
    "math", "statistics", "json", "re", "datetime", "decimal", "fractions",
    "itertools", "collections", "functools", "random",
    "pandas", "numpy",
}

RUNNER = r'''
import builtins, io, json, sys, resource, contextlib

ALLOWED = set(json.loads(sys.argv[1]))
LIMIT_MB = int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_AS, (LIMIT_MB * 1024 * 1024,) * 2)
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))

_real_import = builtins.__import__
def _guarded(name, *a, **kw):
    root = name.split(".")[0]
    if root not in ALLOWED:
        raise ImportError(f"import of '{name}' is not permitted in the sandbox")
    return _real_import(name, *a, **kw)
builtins.__import__ = _guarded
for unsafe in ("open", "exec", "eval", "compile", "input", "__import__"):
    if unsafe not in ("__import__",):
        setattr(builtins, unsafe, None)
builtins.__import__ = _guarded

source = sys.stdin.read()
buf = io.StringIO()
result = {"stdout": "", "error": None}
try:
    with contextlib.redirect_stdout(buf):
        scope = {"__name__": "__sandbox__"}
        exec(compile(source, "<sandbox>", "exec"), scope)
    result["stdout"] = buf.getvalue()
except BaseException as exc:
    result["stdout"] = buf.getvalue()
    result["error"] = f"{type(exc).__name__}: {exc}"
sys.stderr.write(json.dumps(result))
'''


@dataclass
class SandboxLimits:
    wall_s: float = 5.0
    memory_mb: int = 256
    max_output_chars: int = 8_000


async def run_python(code: str,
                     limits: SandboxLimits | None = None) -> ToolResult:
    """Execute a snippet out-of-process with hard resource caps.

    Network is denied by the container's network policy in production; the
    import allowlist here removes the in-process paths to it.
    """
    limits = limits or SandboxLimits()
    started = time.perf_counter()
    workdir = tempfile.mkdtemp(prefix="reasoning-sandbox-")
    runner_path = os.path.join(workdir, "_runner.py")
    with open(runner_path, "w") as fh:
        fh.write(RUNNER)

    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", runner_path,
            json.dumps(sorted(IMPORT_ALLOWLIST)), str(limits.memory_mb),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env={"PATH": "/usr/bin:/bin", "HOME": workdir,
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            _, raw = await asyncio.wait_for(
                proc.communicate(code.encode()), timeout=limits.wall_s)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolError(kind="timeout", retryable=False,
                             source="python_sandbox",
                             detail=f"execution exceeded {limits.wall_s}s")

        payload = json.loads(raw.decode() or '{"stdout": "", "error": null}')
        elapsed = int((time.perf_counter() - started) * 1000)

        if payload["error"]:
            # A semantic error: the model wrote broken code and can repair it.
            return ToolError(kind="invalid_query", retryable=False,
                             source="python_sandbox", detail=payload["error"])

        out = payload["stdout"]
        truncated = len(out) > limits.max_output_chars
        return ToolOk(
            data={"stdout": out[:limits.max_output_chars]},
            truncated=truncated,
            omitted=max(0, len(out) - limits.max_output_chars),
            source="python_sandbox", elapsed_ms=elapsed,
        )
    except Exception as exc:
        logger.exception("sandbox failed")
        return ToolError(kind="upstream", retryable=True,
                         source="python_sandbox", detail=type(exc).__name__)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# SQL: validate the AST, not a keyword blocklist.
# ---------------------------------------------------------------------------
class SQLValidationError(Exception):
    pass


def validate_select_only(sql: str) -> str:
    """Parse with a real SQL parser and reject anything that is not a single
    SELECT. Keyword blocklists are evaded by comments, casing, and string
    concatenation, and the evasions look accidental in review.
    """
    try:
        import sqlglot
        from sqlglot import expressions as exp
    except ImportError as err:                       # pragma: no cover
        raise SQLValidationError(
            "sqlglot is required for SQL validation; refusing to execute "
            "model-generated SQL without an AST check"
        ) from err

    statements = [s for s in sqlglot.parse(sql, read="postgres") if s]
    if len(statements) != 1:
        raise SQLValidationError(
            f"expected exactly one statement, parsed {len(statements)}")

    stmt = statements[0]
    if not isinstance(stmt, (exp.Select, exp.Union, exp.With)):
        raise SQLValidationError(
            f"only SELECT is permitted; parsed a {type(stmt).__name__}")

    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                 exp.Alter, exp.Command, exp.Grant)
    for node in stmt.walk():
        if isinstance(node, forbidden):
            raise SQLValidationError(
                f"statement contains a forbidden node: {type(node).__name__}")

    return stmt.sql(dialect="postgres")


async def run_sql(sql: str, tenant_id: str, *, row_limit: int = 500,
                  statement_timeout_ms: int | None = None) -> ToolResult:
    """Execute validated read-only SQL against the analytics replica.

    Access control lives in the DATABASE (row-level security bound to the
    requesting tenant), never in a WHERE clause the model was asked to add.
    """
    started = time.perf_counter()
    try:
        safe_sql = validate_select_only(sql)
    except SQLValidationError as err:
        return ToolError(kind="invalid_query", retryable=False, source="sql",
                         detail=str(err))

    timeout = statement_timeout_ms or settings.sql_statement_timeout_ms
    try:
        import asyncpg

        conn = await asyncpg.connect(settings.replica_uri)
        try:
            await conn.execute(f"SET statement_timeout = {int(timeout)}")
            await conn.execute("SET ROLE reasoning_reader")
            await conn.execute(
                "SELECT set_config('app.tenant_id', $1, true)", tenant_id)
            rows = await conn.fetch(f"SELECT * FROM ({safe_sql}) q "
                                    f"LIMIT {int(row_limit) + 1}")
        finally:
            await conn.close()
    except Exception as exc:
        return ToolError(kind="upstream", retryable=True, source="sql",
                         detail=type(exc).__name__)

    elapsed = int((time.perf_counter() - started) * 1000)
    records = [dict(r) for r in rows[:row_limit]]
    if not records:
        from app.reasoning.tools.contract import ToolEmpty
        return ToolEmpty(source="sql",
                         hint="query is valid; no rows match the filters")
    return ToolOk(data=records, row_count=len(records),
                  truncated=len(rows) > row_limit,
                  omitted=max(0, len(rows) - row_limit),
                  source=safe_sql, elapsed_ms=elapsed)
