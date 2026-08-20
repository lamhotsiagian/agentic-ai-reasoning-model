"""The default tool registry for the reference project (Chapter 5).

Names describe intent, descriptions state what the tool does NOT cover, and
arguments use enums rather than free strings wherever the domain is closed.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.reasoning.tools.contract import ToolEmpty, ToolOk, ToolResult
from app.reasoning.tools.router import ToolRouter, ToolSpec
from app.reasoning.tools.sandbox import run_python, run_sql


# --------------------------------------------------------------------------
# Argument models. Enums close the domain so a constrained decode cannot
# produce "complete" where the system expects "completed".
# --------------------------------------------------------------------------
class PythonArgs(BaseModel):
    code: str = Field(description="A self-contained snippet. print() results.")

    model_config = {"extra": "forbid"}      # never silently ignore a field


class SQLReadArgs(BaseModel):
    sql: str = Field(description="A single SELECT against the read replica.")
    tenant_id: str
    row_limit: int = Field(default=200, ge=1, le=1000)

    model_config = {"extra": "forbid"}


class VectorSearchArgs(BaseModel):
    query: str
    tenant_id: str
    k: int = Field(default=8, ge=1, le=50)

    model_config = {"extra": "forbid"}


class GraphQueryArgs(BaseModel):
    entity: str
    relation: Literal["supplies", "owns", "reports_to", "mentions",
                      "depends_on", "any"] = "any"
    hops: int = Field(default=2, ge=1, le=4)
    tenant_id: str

    model_config = {"extra": "forbid"}


class FinalAnswerArgs(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------
async def _python(code: str) -> ToolResult:
    return await run_python(code)


async def _sql_read(sql: str, tenant_id: str, row_limit: int = 200) -> ToolResult:
    return await run_sql(sql, tenant_id, row_limit=row_limit)


async def _vector_search(query: str, tenant_id: str, k: int = 8) -> ToolResult:
    from app.reasoning.retrieval.channels import pgvector_search

    passages = await pgvector_search(query, tenant_id, k)
    if not passages:
        return ToolEmpty(source="pgvector",
                         hint="no passages above the similarity floor")
    return ToolOk(
        data=[{"doc_id": p.doc_id, "chunk_id": p.chunk_id,
               "text": p.text, "score": round(p.score, 4)} for p in passages],
        row_count=len(passages), source="pgvector",
    )


async def _graph_query(entity: str, tenant_id: str, relation: str = "any",
                       hops: int = 2) -> ToolResult:
    from app.reasoning.retrieval.channels import graph_neighbourhood

    paths = await graph_neighbourhood(entity, tenant_id, relation, hops)
    if not paths:
        return ToolEmpty(source="neo4j",
                         hint=f"'{entity}' has no {relation} edges within "
                              f"{hops} hops; check entity resolution")
    return ToolOk(data=paths, row_count=len(paths), source="neo4j")


async def _final_answer(answer: str, citations: list[str] | None = None
                        ) -> ToolResult:
    return ToolOk(data={"answer": answer, "citations": citations or []},
                  source="final_answer")


# --------------------------------------------------------------------------
# Registry. `tags` drive per-sub-goal scoping; `model_retryable=False` marks
# actions the model must not be allowed to repeat on its own initiative.
# --------------------------------------------------------------------------
def build_registry() -> ToolRouter:
    return ToolRouter([
        ToolSpec(
            name="python_sandbox",
            description=(
                "Execute a short Python snippet in an isolated sandbox and "
                "return its stdout. Use for ALL arithmetic, unit conversion, "
                "date maths, and statistics. Never compute in prose. "
                "Does NOT have network or filesystem access, and cannot read "
                "the database; fetch data with sql_read first."
            ),
            args_model=PythonArgs, fn=_python,
            reversibility=0, timeout_s=8.0,
            tags={"compute", "verify"},
        ),
        ToolSpec(
            name="sql_read",
            description=(
                "Run one read-only SELECT against the analytics replica and "
                "return rows. Use for structured business data: orders, "
                "customers, revenue, tickets. Does NOT write, and does NOT "
                "search free text; use search_documents for prose."
            ),
            args_model=SQLReadArgs, fn=_sql_read,
            reversibility=0, timeout_s=15.0,
            tags={"retrieve", "structured"},
        ),
        ToolSpec(
            name="search_documents",
            description=(
                "Semantic search over the tenant's document corpus; returns "
                "passages with provenance. Use for policy, contract, and "
                "narrative questions. Does NOT answer relational questions "
                "spanning entities; use query_graph for those."
            ),
            args_model=VectorSearchArgs, fn=_vector_search,
            reversibility=0, timeout_s=10.0,
            tags={"retrieve", "unstructured"},
        ),
        ToolSpec(
            name="query_graph",
            description=(
                "Traverse the entity-relationship graph from a named entity. "
                "Use for questions about connections, dependency chains, and "
                "counts over related entities. Does NOT return document text "
                "-- follow up with search_documents for supporting passages."
            ),
            args_model=GraphQueryArgs, fn=_graph_query,
            reversibility=0, timeout_s=10.0,
            tags={"retrieve", "graph"},
        ),
        ToolSpec(
            name="final_answer",
            description=(
                "Emit the final answer with its citations. Call exactly once, "
                "when the evidence supports an answer or when you must report "
                "that it does not."
            ),
            args_model=FinalAnswerArgs, fn=_final_answer,
            reversibility=0, timeout_s=5.0,
            tags={"answer"},
        ),
    ])
