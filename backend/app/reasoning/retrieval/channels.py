"""The three retrieval channels (Chapter 6).

Dense, sparse, and graph fail on complementary inputs, and that non-overlap is
the entire argument for running all three.
"""
from __future__ import annotations

from typing import Any

from loguru import logger
from sqlalchemy import text

from app.config import settings
from app.reasoning.retrieval.loop import Passage

# ---------------------------------------------------------------------------
# Dense: pgvector. Strong on paraphrase and synonymy; weak on identifiers.
# ---------------------------------------------------------------------------
VECTOR_SQL = text("""
    SELECT c.doc_id, c.chunk_id, c.content, c.section,
           d.authority, d.updated_at,
           1 - (c.embedding <=> :qvec) AS score
    FROM   reasoning_chunks c
    JOIN   reasoning_documents d ON d.doc_id = c.doc_id
    WHERE  c.tenant_id = :tenant          -- filter IN the query, not after
      AND  d.superseded_by IS NULL
    ORDER  BY c.embedding <=> :qvec
    LIMIT  :k
""")


async def embed(query: str) -> list[float]:
    from langchain_ollama import OllamaEmbeddings

    embedder = OllamaEmbeddings(model=settings.embeddings_model_name,
                                base_url=settings.embeddings_base_url)
    return await embedder.aembed_query(query)


async def pgvector_search(query: str, tenant: str, k: int = 30
                          ) -> list[Passage]:
    from app.db.main import async_session

    try:
        qvec = await embed(query)
        async with async_session() as session:
            rows = await session.execute(
                VECTOR_SQL, {"qvec": str(qvec), "tenant": tenant, "k": k})
            return [
                Passage(doc_id=r.doc_id, chunk_id=r.chunk_id, text=r.content,
                        source="vector", score=float(r.score),
                        authority=int(r.authority or 0),
                        updated_at=str(r.updated_at) if r.updated_at else None,
                        section=r.section)
                for r in rows
            ]
    except Exception as exc:
        logger.warning(f"pgvector channel unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# Sparse: Postgres full-text. Strong on identifiers, codes, rare tokens.
# ---------------------------------------------------------------------------
FULLTEXT_SQL = text("""
    SELECT c.doc_id, c.chunk_id, c.content, c.section,
           d.authority, d.updated_at,
           ts_rank_cd(c.tsv, websearch_to_tsquery('english', :q)) AS score
    FROM   reasoning_chunks c
    JOIN   reasoning_documents d ON d.doc_id = c.doc_id
    WHERE  c.tenant_id = :tenant
      AND  d.superseded_by IS NULL
      AND  c.tsv @@ websearch_to_tsquery('english', :q)
    ORDER  BY score DESC
    LIMIT  :k
""")


async def fulltext_search(query: str, tenant: str, k: int = 30
                          ) -> list[Passage]:
    from app.db.main import async_session

    try:
        async with async_session() as session:
            rows = await session.execute(
                FULLTEXT_SQL, {"q": query, "tenant": tenant, "k": k})
            return [
                Passage(doc_id=r.doc_id, chunk_id=r.chunk_id, text=r.content,
                        source="sparse", score=float(r.score),
                        authority=int(r.authority or 0),
                        updated_at=str(r.updated_at) if r.updated_at else None,
                        section=r.section)
                for r in rows
            ]
    except Exception as exc:
        logger.warning(f"fulltext channel unavailable: {exc}")
        return []


# ---------------------------------------------------------------------------
# Graph: Neo4j. The only channel that can answer relational questions.
# ---------------------------------------------------------------------------
NEIGHBOURHOOD_CYPHER = """
MATCH path = (e:Entity {tenant_id: $tenant})-[r*1..$hops]-(n:Entity)
WHERE toLower(e.name) = toLower($entity)
  AND ($relation = 'any' OR ALL(x IN r WHERE type(x) = toUpper($relation)))
RETURN [x IN nodes(path) | x.name]  AS nodes,
       [x IN relationships(path) | type(x)] AS rels,
       n.name                        AS target,
       n.summary                     AS summary,
       [x IN nodes(path) | x.doc_id] AS doc_ids
LIMIT $k
"""

MENTION_CYPHER = """
CALL db.index.fulltext.queryNodes('entity_name_ft', $q) YIELD node, score
WHERE node.tenant_id = $tenant
MATCH (node)-[:MENTIONED_IN]->(c:Chunk)
RETURN c.doc_id AS doc_id, c.chunk_id AS chunk_id, c.content AS content,
       c.section AS section, score
ORDER BY score DESC
LIMIT $k
"""


async def graph_search(query: str, tenant: str, k: int = 30) -> list[Passage]:
    """Entity-anchored retrieval: find the entities, return their chunks."""
    from app.graph.client import run_cypher

    try:
        rows = await run_cypher(MENTION_CYPHER,
                                {"q": query, "tenant": tenant, "k": k})
        return [
            Passage(doc_id=r["doc_id"], chunk_id=r["chunk_id"],
                    text=r["content"], source="graph",
                    score=float(r["score"]), section=r.get("section"))
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"graph channel unavailable: {exc}")
        return []


async def graph_neighbourhood(entity: str, tenant: str, relation: str = "any",
                              hops: int = 2, k: int = 25
                              ) -> list[dict[str, Any]]:
    """Multi-hop traversal. This is what no passage can answer."""
    from app.graph.client import run_cypher

    try:
        rows = await run_cypher(
            NEIGHBOURHOOD_CYPHER,
            {"entity": entity, "tenant": tenant, "relation": relation,
             "hops": hops, "k": k},
        )
        return [
            {"path": _render_path(r["nodes"], r["rels"]),
             "target": r["target"], "summary": r.get("summary"),
             "doc_ids": [d for d in (r.get("doc_ids") or []) if d]}
            for r in rows
        ]
    except Exception as exc:
        logger.warning(f"graph neighbourhood unavailable: {exc}")
        return []


def _render_path(nodes: list[str], rels: list[str]) -> str:
    parts = [nodes[0]] if nodes else []
    for rel, node in zip(rels, nodes[1:]):
        parts.append(f"-[{rel}]->")
        parts.append(node)
    return " ".join(parts)
