"""Entity and relationship extraction with resolution (Chapter 6).

GraphRAG's dominant cost is this pipeline, and its dominant risk is entity
resolution: if 'Acme Corp.', 'Acme Corporation', and 'ACME' become three
nodes, multi-hop paths silently break and queries still return *something*.
Resolution quality is therefore a first-class metric with its own alert, not
an implementation detail of the ingestion job.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from app.graph.client import run_cypher
from app.reasoning.llm import structured_call

RelationType = Literal["SUPPLIES", "OWNS", "REPORTS_TO", "MENTIONS",
                       "DEPENDS_ON", "LOCATED_IN", "PARTY_TO"]


class Entity(BaseModel):
    name: str
    type: Literal["organisation", "person", "product", "location",
                  "document", "concept"]
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""


class Relationship(BaseModel):
    source: str
    target: str
    type: RelationType
    evidence: str = ""


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)


EXTRACT_PROMPT = """Extract entities and the relationships between them from
the passage below.

Rules:
- Only extract entities the passage actually names. Do not infer.
- Only extract a relationship if the passage states it. Quote the supporting
  span in `evidence`.
- List every surface form of a name you see in `aliases`, because alias capture
  what makes entity resolution work downstream.

Passage:
{text}"""

_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|plc|gmbh|sa|nv|ag|pty)\b\.?",
    re.I)
_PUNCT = re.compile(r"[^\w\s]")


def canonical_name(name: str) -> str:
    """Deterministic normalisation. Cheap, and it catches most duplicates
    before any fuzzy matching is needed."""
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    text = _SUFFIXES.sub("", text.lower())
    text = _PUNCT.sub(" ", text)
    return " ".join(text.split())


MERGE_ENTITY = """
MERGE (e:Entity {tenant_id: $tenant, canonical_name: $canonical})
ON CREATE SET e.name = $name, e.type = $type, e.aliases = $aliases,
              e.summary = $summary, e.created_at = timestamp()
ON MATCH  SET e.aliases = apoc.coll.toSet(coalesce(e.aliases, []) + $aliases),
              e.summary = CASE WHEN size(coalesce(e.summary,'')) < size($summary)
                               THEN $summary ELSE e.summary END
RETURN e.canonical_name AS canonical
"""

MERGE_ENTITY_FALLBACK = """
MERGE (e:Entity {tenant_id: $tenant, canonical_name: $canonical})
ON CREATE SET e.name = $name, e.type = $type, e.aliases = $aliases,
              e.summary = $summary, e.created_at = timestamp()
RETURN e.canonical_name AS canonical
"""

MERGE_CHUNK = """
MERGE (c:Chunk {tenant_id: $tenant, chunk_id: $chunk_id})
SET   c.doc_id = $doc_id, c.content = $content, c.section = $section
WITH c
MATCH (e:Entity {tenant_id: $tenant, canonical_name: $canonical})
MERGE (e)-[:MENTIONED_IN]->(c)
"""

MERGE_REL = """
MATCH (a:Entity {tenant_id: $tenant, canonical_name: $src})
MATCH (b:Entity {tenant_id: $tenant, canonical_name: $dst})
MERGE (a)-[r:%s]->(b)
SET   r.evidence = $evidence, r.doc_id = $doc_id
"""


async def extract_and_load(text: str, *, tenant: str, doc_id: str,
                           chunk_id: str, section: str | None = None
                           ) -> Extraction:
    result, _ = await structured_call(EXTRACT_PROMPT.format(text=text[:4000]),
                                      schema=Extraction)
    if not result.ok:
        logger.warning(f"extraction failed for {doc_id}#{chunk_id}: "
                       f"{result.error}")
        return Extraction()

    ext: Extraction = result.value
    canon: dict[str, str] = {}

    for entity in ext.entities:
        key = canonical_name(entity.name)
        canon[entity.name] = key
        for alias in entity.aliases:
            canon[alias] = key
        params = {"tenant": tenant, "canonical": key, "name": entity.name,
                  "type": entity.type,
                  "aliases": sorted({entity.name, *entity.aliases}),
                  "summary": entity.summary}
        try:
            await run_cypher(MERGE_ENTITY, params)
        except Exception:                     # APOC not installed
            await run_cypher(MERGE_ENTITY_FALLBACK, params)

        await run_cypher(MERGE_CHUNK, {
            "tenant": tenant, "chunk_id": chunk_id, "doc_id": doc_id,
            "content": text[:2000], "section": section, "canonical": key,
        })

    for rel in ext.relationships:
        src, dst = canon.get(rel.source), canon.get(rel.target)
        if not src or not dst:
            # A relationship whose endpoints did not resolve is dropped
            # rather than guessed: a wrong edge is worse than a missing one.
            logger.debug(f"unresolved relationship endpoints: {rel}")
            continue
        await run_cypher(MERGE_REL % rel.type, {
            "tenant": tenant, "src": src, "dst": dst,
            "evidence": rel.evidence[:500], "doc_id": doc_id,
        })

    return ext


RESOLUTION_AUDIT = """
MATCH (e:Entity {tenant_id: $tenant})
WITH e.canonical_name AS canonical, collect(e.name) AS names,
     count(*) AS n
WHERE n > 1
RETURN canonical, names, n ORDER BY n DESC LIMIT 50
"""


async def resolution_report(tenant: str) -> list[dict]:
    """Surface suspected over- and under-merging. Alert on this, do not
    treat it as an ingestion implementation detail."""
    return await run_cypher(RESOLUTION_AUDIT, {"tenant": tenant})
