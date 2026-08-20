"""Seed the reasoning corpus (Chapter 6 labs).

Small on purpose. It contains the specific shapes the labs demonstrate:
a superseded policy pair (temporal conflict), an identifier-bearing passage
that only the sparse channel ranks first, and a two-hop supplier chain that
no single passage answers.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger
from sqlalchemy import select, text

from app.config import settings
from app.db.main import async_session
from app.db.reasoning_models import ReasoningChunk, ReasoningDocument

TENANT = "demo"
NOW = datetime.now(timezone.utc)

DOCUMENTS = [
    # (doc_id, title, authority, effective_from, superseded_by)
    ("policy-refunds-2024", "Refund Policy (2024)", 2,
     NOW - timedelta(days=600), "policy-refunds-2026"),
    ("policy-refunds-2026", "Refund Policy (2026)", 3,
     NOW - timedelta(days=90), None),
    ("po-2025-q3", "Purchase Orders, Q3 2025", 3,
     NOW - timedelta(days=240), None),
    ("recalls-2025", "Supplier Safety Recall Register 2025", 3,
     NOW - timedelta(days=120), None),
    ("supplier-map", "Supplier Dependency Map", 2,
     NOW - timedelta(days=300), None),
    ("draft-memo-refunds", "Draft memo: proposed refund changes", 0,
     NOW - timedelta(days=30), None),
]

CHUNKS = [
    # --- temporal conflict pair: same question, two answers, one current ---
    ("policy-refunds-2024", "refunds-2024#window", "Refund window",
     "Customers may request a refund for damaged goods within 14 days of "
     "delivery. Refunds are issued to the original payment method."),
    ("policy-refunds-2026", "refunds-2026#window", "Refund window",
     "Customers may request a refund for damaged goods within 30 days of "
     "delivery. This supersedes the 14-day window published in 2024. "
     "Refunds are issued to the original payment method within 5 business "
     "days of approval."),
    ("draft-memo-refunds", "memo#proposal", "Proposal",
     "We propose extending the damaged-goods refund window to 60 days. This "
     "memo is a draft and has not been approved."),

    # --- identifier-bearing passage: sparse wins, dense does not -----------
    ("po-2025-q3", "po-2025-q3#vendor", "Battery shipment",
     "Purchase order PO-2025-Q3-4471 covers 4,250 lithium cell units "
     "supplied by Meridian Cells Ltd. Unit list price 18.40 USD before the "
     "contracted 18% volume discount."),
    ("po-2025-q3", "po-2025-q3#terms", "Commercial terms",
     "A 2.5% handling surcharge applies to all orders above 2,000 units. "
     "The surcharge is applied after the volume discount."),

    # --- two-hop chain: no single passage answers the question -------------
    ("recalls-2025", "recalls-2025#meridian", "Meridian Cells Ltd",
     "Meridian Cells Ltd issued two safety recalls in 2025: recall R-2025-08 "
     "covering thermal runaway in the MC-9 series, and recall R-2025-14 "
     "covering separator defects in cells manufactured before March 2025."),
    ("recalls-2025", "recalls-2025#northwind", "Northwind Components",
     "Northwind Components reported no safety recalls in 2025."),
    ("supplier-map", "supplier-map#meridian", "Meridian Cells Ltd",
     "Meridian Cells Ltd sources separator film exclusively from Kestrel "
     "Materials. Kestrel Materials also supplies Northwind Components, our "
     "largest competitor in the industrial battery segment."),
]


async def seed_reasoning_corpus() -> None:
    async with async_session() as session:
        existing = await session.execute(
            select(ReasoningDocument.doc_id).limit(1))
        if existing.first() is not None:
            logger.info("reasoning corpus already seeded; skipping")
            return

        for doc_id, title, authority, effective, superseded in DOCUMENTS:
            session.add(ReasoningDocument(
                doc_id=doc_id, tenant_id=TENANT, title=title,
                authority=authority, effective_from=effective,
                superseded_by=superseded,
            ))
        for doc_id, chunk_id, section, content in CHUNKS:
            # Prepend document and section context BEFORE embedding: a chunk
            # that lost its heading embeds as generic policy language.
            session.add(ReasoningChunk(
                chunk_id=chunk_id, doc_id=doc_id, tenant_id=TENANT,
                section=section,
                content=f"[{section}] {content}",
                token_count=len(content.split()),
            ))
        await session.commit()
        logger.info(f"seeded {len(DOCUMENTS)} documents / {len(CHUNKS)} chunks")

    await _embed_missing()


async def _embed_missing() -> None:
    """Embed chunks that have no vector yet. Idempotent."""
    from langchain_ollama import OllamaEmbeddings

    embedder = OllamaEmbeddings(model=settings.embeddings_model_name,
                                base_url=settings.embeddings_base_url)
    async with async_session() as session:
        rows = (await session.execute(text(
            "SELECT chunk_id, content FROM reasoning_chunks "
            "WHERE embedding IS NULL"))).all()
        if not rows:
            return
        try:
            vectors = await embedder.aembed_documents([r.content for r in rows])
        except Exception as exc:
            logger.warning(f"embedding skipped (is Ollama running?): {exc}")
            return
        for row, vec in zip(rows, vectors):
            await session.execute(
                text("UPDATE reasoning_chunks SET embedding = :v "
                     "WHERE chunk_id = :c"),
                {"v": str(vec), "c": row.chunk_id})
        await session.commit()
        logger.info(f"embedded {len(rows)} chunks")


GRAPH_SEED = [
    ("Meridian Cells Ltd", "organisation",
     ["Meridian Cells", "Meridian", "MERIDIAN CELLS LTD"]),
    ("Kestrel Materials", "organisation", ["Kestrel", "Kestrel Materials Inc"]),
    ("Northwind Components", "organisation", ["Northwind"]),
]

GRAPH_EDGES = [
    ("Kestrel Materials", "SUPPLIES", "Meridian Cells Ltd",
     "sources separator film exclusively from Kestrel Materials"),
    ("Kestrel Materials", "SUPPLIES", "Northwind Components",
     "Kestrel Materials also supplies Northwind Components"),
    ("Meridian Cells Ltd", "SUPPLIES", "Acme Industrial",
     "supplied by Meridian Cells Ltd"),
]


async def seed_graph() -> None:
    """Seed the entity graph, including deliberate alias variation so the
    entity-resolution lab has something real to resolve."""
    from app.graph.client import run_cypher
    from app.graph.extractor import canonical_name

    try:
        for name, kind, aliases in GRAPH_SEED:
            await run_cypher(
                "MERGE (e:Entity {tenant_id:$t, canonical_name:$c}) "
                "SET e.name=$n, e.type=$k, e.aliases=$a",
                {"t": TENANT, "c": canonical_name(name), "n": name,
                 "k": kind, "a": aliases},
            )
        for src, rel, dst, evidence in GRAPH_EDGES:
            await run_cypher(
                f"MERGE (a:Entity {{tenant_id:$t, canonical_name:$s}}) "
                f"MERGE (b:Entity {{tenant_id:$t, canonical_name:$d}}) "
                f"MERGE (a)-[r:{rel}]->(b) SET r.evidence=$e",
                {"t": TENANT, "s": canonical_name(src),
                 "d": canonical_name(dst), "e": evidence},
            )
        logger.info("seeded graph entities and edges")
    except Exception as exc:
        logger.warning(f"graph seeding skipped (is Neo4j up?): {exc}")
