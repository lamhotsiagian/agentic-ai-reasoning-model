"""Neo4j driver lifecycle and query helper."""
from __future__ import annotations

from typing import Any

from loguru import logger

from app.config import settings

_DRIVER: Any = None


def _driver():
    global _DRIVER
    if _DRIVER is None:
        from neo4j import AsyncGraphDatabase

        _DRIVER = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=16,
        )
    return _DRIVER


async def run_cypher(query: str, params: dict[str, Any] | None = None,
                     ) -> list[dict[str, Any]]:
    async with _driver().session(database=settings.neo4j_database) as session:
        result = await session.run(query, params or {})
        return [dict(record) async for record in result]


async def close_driver() -> None:
    global _DRIVER
    if _DRIVER is not None:
        await _DRIVER.close()
        _DRIVER = None
        logger.info("neo4j driver closed")


SCHEMA_CYPHER = [
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.tenant_id, e.canonical_name) IS UNIQUE",
    "CREATE CONSTRAINT chunk_key IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE (c.tenant_id, c.chunk_id) IS UNIQUE",
    "CREATE INDEX entity_tenant IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)",
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.aliases]",
]


async def ensure_schema() -> None:
    for statement in SCHEMA_CYPHER:
        try:
            await run_cypher(statement)
        except Exception as exc:
            logger.warning(f"graph schema statement skipped: {exc}")
