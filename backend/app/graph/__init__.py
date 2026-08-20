"""Neo4j knowledge graph: extraction, resolution, and traversal (Chapter 6)."""
from app.graph.client import close_driver, run_cypher

__all__ = ["run_cypher", "close_driver"]
