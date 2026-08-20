"""Reasoning-aware retrieval (Chapter 6)."""
from app.reasoning.retrieval.loop import (EvidenceSet, Passage,
                                          hybrid_retrieve,
                                          retrieve_for_reasoning, rrf_fuse)

__all__ = ["Passage", "EvidenceSet", "rrf_fuse", "hybrid_retrieve",
           "retrieve_for_reasoning"]
