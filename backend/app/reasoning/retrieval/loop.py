"""Iterative, sufficiency-gated hybrid retrieval (Chapter 6)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from loguru import logger

from app.reasoning.state import ReasoningStep, StepKind, Trajectory


@dataclass
class Passage:
    doc_id: str
    chunk_id: str
    text: str
    source: str                 # "vector" | "sparse" | "graph"
    score: float = 0.0
    authority: int = 0          # explicit tier; never inferred by the model
    updated_at: str | None = None
    section: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.doc_id, self.chunk_id)

    def citation(self) -> str:
        return f"{self.doc_id}#{self.section or self.chunk_id}"


@dataclass
class EvidenceSet:
    """Accumulated, deduplicated, provenance-carrying evidence."""

    passages: list[Passage] = field(default_factory=list)
    covered: set[str] = field(default_factory=set)   # sub-questions answered

    def add(self, new: list[Passage]) -> int:
        known = {p.key for p in self.passages}
        fresh = [p for p in new if p.key not in known]
        self.passages.extend(fresh)
        return len(fresh)

    def entities(self) -> set[str]:
        words: set[str] = set()
        for p in self.passages:
            words |= {w.strip(".,;:()").lower()
                      for w in p.text.split() if w[:1].isupper()}
        return words

    def citations(self) -> list[str]:
        return sorted({p.citation() for p in self.passages})

    def render(self, limit: int = 12) -> str:
        """Answer-first rendering with provenance on every fact."""
        ranked = sorted(self.passages,
                        key=lambda p: (-p.authority, -p.score))[:limit]
        return "\n\n".join(
            f"[{p.citation()}] (authority={p.authority}, "
            f"updated={p.updated_at or 'unknown'})\n{p.text}"
            for p in ranked
        )


def rrf_fuse(rankings: list[list[Passage]], k: int = 60) -> list[Passage]:
    """Reciprocal rank fusion.

    Uses ranks, not scores, so cosine similarity and BM25 never have to be
    normalised against each other, the fragile step every weighted-sum
    implementation eventually breaks on.
    """
    scores: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], Passage] = {}
    for ranking in rankings:
        for rank, p in enumerate(ranking, start=1):
            scores[p.key] = scores.get(p.key, 0.0) + 1.0 / (k + rank)
            best.setdefault(p.key, p)
    for key, score in scores.items():
        best[key].score = score
    return sorted(best.values(), key=lambda p: p.score, reverse=True)


async def hybrid_retrieve(query: str, tenant: str, k: int = 30,
                          channels: set[str] | None = None) -> list[Passage]:
    """Three channels concurrently, so latency is the max, not the sum.

    A fallback cascade would require detecting that a channel 'failed', which
    is exactly what you cannot do reliably.
    """
    from app.reasoning.retrieval.channels import (fulltext_search,
                                                  graph_search,
                                                  pgvector_search)

    channels = channels or {"vector", "sparse", "graph"}
    tasks = []
    if "vector" in channels:
        tasks.append(pgvector_search(query, tenant, k))
    if "sparse" in channels:
        tasks.append(fulltext_search(query, tenant, k))
    if "graph" in channels:
        tasks.append(graph_search(query, tenant, k))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    rankings: list[list[Passage]] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"retrieval channel failed: {r}")
            continue
        if r:
            rankings.append(r)
    if not rankings:
        return []
    return rrf_fuse(rankings)


async def retrieve_for_reasoning(question: str, tenant: str,
                                 traj: Trajectory, max_hops: int = 3,
                                 rerank_to: int = 6,
                                 channels: set[str] | None = None,
                                 use_reranker: bool = True) -> EvidenceSet:
    """Retrieve -> assess -> refine, terminating on sufficiency, not hops."""
    from app.reasoning.retrieval.rerank import cross_encoder_rerank
    from app.reasoning.retrieval.sufficiency import assess_sufficiency

    evidence = EvidenceSet()
    sub_query = question
    goal_entities = {w.strip(".,;:()").lower()
                     for w in question.split() if w[:1].isupper()}

    for hop in range(max_hops):
        candidates = await hybrid_retrieve(sub_query, tenant, k=30,
                                           channels=channels)
        # Precision per hop matters more than recall: a bad passage does not
        # merely fail to help, it generates the next (bad) query.
        top = (await cross_encoder_rerank(sub_query, candidates, n=rerank_to)
               if use_reranker else candidates[:rerank_to])
        added = evidence.add(top)

        verdict = await assess_sufficiency(question, evidence)
        step = traj.append(ReasoningStep(
            kind=StepKind.RETRIEVE,
            thought=f"hop {hop + 1}: '{sub_query[:80]}'",
            payload={"candidates": len(candidates), "kept": len(top),
                     "new": added, "sufficient": verdict.sufficient,
                     "channels": sorted(channels or {"vector", "sparse",
                                                     "graph"})},
            observation=verdict.missing or "sufficient",
        ))

        if verdict.sufficient:
            break
        if added == 0:
            # No new information: refining the same query again will not help.
            step.error = "hop added no new evidence; stopping"
            break

        next_query = verdict.next_query or sub_query
        # Hop drift: a sub-query sharing no entities with the goal has
        # wandered off. Re-anchor rather than chase it.
        next_entities = {w.strip(".,;:()").lower()
                         for w in next_query.split() if w[:1].isupper()}
        if goal_entities and next_entities and \
                not (goal_entities & next_entities) and \
                not (evidence.entities() & next_entities):
            step.error = "hop drift detected; re-anchoring to the goal"
            next_query = question
        sub_query = next_query

    return evidence
