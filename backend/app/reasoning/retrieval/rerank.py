"""Reranking (Chapter 6).

Retrieval is a recall instrument; reranking is a precision instrument. In a
multi-hop loop the reranker is what keeps per-hop precision high enough that
three hops remain viable.
"""
from __future__ import annotations

import re

from loguru import logger

from app.config import settings
from app.reasoning.retrieval.loop import Passage

RERANK_PROMPT = """Rank these passages by how directly they answer the query.
Compare them against each other; relative judgements are more reliable than
absolute scores.

Query: {query}

{listing}

Return the passage numbers in descending order of relevance, comma-separated.
Include only passages that contain information relevant to the query."""


async def cross_encoder_rerank(query: str, candidates: list[Passage],
                               n: int = 6) -> list[Passage]:
    """Score query-document pairs jointly.

    Prefers a local cross-encoder when one is configured; otherwise falls
    back to a single comparative call on the small model, which is cheaper
    and lower-variance than n absolute scoring calls.
    """
    if not candidates:
        return []
    if len(candidates) <= n:
        return candidates

    if settings.reranker_model:
        try:
            return _local_rerank(query, candidates, n)
        except Exception as exc:                     # pragma: no cover
            logger.warning(f"local reranker unavailable: {exc}")

    return await _llm_rerank(query, candidates, n)


def _local_rerank(query: str, candidates: list[Passage], n: int
                  ) -> list[Passage]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.reranker_model)
    pairs = [(query, p.text) for p in candidates]
    scores = model.predict(pairs)
    for p, s in zip(candidates, scores):
        p.score = float(s)
    ranked = sorted(candidates, key=lambda p: p.score, reverse=True)
    return [p for p in ranked[:n] if p.score >= settings.rerank_score_floor]


async def _llm_rerank(query: str, candidates: list[Passage], n: int
                      ) -> list[Passage]:
    from app.reasoning.llm import critic, text_call

    pool = candidates[:30]
    listing = "\n\n".join(
        f"[{i + 1}] {p.text[:400]}" for i, p in enumerate(pool))
    text, _ = await text_call(
        RERANK_PROMPT.format(query=query, listing=listing), model=critic())

    order = [int(x) - 1 for x in re.findall(r"\b\d{1,2}\b", text)]
    seen: set[int] = set()
    ranked: list[Passage] = []
    for idx in order:
        if 0 <= idx < len(pool) and idx not in seen:
            seen.add(idx)
            ranked.append(pool[idx])
        if len(ranked) >= n:
            break
    # Backfill from the fused order if the model returned too few.
    for p in pool:
        if len(ranked) >= n:
            break
        if p not in ranked:
            ranked.append(p)
    return ranked[:n]
